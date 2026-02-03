import os
import scanpy as sc
import scib
import numpy as np
import pandas as pd
from pathlib import Path
from config import BaselineConfig
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from utils import load_adata_from_dir, logger


def best_leiden_by_nmi(adata: sc.AnnData, label_key: str, resolutions=(0.2, 0.4, 0.6, 0.8, 1.0)):
    best = (-1.0, -1.0, None)
    y = adata.obs[label_key].values
    for r in resolutions:
        sc.tl.leiden(adata, key_added="cluster", resolution=r)
        pred = adata.obs["cluster"].values
        nmi = normalized_mutual_info_score(y, pred)
        ari = adjusted_rand_score(y, pred)
        if nmi > best[0]:
            best = (nmi, ari, r)
    return best


def subsample_data(adata, fraction, seed):
    np.random.seed(seed)
    adata = adata.copy()
    sc.pp.subsample(adata, fraction=fraction, random_state=seed)
    return adata


def evaluate(adata_dict, config, fraction, seed):
    """
    Unified evaluation function for all batch correction methods.

    Args:
        adata_dict: Dictionary mapping method names to AnnData objects
        config: BaselineConfig object containing method configurations
        fraction: Subsampling fraction for evaluation
        seed: Random seed for subsampling

    Returns:
        Dictionary of results for each method
    """
    results = {}
    batch_key = 'BATCH'
    label_key = 'celltype'

    raw_adata = adata_dict.get('Raw')
    if raw_adata is None:
        raise ValueError("adata_dict must contain 'Raw'")
    sub_raw_adata = subsample_data(raw_adata, fraction=fraction, seed=seed)
    sc.pp.pca(sub_raw_adata)
    sc.pp.neighbors(sub_raw_adata, use_rep='X_pca')

    for method_name, adata in adata_dict.items():
        if method_name == 'Raw':
            continue

        if method_name in config.methods:
            embed = config.get_method(method_name).embed
        else:
            logger.warning(f"Skip {method_name}: not in config")
            continue

        sub_adata = subsample_data(adata, fraction=fraction, seed=seed)
        sc.pp.neighbors(sub_adata, use_rep=embed)
        results[method_name] = compute_metrics(
            sub_raw_adata, sub_adata, batch_key, label_key, embed
        )

    return results


def compute_metrics(adata_raw, adata, batch_key, label_key, embed):
    """Compute batch correction and biological conservation metrics."""
    # Batch effect metrics
    ilisi = scib.metrics.ilisi_graph(adata, batch_key=batch_key, type_='embed', use_rep=embed)
    pcr = scib.me.pcr_comparison(adata_raw, adata, covariate=batch_key, embed=embed)
    graph_connectivity = scib.me.graph_connectivity(adata, label_key=label_key)
    asw_batch = scib.me.silhouette_batch(adata, batch_key=batch_key, label_key=label_key, embed=embed)

    # Biological conservation metrics
    asw_cell = scib.me.silhouette(adata, label_key=label_key, embed=embed)
    nmi, ari, _best_r = best_leiden_by_nmi(adata, label_key)

    # Aggregate scores
    batch_score = (ilisi + graph_connectivity + asw_batch + pcr) / 4
    bio_score = (asw_cell + ari + nmi) / 3
    total_score = (batch_score + bio_score) / 2

    return {
        'iLISI': ilisi,
        'GraphConn': graph_connectivity,
        'ASW_batch': asw_batch,
        'PCR': pcr,
        'BatchScore': batch_score,
        'ASW': asw_cell,
        'ARI': ari,
        'NMI': nmi,
        'BioScore': bio_score,
        'TotalScore': total_score
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Batch evaluate adata in a directory')
    parser.add_argument('--dir', '-d', type=str, required=True,
                       help='Directory containing h5ad files; metrics.csv saved here')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Config yaml (default: baseline/config.yaml)')
    parser.add_argument('--fraction', type=float, default=1.0, help='Subsample fraction')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    config_path = args.config_path or str(Path(__file__).parent / 'config.yaml')
    config = BaselineConfig.load(config_path)
    adata_dict = load_adata_from_dir(args.dir)
    if not adata_dict:
        logger.warning(f"No h5ad files in {args.dir}")
        return

    metrics = evaluate(adata_dict, config, fraction=args.fraction, seed=args.seed)
    out_path = os.path.join(args.dir, 'metrics.csv')
    pd.DataFrame(metrics).T.to_csv(out_path)
    logger.info(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
