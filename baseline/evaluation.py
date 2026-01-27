import scanpy as sc
import scib
import numpy as np
from config import BaselineConfig
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


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

    # Prepare raw adata reference
    raw_adata = adata_dict.get('Raw')
    sub_raw_adata = subsample_data(raw_adata, fraction=fraction, seed=seed)
    sc.pp.pca(sub_raw_adata)
    sc.pp.neighbors(sub_raw_adata, use_rep='X_pca')

    for method_name, adata in adata_dict.items():
        if method_name == 'Raw':
            continue

        # Get method config
        method_cfg = config.get_method(method_name)
        embed = method_cfg.embed

        # Subsample and prepare
        sub_adata = subsample_data(adata, fraction=fraction, seed=seed)

        # Compute neighbors
        sc.pp.neighbors(sub_adata, use_rep=embed)

        # Compute metrics
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
