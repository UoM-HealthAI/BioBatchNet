import scanpy as sc
import scib
import numpy as np
from config import BaselineConfig


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
        need_pca = method_cfg.need_pca

        # Subsample and prepare
        sub_adata = subsample_data(adata, fraction=fraction, seed=seed)

        # Apply PCA if needed
        if need_pca:
            sc.pp.pca(sub_adata)

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

    # Clustering evaluation
    scib.me.cluster_optimal_resolution(adata, cluster_key="cluster", label_key="celltype")
    ari = scib.me.ari(adata, cluster_key="cluster", label_key="celltype")
    nmi = scib.me.nmi(adata, cluster_key="cluster", label_key="celltype")

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
