import scanpy as sc
import scib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import anndata as ad


def visualization(embedding, batch_labels, cell_types, save_path, batch_key='BATCH', label_key='celltype'):
    adata = ad.AnnData(embedding)
    adata.obs[batch_key] = pd.Categorical(batch_labels)
    adata.obs[label_key] = pd.Categorical(cell_types)

    adata = sc.pp.subsample(adata, fraction=0.3, random_state=42, copy=True)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=[batch_key, label_key], frameon=False)
    plt.savefig(save_path)
    plt.close()


def best_leiden_by_nmi(adata: sc.AnnData, label_key: str, resolutions=(0.2, 0.4, 0.6, 0.8, 1.0)):
    best = (-1.0, -1.0, None) 
    y = adata.obs[label_key].values
    for r in resolutions:
        sc.tl.leiden(adata, key_added='cluster', resolution=r)
        pred = adata.obs['cluster'].values
        nmi = normalized_mutual_info_score(y, pred)
        ari = adjusted_rand_score(y, pred)
        if nmi > best[0]:
            best = (nmi, ari, r)
    return best


def seq_preprocess(adata: sc.AnnData) -> sc.AnnData:
    """Standard preprocessing for scRNA-seq data."""
    adata = adata.copy()

    if issparse(adata.X):
        adata.X = adata.X.toarray()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3', subset=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


def load_adata(path: str, data_type: str, preprocess: bool = False, batch_key: str = 'BATCH', cell_type_key: str = 'celltype'):
    """Load AnnData and extract data, batch_labels, cell_types."""
    adata = sc.read_h5ad(path)

    if data_type == 'seq' and preprocess:
        adata = seq_preprocess(adata)

    data = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
    
    batch_labels = pd.Categorical(adata.obs[batch_key]).codes
    cell_types = pd.Categorical(adata.obs[cell_type_key]).codes if cell_type_key in adata.obs.columns else None
    return data, batch_labels, cell_types


def evaluate(
    adata,
    adata_raw,
    embed='X_biobatchnet',
    batch_key='BATCH',
    label_key='celltype',
    fraction=1.0,
):
    adata_sub = sc.pp.subsample(adata, fraction=fraction, random_state=42, copy=True)
    adata_raw_sub = adata_raw[adata_sub.obs_names].copy()

    sc.pp.neighbors(adata_sub, use_rep=embed)
    sc.pp.pca(adata_raw_sub)
    sc.pp.neighbors(adata_raw_sub, use_rep='X_pca')

    # Batch correction metrics
    ilisi = scib.metrics.ilisi_graph(adata_sub, batch_key=batch_key, type_='embed', use_rep=embed)
    graph_conn = scib.me.graph_connectivity(adata_sub, label_key=label_key)
    asw_batch = scib.me.silhouette_batch(adata_sub, batch_key=batch_key, label_key=label_key, embed=embed)
    pcr = scib.me.pcr_comparison(adata_raw_sub, adata_sub, covariate=batch_key, embed=embed)

    # Biological conservation metrics
    asw_cell = scib.me.silhouette(adata_sub, label_key=label_key, embed=embed)
    sc.tl.leiden(adata_sub, key_added='cluster', resolution=0.4)
    nmi = normalized_mutual_info_score(adata_sub.obs[label_key].values, adata_sub.obs['cluster'].values)
    ari = adjusted_rand_score(adata_sub.obs[label_key].values, adata_sub.obs['cluster'].values)

    # Aggregate scores
    batch_score = (ilisi + graph_conn + asw_batch + pcr) / 4
    bio_score = (asw_cell + ari + nmi) / 3
    total_score = (batch_score + bio_score) / 2

    return {
        'iLISI': ilisi,
        'GraphConn': graph_conn,
        'ASW_batch': asw_batch,
        'PCR': pcr,
        'BatchScore': batch_score,
        'ASW': asw_cell,
        'ARI': ari,
        'NMI': nmi,
        'BioScore': bio_score,
        'TotalScore': total_score,
    }


def independence_metrics(bio_z: np.ndarray, batch_z: np.ndarray) -> dict:
    corrs = []
    for i in range(bio_z.shape[1]):
        for j in range(batch_z.shape[1]):
            r, _ = pearsonr(bio_z[:, i], batch_z[:, j])
            corrs.append(abs(r))
    mean_corr = np.mean(corrs)
    max_corr = np.max(corrs)

    # Mutual information: estimate MI between batch_z and each dimension of bio_z
    mi_scores = []
    for i in range(bio_z.shape[1]):
        mi = mutual_info_regression(batch_z, bio_z[:, i], random_state=42)
        mi_scores.append(mi.mean())
    mean_mi = np.mean(mi_scores)

    return {
        'mean_abs_corr': mean_corr,
        'max_abs_corr': max_corr,
        'mean_MI': mean_mi,
    }
