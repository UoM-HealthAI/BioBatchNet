import scanpy as sc
import scib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import issparse


def visualization(save_dir, adata, emb, epoch):
    """UMAP visualization of embeddings."""
    sc.pp.subsample(adata, fraction=0.3)
    sc.pp.neighbors(adata, use_rep=emb)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=['BATCH', 'celltype'], frameon=False)
    plt.savefig(f'{save_dir}/{emb}_{epoch}_umap.png')
    plt.close()


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

    if issparse(adata.X):
        data = adata.X.toarray()
    else:
        data = np.array(adata.X)

    batch_labels = pd.Categorical(adata.obs[batch_key]).codes
    cell_types = pd.Categorical(adata.obs[cell_type_key]).codes if cell_type_key in adata.obs.columns else None

    return data, batch_labels, cell_types


def evaluate(
    adata,
    adata_raw,
    embed='X_biobatchnet',
    batch_key='BATCH',
    label_key='celltype',
):
    sc.pp.neighbors(adata, use_rep=embed)
    sc.pp.pca(adata_raw)
    sc.pp.neighbors(adata_raw, use_rep='X_pca')

    # Batch correction metrics
    ilisi = scib.metrics.ilisi_graph(adata, batch_key=batch_key, type_='embed', use_rep=embed)
    graph_conn = scib.me.graph_connectivity(adata, label_key=label_key)
    asw_batch = scib.me.silhouette_batch(adata, batch_key=batch_key, label_key=label_key, embed=embed)
    pcr = scib.me.pcr_comparison(adata_raw, adata, covariate=batch_key, embed=embed)

    # Biological conservation metrics
    asw_cell = scib.me.silhouette(adata, label_key=label_key, embed=embed)
    scib.me.cluster_optimal_resolution(adata, cluster_key='cluster', label_key=label_key)
    ari = scib.me.ari(adata, cluster_key='cluster', label_key=label_key)
    nmi = scib.me.nmi(adata, cluster_key='cluster', label_key=label_key)

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
