import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
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


def load_preset(name: str):
    """Load preset config and data path from presets.yaml."""
    presets_path = Path(__file__).resolve().parent.parent / 'config' / 'presets.yaml'
    with open(presets_path, 'r') as f:
        presets = yaml.safe_load(f)

    for mode in ['imc', 'rna']:
        if name in presets.get(mode, {}):
            preset = presets[mode][name]
            preset['mode'] = mode
            return preset

    available = []
    for mode in ['imc', 'rna']:
        available.extend(presets.get(mode, {}).keys())
    raise ValueError(f"Dataset '{name}' not found. Available: {available}")


def scRNA_preprocess(adata: sc.AnnData) -> sc.AnnData:
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

    if data_type == 'rna' and preprocess:
        adata = scRNA_preprocess(adata)

    if issparse(adata.X):
        data = adata.X.toarray()
    else:
        data = np.array(adata.X)

    batch_labels = pd.Categorical(adata.obs[batch_key]).codes
    cell_types = pd.Categorical(adata.obs[cell_type_key]).codes if cell_type_key in adata.obs.columns else None

    return data, batch_labels, cell_types
