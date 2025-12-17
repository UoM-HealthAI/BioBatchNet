"""Unified dataset for batch effect correction."""
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import scanpy as sc
from scipy.sparse import issparse
import yaml


class BatchDataset(Dataset):
    """Unified dataset for batch effect correction using AnnData format."""

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    def __init__(
        self,
        data,
        batch_labels,
        cell_types=None,
    ):
        """Create dataset from arrays.

        Args:
            data: numpy array or tensor (n_cells, n_features)
            batch_labels: batch labels (array-like)
            cell_types: optional cell type labels (array-like)
        """
        # Convert to tensors
        if isinstance(data, np.ndarray):
            self.data = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, torch.Tensor):
            self.data = data.float()
        else:
            self.data = torch.tensor(np.array(data), dtype=torch.float32)

        # Batch labels
        if isinstance(batch_labels, (pd.Categorical, pd.Series)):
            batch_labels = pd.Categorical(batch_labels).codes
        self.batch_labels = torch.tensor(np.array(batch_labels), dtype=torch.long)

        # Cell types (optional)
        if cell_types is not None:
            if isinstance(cell_types, (pd.Categorical, pd.Series)):
                cell_types = pd.Categorical(cell_types).codes
            self.cell_types = torch.tensor(np.array(cell_types), dtype=torch.long)
        else:
            self.cell_types = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.cell_types is not None:
            return self.data[idx], self.batch_labels[idx], self.cell_types[idx]
        return self.data[idx], self.batch_labels[idx]

    @classmethod
    def from_adata(
        cls,
        adata: sc.AnnData,
        batch_key: str = 'BATCH',
        cell_type_key: str = 'celltype',
        preprocess_rna: bool = False,
    ):
        """Create dataset from AnnData object.

        Args:
            adata: AnnData object
            batch_key: column name for batch labels in adata.obs
            cell_type_key: column name for cell types in adata.obs
            preprocess_rna: whether to apply RNA preprocessing
        """
        if preprocess_rna:
            adata = cls._preprocess_rna(adata)

        # Get data matrix
        if issparse(adata.X):
            data = adata.X.toarray()
        else:
            data = np.array(adata.X)

        # Get labels
        batch_labels = pd.Categorical(adata.obs[batch_key]).codes
        cell_types = None
        if cell_type_key in adata.obs.columns:
            cell_types = pd.Categorical(adata.obs[cell_type_key]).codes

        return cls(data, batch_labels, cell_types)

    @classmethod
    def from_preset(cls, name: str):
        """Load a preset dataset from presets.yaml.

        Args:
            name: preset name (e.g., 'damond', 'pancreas')
        """
        presets_path = Path(__file__).parent.parent / 'config' / 'presets.yaml'
        with open(presets_path, 'r') as f:
            presets = yaml.safe_load(f)

        # Find dataset in presets
        preset = None
        mode = None
        for m in ['imc', 'rna']:
            if name in presets.get(m, {}):
                preset = presets[m][name]
                mode = m
                break

        if preset is None:
            available = []
            for m in ['imc', 'rna']:
                available.extend(presets.get(m, {}).keys())
            raise ValueError(f"Dataset '{name}' not found. Available: {available}")

        # Load h5ad file
        data_path = cls.BASE_DIR / preset['data']
        adata = sc.read_h5ad(data_path)

        # Preprocess RNA data
        preprocess = (mode == 'rna')

        return cls.from_adata(adata, preprocess_rna=preprocess)

    @staticmethod
    def _preprocess_rna(adata: sc.AnnData) -> sc.AnnData:
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


class MLDataset(Dataset):
    """Must-link pairs dataset."""
    def __init__(self, ml_ind1, ml_ind2, data):
        self.ml_pairs = list(zip(ml_ind1, ml_ind2))
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.ml_pairs)

    def __getitem__(self, idx):
        i, j = self.ml_pairs[idx]
        sample_i = self.data[i]
        sample_j = self.data[j]
        return sample_i, sample_j


class CLDataset(Dataset):
    """Cannot-link pairs dataset."""
    def __init__(self, cl_ind1, cl_ind2, data):
        self.cl_pairs = list(zip(cl_ind1, cl_ind2))
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.cl_pairs)

    def __getitem__(self, idx):
        i, j = self.cl_pairs[idx]
        sample_i = self.data[i]
        sample_j = self.data[j]
        return sample_i, sample_j
