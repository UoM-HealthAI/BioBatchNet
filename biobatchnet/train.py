"""Unified training script for BioBatchNet."""
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
import yaml

from .config import Config
from .models.model import IMCVAE, GeneVAE
from .utils.dataset import BBNDataset
from .utils.trainer import Trainer
from .utils.util import set_random_seed


BASE_DIR = Path(__file__).resolve().parent.parent


def load_preset(name: str):
    """Load preset config and data path from presets.yaml."""
    presets_path = Path(__file__).parent / 'config' / 'presets.yaml'
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


def load_adata(path: str, preprocess: bool = False, batch_key: str = 'BATCH', cell_type_key: str = 'celltype'):
    """Load AnnData and extract data, batch_labels, cell_types."""
    adata = sc.read_h5ad(path)

    if preprocess:
        adata = scRNA_preprocess(adata)

    if issparse(adata.X):
        data = adata.X.toarray()
    else:
        data = np.array(adata.X)

    batch_labels = pd.Categorical(adata.obs[batch_key]).codes
    cell_types = pd.Categorical(adata.obs[cell_type_key]).codes if cell_type_key in adata.obs.columns else None

    return data, batch_labels, cell_types


def train(config: Config, seed: int = 42):
    """Train model with given config and seed."""
    set_random_seed(seed)
    config.seed = seed

    # Load data from preset
    preset = load_preset(config.name)
    data_path = BASE_DIR / preset['data']
    preprocess = (preset['mode'] == 'rna')

    data, batch_labels, cell_types = load_adata(data_path, preprocess=preprocess)

    # Create dataset
    dataset = BBNDataset(data, batch_labels, cell_types)

    # Select model based on mode
    if config.mode == 'imc':
        model = IMCVAE(config.model)
    else:
        model = GeneVAE(config.model)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Train
    trainer = Trainer(
        config=config,
        model=model,
        train_dataloader=dataloader,
        eval_dataloader=dataloader,
        device=device,
    )
    trainer.train()

    return trainer


def main():
    parser = argparse.ArgumentParser(description='BioBatchNet Training')
    parser.add_argument('--preset', type=str, help='Dataset preset (damond, pancreas, etc.)')
    parser.add_argument('--config', type=str, help='Path to config yaml file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    if args.preset:
        config = Config.from_preset(args.preset)
    elif args.config:
        config = Config.from_yaml(args.config)
    else:
        parser.error('Either --preset or --config is required')

    train(config, args.seed)


if __name__ == '__main__':
    main()
