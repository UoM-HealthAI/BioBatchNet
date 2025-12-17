"""Training script for BioBatchNet using PyTorch Lightning."""
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from .config import Config
from .module import IMCModule, SeqModule
from .utils.dataset import BBNDataset
from .utils.tools import load_adata


def train(config: Config, seed: int = 42):
    pl.seed_everything(seed)
    config.seed = seed

    # Load data
    data_path = config.data.path
    preprocess = config.data.preprocess if config.data.preprocess is not None else (config.mode == 'seq')

    data, batch_labels, cell_types = load_adata(
        str(data_path),
        data_type=config.mode,
        preprocess=preprocess,
        batch_key=config.data.batch_key,
        cell_type_key=config.data.cell_type_key,
    )

    dataset = BBNDataset(data, batch_labels, cell_types)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    Module = IMCModule if config.mode == 'imc' else SeqModule
    model = Module(config)

    callbacks = [
        EarlyStopping(monitor='loss', patience=config.trainer.early_stop, mode='min'),
        ModelCheckpoint(
            dirpath=f'./saved/{config.name}/seed_{seed}',
            filename='best',
            monitor='loss',
            mode='min',
            save_top_k=1,
        ),
    ]

    logger = WandbLogger(
        project='biobatchnet',
        name=f'{config.name}_seed{seed}',
    )

    trainer = pl.Trainer(
        max_epochs=config.trainer.epochs,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        accelerator='auto',
    )

    trainer.fit(model, dataloader)

    return model


def main():
    parser = argparse.ArgumentParser(description='BioBatchNet Training')
    parser.add_argument('--config', type=str, required=True, help='Config YAML path or preset name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    config = Config.load(args.config)
    train(config, args.seed)


if __name__ == '__main__':
    main()
