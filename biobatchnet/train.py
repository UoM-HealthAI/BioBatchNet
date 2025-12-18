"""Training script for BioBatchNet using PyTorch Lightning."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import scanpy as sc
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from .config import Config
from .module import IMCModule, SeqModule
from .utils.dataset import BBNDataset
from .utils.tools import load_adata, evaluate, independence_metrics

SAVE_ROOT = Path(__file__).parent / 'saved'


def train(config: Config, seed: int = 42):
    pl.seed_everything(seed)
    config.seed = seed

    # Setup save directory: saved/{name}/{timestamp}/seed_{seed}/
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = SAVE_ROOT / config.name / timestamp / f'seed_{seed}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = config.data.path
    preprocess = config.data.preprocess if config.data.preprocess is not None else (config.mode == 'seq')

    adata = sc.read_h5ad(str(data_path))
    adata_raw = adata.copy()

    data, batch_labels, cell_types = load_adata(
        str(data_path),
        data_type=config.mode,
        preprocess=preprocess,
        batch_key=config.data.batch_key,
        cell_type_key=config.data.cell_type_key,
    )

    # Infer model dimensions from data
    in_sz = data.shape[1]
    num_batch = len(np.unique(batch_labels))

    dataset = BBNDataset(data, batch_labels, cell_types)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    Module = IMCModule if config.mode == 'imc' else SeqModule
    model = Module(config, in_sz, num_batch, save_dir=save_dir)

    callbacks = [
        EarlyStopping(monitor='loss', patience=config.trainer.early_stop, mode='min'),
    ]

    logger = WandbLogger(
        project='biobatchnet',
        name=f'{config.name}_{timestamp}_seed{seed}',
    )

    trainer = pl.Trainer(
        max_epochs=config.trainer.epochs,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator='auto',
    )

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable_params:", n)

    trainer.fit(model, dataloader)

    # Evaluate
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=config.trainer.batch_size, shuffle=False)
    z_bio, z_batch = model.get_embeddings(eval_loader)
    adata.obsm['X_biobatchnet'] = z_bio

    # Evaluation metrics
    eval_metrics = evaluate(
        adata, adata_raw,
        embed='X_biobatchnet',
        batch_key=config.data.batch_key,
        label_key=config.data.cell_type_key,
        fraction=config.trainer.sampling_fraction,
    )
    for k, v in eval_metrics.items():
        print(f"{k}: {v:.4f}")

    # Independence metrics
    inde_metrics = independence_metrics(z_bio, z_batch)
    for k, v in inde_metrics.items():
        print(f"{k}: {v:.4f}")

    # Save metrics
    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump({'eval': eval_metrics, 'independence': inde_metrics}, f, indent=2)

    return model, eval_metrics, inde_metrics


def main():
    parser = argparse.ArgumentParser(description='BioBatchNet Training')
    parser.add_argument('--config', type=str, required=True, help='Config YAML path or preset name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    config = Config.load(args.config)
    train(config, args.seed)


if __name__ == '__main__':
    main()
