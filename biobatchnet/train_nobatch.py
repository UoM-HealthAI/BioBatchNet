"""Ablation study: train without batch encoder."""
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
from .module import NOModule
from .utils.dataset import BBNDataset
from .utils.tools import load_adata, evaluate, independence_metrics

SAVE_ROOT = Path(__file__).parent / 'saved'


def train(config: Config, seed: int = 42, run_name: str = None, do_eval: bool = True):
    pl.seed_everything(seed)
    config.seed = seed

    run_name = run_name or datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = SAVE_ROOT / f'{config.preset}_nobatch' / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_dir = run_dir / f'seed_{seed}'
    save_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(run_dir / 'config.yaml')

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

    model = NOModule(config, in_sz, num_batch, save_dir=save_dir)

    callbacks = [
        EarlyStopping(monitor='loss', patience=config.trainer.early_stop, mode='min'),
    ]

    logger = WandbLogger(
        project='biobatchnet',
        name=f'{config.preset}_nobatch_{run_name}_seed{seed}',
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

    # Get embeddings
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=config.trainer.batch_size, shuffle=False)
    z_bio, _ = model.get_embeddings(eval_loader)
    adata.obsm['X_biobatchnet'] = z_bio

    eval_metrics = {}
    if do_eval:
        eval_metrics = evaluate(
            adata, adata_raw,
            embed='X_biobatchnet',
            batch_key=config.data.batch_key,
            label_key=config.data.cell_type_key,
            fraction=config.trainer.sampling_fraction,
        )
        for k, v in eval_metrics.items():
            print(f"{k}: {v:.4f}")

        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(eval_metrics, f, indent=2)

    return model, adata, eval_metrics


def main():
    parser = argparse.ArgumentParser(description='BioBatchNet Ablation: No Batch Encoder')
    parser.add_argument('--config', type=str, required=True, help='Config YAML path or preset name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--run_name', type=str, default=None, help='Run name (save dir + wandb)')
    parser.add_argument('--no-eval', action='store_true', help='Skip evaluation metrics')
    args = parser.parse_args()

    config = Config.load(args.config)
    train(config, args.seed, run_name=args.run_name, do_eval=not args.no_eval)


if __name__ == '__main__':
    main()
