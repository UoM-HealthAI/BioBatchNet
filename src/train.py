import argparse
import json
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import scanpy as sc
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from .config import Config
from .module import IMCModule, SeqModule
from .utils.dataset import BBNDataset
from .utils.tools import load_adata, evaluate, independence_metrics, visualization_with_leiden

SAVE_ROOT = Path(__file__).parent / 'saved'


def set_nested_attr(obj, key: str, value):
    """Set nested attribute like 'loss.discriminator'."""
    parts = key.split('.')
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def train(config: Config, seed: int = 42, run_name: Optional[str] = None, do_eval: bool = True, devices: int = 1):
    pl.seed_everything(seed, workers=True)
    config.seed = seed

    timestamp = os.environ.get("BIOBATCHNET_RUN_TS")
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.environ["BIOBATCHNET_RUN_TS"] = timestamp

    # Combine timestamp with run_name
    if run_name is None:
        run_name = timestamp
    else:
        run_name = f"{timestamp}_{run_name}"

    run_dir = SAVE_ROOT / config.preset / run_name
    save_dir = run_dir / f'seed_{seed}'

    # Load data
    data_path = config.data.path
    preprocess = config.data.preprocess if config.data.preprocess is not None else (config.mode == 'seq')

    adata = sc.read_h5ad(str(data_path))
    adata_raw = adata.copy()

    data, batch_labels, cell_types, batch_names, cell_type_names = load_adata(
        str(data_path),
        data_type=config.mode,
        preprocess=preprocess,
        batch_key=config.data.batch_key,
        cell_type_key=config.data.cell_type_key,
    )

    in_sz = data.shape[1]
    num_batch = len(np.unique(batch_labels))

    dataset = BBNDataset(data, batch_labels, cell_types, batch_names, cell_type_names)
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
        name=f'{config.preset}_{run_name}_seed{seed}',
    )

    trainer = pl.Trainer(
        max_epochs=config.trainer.epochs,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator='auto',
        strategy='ddp' if devices > 1 else 'auto',
        devices=devices,
        deterministic=True,
        # gradient_clip_val=1.0,
        # gradient_clip_algorithm='norm',
    )

    # In DDP, every rank may hit filesystem writes later; ensure dirs exist everywhere.
    run_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    if trainer.is_global_zero:
        config.to_yaml(run_dir / 'config.yaml')

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable_params:", n)

    trainer.fit(model, dataloader)

    eval_metrics, inde_metrics = {}, {}
    if trainer.is_global_zero:
        # Get embeddings / eval only once to avoid multi-rank conflicts.
        eval_loader = torch.utils.data.DataLoader(dataset, batch_size=config.trainer.batch_size, shuffle=False)
        z_bio, z_batch = model.get_embeddings(eval_loader)
        adata.obsm['X_biobatchnet'] = z_bio
        adata.obsm['X_batch'] = z_batch

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

            # Visualization with leiden clustering (also adds leiden to adata.obs)
            visualization_with_leiden(
                adata,
                save_path=save_dir / 'umap.png',
                batch_key=config.data.batch_key,
                label_key=config.data.cell_type_key,
                seed=seed,
            )

        # Save adata after visualization so it includes leiden results
        adata.write(save_dir / 'biobatchnet.h5ad')

        # Cast numpy/torch scalars to Python floats for serialization
        eval_metrics = {k: float(v) for k, v in eval_metrics.items()}

        # Also save a simple CSV row where metrics dicts are stored as strings
        csv_path = save_dir / 'metrics.csv'
        write_header = not csv_path.exists()
        with open(csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['preset', 'run_name', 'seed', 'eval'])
            if write_header:
                w.writeheader()
            w.writerow({
                'preset': config.preset,
                'run_name': run_name,
                'seed': seed,
                'eval': json.dumps(eval_metrics),
            })


    return model, adata, eval_metrics, inde_metrics


def main():
    parser = argparse.ArgumentParser(description='BioBatchNet Training')
    parser.add_argument('--config', type=str, required=True, help='Config YAML path or preset name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--run_name', type=str, default=None, help='Run name (save dir + wandb)')
    parser.add_argument('--no-eval', action='store_true', help='Skip evaluation metrics')
    parser.add_argument('--devices', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--batch_size', '--bs', type=int, default=None, help='Batch size per GPU')
    parser.add_argument('--use_bn', action='store_true', help='Use BatchNorm in ResBlock')

    # Config overrides (e.g., --loss.discriminator 0)
    parser.add_argument('--loss.recon', type=float, default=None)
    parser.add_argument('--loss.discriminator', type=float, default=None)
    parser.add_argument('--loss.classifier', type=float, default=None)
    parser.add_argument('--loss.ortho', type=float, default=None)
    parser.add_argument('--loss.kl_bio', type=float, default=None)
    parser.add_argument('--loss.kl_batch', type=float, default=None)
    parser.add_argument('--trainer.epochs', type=int, default=None)
    parser.add_argument('--trainer.batch_size', type=int, default=None)
    parser.add_argument('--trainer.lr', type=float, default=None)

    args = parser.parse_args()

    config = Config.load(args.config)

    # Apply overrides
    if args.use_bn:
        config.model.use_bn = True

    overrides = {
        'loss.recon': getattr(args, 'loss.recon'),
        'loss.discriminator': getattr(args, 'loss.discriminator'),
        'loss.classifier': getattr(args, 'loss.classifier'),
        'loss.ortho': getattr(args, 'loss.ortho'),
        'loss.kl_bio': getattr(args, 'loss.kl_bio'),
        'loss.kl_batch': getattr(args, 'loss.kl_batch'),
        'trainer.epochs': getattr(args, 'trainer.epochs'),
        'trainer.batch_size': args.batch_size or getattr(args, 'trainer.batch_size'),
        'trainer.lr': getattr(args, 'trainer.lr'),
    }
    for key, value in overrides.items():
        if value is not None:
            set_nested_attr(config, key, value)

    train(config, args.seed, run_name=args.run_name, do_eval=not args.no_eval, devices=args.devices)


if __name__ == '__main__':
    main()


"""
python -m src.train --config pancreas --devices 1 --bs 256 --seed 43
"""