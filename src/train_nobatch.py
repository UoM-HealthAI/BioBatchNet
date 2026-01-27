"""Ablation study: train without batch encoder."""
import argparse
import json
import csv
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import scanpy as sc
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from .config import Config
from .module import NOBatchModule
from .utils.dataset import BBNDataset
from .utils.tools import load_adata, evaluate

SAVE_ROOT = Path(__file__).parent / 'saved'


def train(config: Config, seed: int = 42, run_name: str = None, do_eval: bool = True, devices: int = 1):
    pl.seed_everything(seed, workers=True)
    config.seed = seed

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if run_name is None:
        run_name = timestamp
    else:
        run_name = f"{timestamp}_{run_name}"
    run_dir = SAVE_ROOT / f'{config.preset}_nobatch' / run_name
    save_dir = run_dir / f'seed_{seed}'

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

    model = NOBatchModule(config, in_sz, num_batch, save_dir=save_dir)

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
        devices=devices,
        deterministic=True,
    )

    # Ensure dirs exist for all ranks; only global zero writes configs/metrics.
    run_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    if trainer.is_global_zero:
        config.to_yaml(run_dir / 'config.yaml')

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable_params:", n)

    trainer.fit(model, dataloader)

    eval_metrics = {}
    if trainer.is_global_zero:
        # Get embeddings / eval only once to avoid multi-rank conflicts.
        eval_loader = torch.utils.data.DataLoader(dataset, batch_size=config.trainer.batch_size, shuffle=False)
        z_bio, _ = model.get_embeddings(eval_loader)
        adata.obsm['X_biobatchnet'] = z_bio

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

        # Cast numpy/torch scalars to Python floats for serialization
        eval_metrics = {k: float(v) for k, v in eval_metrics.items()}

        # Also save a simple CSV row where metrics dict is stored as a string
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

            with open(save_dir / 'metrics.json', 'w') as f:
                json.dump(eval_metrics, f, indent=2)

    return model, adata, eval_metrics


def main():
    parser = argparse.ArgumentParser(description='BioBatchNet Ablation: No Batch Encoder')
    parser.add_argument('--config', type=str, required=True, help='Config YAML path or preset name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--run_name', type=str, default=None, help='Run name (save dir + wandb)')
    parser.add_argument('--no-eval', action='store_true', help='Skip evaluation metrics')
    parser.add_argument('--devices', type=int, default=1, help='Number of GPUs to use')
    args = parser.parse_args()

    config = Config.load(args.config)
    train(config, args.seed, run_name=args.run_name, do_eval=not args.no_eval, devices=args.devices)


if __name__ == '__main__':
    main()


"""
python -u -m src.train_nobatch --config macaque --seed 42 --run_name nobatch 
"""