"""Ablation study script for BioBatchNet."""
import argparse
from copy import deepcopy

import pandas as pd
import scanpy as sc
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from .config import Config
from .module import IMCModule, SeqModule
from .utils.dataset import BBNDataset
from .utils.tools import load_adata, evaluate

# Ablation configs: set weight to 0 to disable
ABLATIONS = {
    "full": {},
    "no_disc": {"loss.discriminator": 0},
    "no_clf": {"loss.classifier": 0},
    "no_ortho": {"loss.ortho": 0},
    "no_kl_bio": {"loss.kl_bio": 0},
    "no_kl_batch": {"loss.kl_batch": 0},
}


def set_nested_attr(obj, key: str, value):
    """Set nested attribute like 'loss.discriminator'."""
    parts = key.split('.')
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def run_one(config: Config, seed: int, adata, adata_raw, dataloader):
    """Single train + evaluate run."""
    pl.seed_everything(seed)

    in_sz = adata.n_vars
    num_batch = adata.obs[config.data.batch_key].nunique()

    Module = IMCModule if config.mode == 'imc' else SeqModule
    model = Module(config, in_sz, num_batch)

    trainer = pl.Trainer(
        max_epochs=config.trainer.epochs,
        callbacks=[EarlyStopping(monitor='loss', patience=config.trainer.early_stop)],
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        accelerator='auto',
    )
    trainer.fit(model, dataloader)

    bio_z, _ = model.get_embeddings(dataloader)
    adata_eval = adata.copy()
    adata_eval.obsm['X_bio'] = bio_z

    return evaluate(
        adata_eval, adata_raw,
        embed='X_bio',
        batch_key=config.data.batch_key,
        label_key=config.data.cell_type_key,
    )


def run_ablation(preset: str, seeds: list[int] = [42, 123, 456], ablations: dict = None):
    """Run all ablation experiments."""
    ablations = ablations or ABLATIONS
    cfg = Config.load(preset)

    # Load data once
    adata = sc.read_h5ad(cfg.data.path)
    adata_raw = adata.copy()
    data, batch_labels, cell_types = load_adata(
        str(cfg.data.path), cfg.mode,
        preprocess=cfg.data.preprocess if cfg.data.preprocess is not None else (cfg.mode == 'seq'),
        batch_key=cfg.data.batch_key,
        cell_type_key=cfg.data.cell_type_key,
    )
    dataset = BBNDataset(data, batch_labels, cell_types)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.trainer.batch_size, shuffle=False, num_workers=4,
    )

    results = []
    for name, overrides in ablations.items():
        for seed in seeds:
            c = Config.load(preset)
            c.name = f"{preset}_{name}"
            for k, v in overrides.items():
                set_nested_attr(c, k, v)

            metrics = run_one(c, seed, adata.copy(), adata_raw, dataloader)
            metrics['ablation'] = name
            metrics['seed'] = seed
            results.append(metrics)
            print(f"{name} seed={seed}: Total={metrics['TotalScore']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(f'ablation_{preset}.csv', index=False)

    summary = df.groupby('ablation')[['TotalScore', 'BatchScore', 'BioScore']].agg(['mean', 'std'])
    print("\n=== Summary ===")
    print(summary.round(4))

    return df


def main():
    parser = argparse.ArgumentParser(description='BioBatchNet Ablation Study')
    parser.add_argument('--preset', type=str, required=True, help='Preset name (e.g., damond)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456], help='Random seeds')
    args = parser.parse_args()

    run_ablation(args.preset, args.seeds)


if __name__ == '__main__':
    main()
