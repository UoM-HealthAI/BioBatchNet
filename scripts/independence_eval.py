"""
Independence Evaluation: Probe + CV R²
- Probe: Z_bio -> batch (should be random)
- CV R²: Z_bio <-> Z_batch (should be ~0)

Usage:
    python scripts/independence_eval.py --adata /home/w29632hl/code/BatchEffect/BioBatchNet/src/saved/immuncan/20260128_130444/seed_42/adata.h5ad
    python scripts/independence_eval.py --adata /home/w29632hl/code/BatchEffect/BioBatchNet/src/saved/pancreas/20260128_133804/seed_42/adata.h5ad --batch_key BATCH --cell_key celltype
"""
import argparse
import json

import numpy as np
import scanpy as sc
from tqdm import tqdm 
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score, r2_score


def probe_balacc(z, labels, desc: str, n_splits=5, seed=42):
    """Linear probe: z -> labels. Returns balacc, chance, balacc_gap."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    X = StandardScaler().fit_transform(z)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    linear = LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced')
    print(f"  - probe: {desc} (LogReg cv predict)")
    pred = cross_val_predict(linear, X, y, cv=cv)
    balacc = float(balanced_accuracy_score(y, pred))
    chance = float(1.0 / n_classes) if n_classes > 0 else 0.0
    return {
        'balacc': balacc,
        'chance': chance,
        'balacc_gap': balacc - chance,
    }


def latent_r2(src, tgt, n_splits=5, seed=42):
    """CV R²: src -> tgt. Returns mean R² (linear)."""
    X = StandardScaler().fit_transform(src)
    Y = StandardScaler().fit_transform(tgt)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def eval_reg(reg, desc: str):
        r2s = []
        for j in tqdm(range(Y.shape[1]), desc=desc, leave=False):
            r2s.append(r2_score(Y[:, j], cross_val_predict(reg, X, Y[:, j], cv=cv)))
        return np.mean(r2s)

    ridge = Ridge(alpha=1.0, random_state=seed)

    return {
        'linear_r2': eval_reg(ridge, desc='  - r2: Ridge per-dim'),
    }


def independence_eval(bio_z, batch_z, batch_labels, cell_labels=None, n_splits=5, seed=42):
    """Full independence evaluation."""
    results = {}

    # 1) Probes (balanced acc)
    print("\n[1/3] probes: balanced accuracy")
    def add_probe(prefix: str, z, labels):
        stats = probe_balacc(z, labels, desc=prefix, n_splits=n_splits, seed=seed)
        results[f'{prefix}_balacc'] = stats['balacc']
        results[f'{prefix}_chance'] = stats['chance']
        results[f'{prefix}_balacc_gap'] = stats['balacc_gap']

    add_probe('batch_from_bio', bio_z, batch_labels)
    add_probe('batch_from_batch', batch_z, batch_labels)
    if cell_labels is not None:
        add_probe('celltype_from_bio', bio_z, cell_labels)
        add_probe('celltype_from_batch', batch_z, cell_labels)

    # 2. CV R² both directions
    print("\n[2/3] r2: batch <-> bio")
    print("  - r2: batch -> bio")
    b2b = latent_r2(batch_z, bio_z, n_splits, seed)
    results['ridge_r2_batch_to_bio'] = b2b['linear_r2']

    print("  - r2: bio -> batch")
    bio2b = latent_r2(bio_z, batch_z, n_splits, seed)
    results['ridge_r2_bio_to_batch'] = bio2b['linear_r2']

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Independence evaluation from saved adata.h5ad')
    parser.add_argument('--adata', type=str, required=True, help='Path to adata.h5ad with X_biobatchnet and X_batch in obsm')
    parser.add_argument('--batch_key', type=str, default='BATCH')
    parser.add_argument('--cell_key', type=str, default='celltype')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"[load] reading adata: {args.adata}")
    adata = sc.read_h5ad(args.adata)

    print("[load] fetching embeddings + labels")
    bio_z = adata.obsm['X_biobatchnet']
    batch_z = adata.obsm['X_batch']
    batch_labels = adata.obs[args.batch_key].values
    cell_labels = adata.obs[args.cell_key].values if args.cell_key in adata.obs.columns else None

    print(f"[run] n_cells={bio_z.shape[0]} bio_dim={bio_z.shape[1]} batch_dim={batch_z.shape[1]} n_splits={args.n_splits}")
    results = independence_eval(bio_z, batch_z, batch_labels, cell_labels, args.n_splits, args.seed)

    # Print
    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    for k, v in results.items():
        print(f"{k:<30} {v:>10.4f}" if isinstance(v, float) else f"{k:<30} {v:>10}")

    # Save alongside adata
    from pathlib import Path
    out = Path(args.adata).parent / 'independence.json'
    print(f"\n[save] writing: {out}")
    with open(out, 'w') as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
    print(f"\nSaved to {out}")
