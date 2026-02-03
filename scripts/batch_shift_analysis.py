#!/usr/bin/env python3
"""
Batch Effect Analysis: iLISI and ASW_batch

Standard metrics for evaluating batch effects:
1. iLISI: Higher = better mixing (less batch effect)
2. ASW_batch: Lower = better mixing (less batch effect)
"""

import argparse
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from collections import Counter


def compute_lisi(X, labels, perplexity=30):
    """Compute Local Inverse Simpson's Index (LISI)."""
    n_cells = X.shape[0]
    k = min(perplexity * 3, n_cells - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    indices = indices[:, 1:]

    unique_labels = np.unique(labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_idx[l] for l in labels])

    lisi_scores = []
    for i in range(n_cells):
        neighbor_labels = numeric_labels[indices[i]]
        label_counts = Counter(neighbor_labels)
        total = sum(label_counts.values())
        simpson = sum((count / total) ** 2 for count in label_counts.values())
        lisi_scores.append(1.0 / simpson if simpson > 0 else 1.0)

    return np.array(lisi_scores)


def analyze_batch_effect(adata, celltype_col='celltype', batch_col='BATCH', perplexity=30):
    """Compute iLISI and ASW_batch for each cell type."""
    celltypes = sorted(adata.obs[celltype_col].unique())
    results = []

    for ct in celltypes:
        mask = adata.obs[celltype_col] == ct
        X = adata[mask].X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        batch_labels = adata.obs.loc[mask, batch_col].values
        n_cells = X.shape[0]

        if n_cells < 10 or len(np.unique(batch_labels)) < 2:
            continue

        # iLISI
        perp = min(perplexity, n_cells // 3)
        ilisi = compute_lisi(X, batch_labels, perplexity=perp)

        # ASW_batch
        asw_samples = silhouette_samples(X, batch_labels, metric='euclidean')

        print(f"  {ct:15s}: iLISI={np.mean(ilisi):.3f}, ASW_batch={np.mean(asw_samples):.3f}, n={n_cells}")

        results.append({
            'celltype': ct, 'n_cells': n_cells,
            'iLISI_mean': np.mean(ilisi), 'iLISI_std': np.std(ilisi),
            'ASW_batch_mean': np.mean(asw_samples), 'ASW_batch_std': np.std(asw_samples),
        })

    return pd.DataFrame(results)


def plot_metrics(df, output_path):
    """Plot iLISI and ASW_batch barplots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # iLISI (ascending = more batch effect)
    df_sorted = df.sort_values('iLISI_mean', ascending=True)
    ax = axes[0]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df_sorted)))
    ax.barh(df_sorted['celltype'], df_sorted['iLISI_mean'],
            xerr=df_sorted['iLISI_std'], capsize=3, color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='No mixing')
    ax.set_xlabel('iLISI (higher = better mixing)', fontsize=11)
    ax.set_ylabel('Cell Type', fontsize=11)
    ax.set_title('iLISI by Cell Type', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ASW_batch (descending = more batch effect)
    df_sorted = df.sort_values('ASW_batch_mean', ascending=False)
    ax = axes[1]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_sorted)))
    ax.barh(df_sorted['celltype'], df_sorted['ASW_batch_mean'],
            xerr=df_sorted['ASW_batch_std'], capsize=3, color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('ASW_batch (lower = better mixing)', fontsize=11)
    ax.set_ylabel('Cell Type', fontsize=11)
    ax.set_title('ASW_batch by Cell Type', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch Effect Analysis (iLISI & ASW_batch)')
    parser.add_argument('--data', type=str, required=True, help='Path to h5ad file')
    parser.add_argument('--celltype_col', type=str, default='celltype')
    parser.add_argument('--batch_col', type=str, default='BATCH')
    parser.add_argument('--output', type=str, default=None, help='Output plot path')
    parser.add_argument('--perplexity', type=int, default=30)
    args = parser.parse_args()

    print(f'Loading {args.data}...')
    adata = ad.read_h5ad(args.data)
    print(f'Loaded {adata.n_obs} cells, {adata.n_vars} markers\n')

    df = analyze_batch_effect(adata, args.celltype_col, args.batch_col, args.perplexity)

    output_path = args.output or str(Path(args.data).parent / 'batch_metrics.png')
    plot_metrics(df, output_path)


if __name__ == '__main__':
    main()
