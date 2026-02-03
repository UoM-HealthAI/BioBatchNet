"""
Calculate iLISI and ASW_batch per cell type using scib.
"""
import scanpy as sc
import scib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def compute_metrics_per_celltype(
    adata_path: str,
    output_dir: str = None,
    embed_key: str = 'X_pca',
    batch_key: str = 'BATCH',
    celltype_key: str = 'celltype',
    exclude_celltypes: list = None,
):
    """
    Compute iLISI and ASW_batch per cell type.
    """
    # Load data
    adata = sc.read_h5ad(adata_path)
    print(f"Loaded: {adata.shape[0]} cells, {adata.shape[1]} markers")

    # Exclude celltypes if specified
    if exclude_celltypes:
        mask = ~adata.obs[celltype_key].isin(exclude_celltypes)
        adata = adata[mask].copy()
        print(f"After excluding {exclude_celltypes}: {adata.shape[0]} cells")

    # Use X directly (low-dim data like IMC)
    from scipy.sparse import issparse
    if embed_key not in adata.obsm:
        X = adata.X.toarray() if issparse(adata.X) else adata.X
        adata.obsm['X_expr'] = X
        embed_key = 'X_expr'

    # Compute neighbors if not present
    if 'connectivities' not in adata.obsp:
        print("Computing neighbors...")
        sc.pp.neighbors(adata, use_rep=embed_key)

    # Get unique celltypes
    celltypes = sorted(adata.obs[celltype_key].unique())
    print(f"Cell types: {celltypes}")

    results = []

    for ct in celltypes:
        print(f"\nProcessing {ct}...")
        adata_ct = adata[adata.obs[celltype_key] == ct].copy()
        n_cells = adata_ct.shape[0]
        n_batches = adata_ct.obs[batch_key].nunique()

        print(f"  {n_cells} cells, {n_batches} batches")

        if n_batches < 2 or n_cells < 50:
            print(f"  Skipping: insufficient data")
            continue

        # Recompute neighbors for subset
        sc.pp.neighbors(adata_ct, use_rep=embed_key)

        # Compute iLISI
        try:
            ilisi = scib.metrics.ilisi_graph(
                adata_ct, batch_key=batch_key, type_='embed',
                use_rep=embed_key, scale=True
            )
            print(f"  iLISI: {ilisi:.4f}")
        except Exception as e:
            print(f"  iLISI failed: {e}")
            ilisi = np.nan

        # Compute ASW_batch
        try:
            asw_batch = scib.metrics.silhouette_batch(
                adata_ct, batch_key=batch_key, label_key=celltype_key,
                embed=embed_key, scale=True
            )
            print(f"  ASW_batch: {asw_batch:.4f}")
        except Exception as e:
            print(f"  ASW_batch failed: {e}")
            asw_batch = np.nan

        results.append({
            'celltype': ct, 'n_cells': n_cells, 'n_batches': n_batches,
            'iLISI': ilisi, 'ASW_batch': asw_batch,
        })

    df = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print(df.to_string(index=False))

    # Save plot
    if output_dir is None:
        output_dir = Path(adata_path).parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metrics(df, output_dir)

    return df


def plot_metrics(df: pd.DataFrame, output_dir: Path):
    """Plot bar plot for iLISI and ASW_batch per cell type."""
    df_plot = df.dropna(subset=['iLISI', 'ASW_batch']).sort_values('iLISI', ascending=False)

    if df_plot.empty:
        print("No valid data to plot")
        return

    # Nature color palette
    nature_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
                     '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
    colors = {ct: nature_colors[i % len(nature_colors)] for i, ct in enumerate(df_plot['celltype'])}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # iLISI - vertical bars
    ax1 = axes[0]
    x = range(len(df_plot))
    bars1 = ax1.bar(x, df_plot['iLISI'], color=[colors[ct] for ct in df_plot['celltype']],
                    edgecolor='white', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_plot['celltype'], rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('iLISI', fontsize=9)
    ax1.set_title('iLISI', fontsize=12)
    ax1.set_ylim(0, max(df_plot['iLISI']) * 1.25)
    ax1.tick_params(axis='y', labelsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # Add value labels
    for bar, val in zip(bars1, df_plot['iLISI']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    # ASW_batch - vertical bars, same order
    ax2 = axes[1]
    bars2 = ax2.bar(x, df_plot['ASW_batch'], color=[colors[ct] for ct in df_plot['celltype']],
                    edgecolor='white', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_plot['celltype'], rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('ASW_batch', fontsize=9)
    ax2.set_title('ASW_batch', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='y', labelsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    # Add value labels
    for bar, val in zip(bars2, df_plot['ASW_batch']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / 'batch_metrics_per_celltype.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'batch_metrics_per_celltype.png'}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('adata_path', type=str)
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--embed', '-e', type=str, default='X_pca')
    parser.add_argument('--exclude', nargs='+', default=['undefined'])
    args = parser.parse_args()

    compute_metrics_per_celltype(
        args.adata_path,
        output_dir=args.output,
        embed_key=args.embed,
        exclude_celltypes=args.exclude
    )
