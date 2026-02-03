"""
Plot per-cell expression heatmap with celltype and patient color bars.
Cells are sorted by celltype first, then by patient within each celltype.
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from scipy.sparse import issparse
import seaborn as sns
from pathlib import Path
import argparse


def plot_percell_heatmap(
    adata_path: str,
    output_path: str = None,
    celltype_key: str = 'celltype',
    batch_key: str = 'BATCH',
    sample_frac: float = None,
    exclude_celltypes: list = None,
    figsize: tuple = (16, 10),
    random_seed: int = 42,
):
    """
    Plot per-cell expression heatmap with celltype and patient color bars.

    Args:
        adata_path: Path to h5ad file
        output_path: Path to save figure (default: same dir as adata)
        celltype_key: Column name for celltype
        batch_key: Column name for batch/patient
        sample_frac: Fraction of cells to sample globally (None = no sampling)
        exclude_celltypes: List of celltypes to exclude
        figsize: Figure size
        random_seed: Random seed for sampling
    """
    np.random.seed(random_seed)

    # Load data
    adata = sc.read_h5ad(adata_path)
    print(f"Loaded: {adata.shape[0]} cells, {adata.shape[1]} markers")

    # Exclude celltypes if specified
    if exclude_celltypes:
        mask = ~adata.obs[celltype_key].isin(exclude_celltypes)
        adata = adata[mask].copy()
        print(f"After excluding {exclude_celltypes}: {adata.shape[0]} cells")

    # Global random sampling (preserves celltype proportions)
    if sample_frac and sample_frac < 1.0:
        n_sample = int(adata.shape[0] * sample_frac)
        sampled_idx = np.random.choice(adata.obs_names, n_sample, replace=False)
        adata = adata[sampled_idx].copy()
        print(f"After sampling {sample_frac*100:.0f}%: {adata.shape[0]} cells")

    # Sort by celltype, then by batch within celltype
    sort_order = adata.obs.sort_values([celltype_key, batch_key]).index

    adata = adata[sort_order].copy()

    # Get expression matrix
    X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)

    # Create color mappings
    celltypes = adata.obs[celltype_key].values
    batches = adata.obs[batch_key].values

    unique_celltypes = sorted(adata.obs[celltype_key].unique())
    unique_batches = sorted(adata.obs[batch_key].unique())

    # Celltype: tab20, Patient: Nature palette
    celltype_palette = sns.color_palette("tab20", n_colors=len(unique_celltypes))
    nature_palette = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']
    batch_palette = nature_palette[:len(unique_batches)]

    celltype_colors = {ct: celltype_palette[i] for i, ct in enumerate(unique_celltypes)}
    batch_colors = {b: batch_palette[i] for i, b in enumerate(unique_batches)}

    # Create numeric arrays for color bars
    celltype_numeric = np.array([unique_celltypes.index(ct) for ct in celltypes])
    batch_numeric = np.array([unique_batches.index(b) for b in batches])

    # Create figure
    fig = plt.figure(figsize=figsize)

    # GridSpec: [celltype bar, batch bar, heatmap, colorbar] + right for legends
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        4, 2,
        height_ratios=[0.4, 0.4, 20, 0.6],
        width_ratios=[20, 2.5],
        hspace=0.02,
        wspace=0.03
    )

    ax_celltype = fig.add_subplot(gs[0, 0])
    ax_batch = fig.add_subplot(gs[1, 0])
    ax_heatmap = fig.add_subplot(gs[2, 0])
    ax_cbar = fig.add_subplot(gs[3, 0])  # Horizontal colorbar at bottom
    ax_legend = fig.add_subplot(gs[2, 1])  # Legends next to heatmap only

    # Plot celltype color bar
    celltype_cmap = ListedColormap([celltype_colors[ct] for ct in unique_celltypes])
    ax_celltype.imshow(
        celltype_numeric.reshape(1, -1),
        aspect='auto',
        cmap=celltype_cmap,
        vmin=0, vmax=len(unique_celltypes) - 1
    )
    ax_celltype.set_xticks([])
    ax_celltype.set_yticks([0])
    ax_celltype.set_yticklabels(['Cell Type'], fontsize=10)
    ax_celltype.spines[:].set_visible(False)

    # Plot batch color bar
    batch_cmap = ListedColormap([batch_colors[b] for b in unique_batches])
    ax_batch.imshow(
        batch_numeric.reshape(1, -1),
        aspect='auto',
        cmap=batch_cmap,
        vmin=0, vmax=len(unique_batches) - 1
    )
    ax_batch.set_xticks([])
    ax_batch.set_yticks([0])
    ax_batch.set_yticklabels(['Patient'], fontsize=10)
    ax_batch.spines[:].set_visible(False)

    # Plot heatmap
    im = ax_heatmap.imshow(X.T, aspect='auto', cmap='viridis', interpolation='none')
    ax_heatmap.set_yticks(range(len(adata.var_names)))
    ax_heatmap.set_yticklabels(adata.var_names, fontsize=8)
    ax_heatmap.set_xticks([])
    ax_heatmap.set_ylabel('Markers', fontsize=11)

    # Add celltype separators
    cumsum = 0
    celltype_counts = adata.obs[celltype_key].value_counts().reindex(
        adata.obs[celltype_key].unique()
    )
    # Recompute based on actual order in sorted data
    prev_ct = None
    for i, ct in enumerate(celltypes):
        if ct != prev_ct and prev_ct is not None:
            ax_heatmap.axvline(x=i - 0.5, color='white', linewidth=1.5)
            ax_celltype.axvline(x=i - 0.5, color='white', linewidth=1.5)
            ax_batch.axvline(x=i - 0.5, color='white', linewidth=1.5)
        prev_ct = ct

    # Right panel for legends (close to heatmap)
    ax_legend.axis('off')

    # Create legend handles
    celltype_handles = [Patch(facecolor=celltype_colors[ct], label=ct, edgecolor='gray', linewidth=0.5)
                        for ct in unique_celltypes]
    batch_handles = [Patch(facecolor=batch_colors[b], label=b, edgecolor='gray', linewidth=0.5)
                     for b in unique_batches]

    # Cell Type legend at top
    legend1 = ax_legend.legend(
        handles=celltype_handles,
        title='Cell Type',
        loc='upper left',
        bbox_to_anchor=(0.0, 1.0),
        ncol=1,
        fontsize=9,
        title_fontsize=9,
        frameon=True,
        edgecolor='lightgray'
    )
    ax_legend.add_artist(legend1)

    # Patient legend below Cell Type
    legend2 = ax_legend.legend(
        handles=batch_handles,
        title='Patient',
        loc='lower left',
        bbox_to_anchor=(0.0, 0.0),
        ncol=1,
        fontsize=9,
        title_fontsize=9,
        frameon=True,
        edgecolor='lightgray'
    )

    # Horizontal expression colorbar at bottom
    cbar = plt.colorbar(im, cax=ax_cbar, orientation='horizontal')
    cbar.set_label('Expression', fontsize=9)
    cbar.ax.tick_params(labelsize=9)

    # Save
    if output_path is None:
        output_path = Path(adata_path).parent / 'IMMUcan_heatmap_percell.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot per-cell expression heatmap')
    parser.add_argument('adata_path', type=str, help='Path to h5ad file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output path')
    parser.add_argument('--sample-frac', '-s', type=float, default=0.1,
                        help='Fraction of cells to sample globally (default: 0.1)')
    parser.add_argument('--exclude', '-e', nargs='+', default=['undefined'],
                        help='Celltypes to exclude (default: undefined)')
    args = parser.parse_args()

    plot_percell_heatmap(
        args.adata_path,
        output_path=args.output,
        sample_frac=args.sample_frac,
        exclude_celltypes=args.exclude,
    )
