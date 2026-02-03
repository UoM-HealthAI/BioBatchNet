"""
Discriminator Sweep Analysis for BioBatchNet on immuncan dataset.

Improvements:
1. Use rank_genes_groups (Wilcoxon) for true biomarker discovery
2. Save top biomarkers per cluster as CSV
3. Heatmap only shows top biomarkers (not all markers)
4. Cell heatmap uses subsampling + top markers only
5. Output organized by disc subdirectories
6. Fixed random seeds for reproducibility
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.sparse import issparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def log(msg):
    print(msg, flush=True)


# Configuration
DISC_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
RESOLUTION = 0.4
RANDOM_SEED = 42
TOP_K_MARKERS = 10  # Top markers per cluster for heatmap
CELLS_PER_CLUSTER = 200  # Subsample for cell heatmap

BASE_DIR = Path(__file__).parent.parent / 'src' / 'saved' / 'immuncan'
OUTPUT_DIR = BASE_DIR / 'analysis'

# Read marker names from original data
ORIGINAL_DATA_PATH = Path(__file__).parent.parent / 'DATA' / 'IMC' / 'IMMUcan_batch.h5ad'

DISC_DIRS = {
    0.1: '20260129_122403_disc0.1',
    0.3: '20260129_122403_disc0.3',
    0.5: '20260129_122403_disc0.5',
    0.7: '20260129_122403_disc0.7',
    0.9: '20260129_123905_disc0.9',
}


def get_marker_names():
    """Read marker names from original data."""
    orig = sc.read_h5ad(ORIGINAL_DATA_PATH)
    return list(orig.var_names)


def load_adata(disc: float, marker_names: list) -> sc.AnnData:
    """Load adata for a specific discriminator strength."""
    path = BASE_DIR / DISC_DIRS[disc] / 'seed_42' / 'adata.h5ad'
    adata = sc.read_h5ad(path)

    # Rename markers from indices to real names
    if len(adata.var_names) == len(marker_names):
        adata.var_names = marker_names
    else:
        log(f"  Warning: marker count mismatch ({len(adata.var_names)} vs {len(marker_names)})")

    log(f"Loaded disc={disc}: {adata.shape[0]} cells, {adata.shape[1]} markers")
    return adata


def cluster_adata(adata: sc.AnnData, label_key: str = 'celltype') -> tuple:
    """Perform Leiden clustering with fixed resolution and seed."""
    log("  Computing neighbors...")
    sc.pp.neighbors(adata, use_rep='X_biobatchnet', random_state=RANDOM_SEED)

    log(f"  Running Leiden (resolution={RESOLUTION})...")
    sc.tl.leiden(adata, key_added='cluster', resolution=RESOLUTION, random_state=RANDOM_SEED)

    # Compute metrics
    y = adata.obs[label_key].values
    pred = adata.obs['cluster'].values
    nmi = normalized_mutual_info_score(y, pred)
    ari = adjusted_rand_score(y, pred)
    n_clusters = adata.obs['cluster'].nunique()

    log(f"  n_clusters: {n_clusters}, NMI: {nmi:.3f}, ARI: {ari:.3f}")
    return nmi, ari, n_clusters


def find_biomarkers(adata: sc.AnnData) -> pd.DataFrame:
    """Run rank_genes_groups to find cluster biomarkers."""
    log("  Running Wilcoxon rank test for biomarkers...")
    sc.tl.rank_genes_groups(adata, groupby='cluster', method='wilcoxon', use_raw=False)

    # Extract results into DataFrame
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names

    rows = []
    for group in groups:
        for rank in range(len(result['names'][group])):
            rows.append({
                'cluster': group,
                'rank': rank + 1,
                'marker': result['names'][group][rank],
                'score': result['scores'][group][rank],
                'logfoldchange': result['logfoldchanges'][group][rank],
                'pval': result['pvals'][group][rank],
                'pval_adj': result['pvals_adj'][group][rank],
            })

    df = pd.DataFrame(rows)
    return df


def get_top_markers(biomarkers_df: pd.DataFrame, top_k: int = TOP_K_MARKERS) -> list:
    """Get unique top markers across all clusters."""
    top_df = biomarkers_df[biomarkers_df['rank'] <= top_k]
    markers = top_df['marker'].unique().tolist()
    return markers


def plot_umap(adata: sc.AnnData, output_path: Path, disc: float):
    """Create UMAP plot with celltype, BATCH, and cluster coloring."""
    from matplotlib.lines import Line2D

    if 'X_umap' not in adata.obsm:
        log("  Computing UMAP...")
        sc.tl.umap(adata, random_state=RANDOM_SEED)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Get unique categories
    cell_cats = sorted(adata.obs['celltype'].unique())
    batch_cats = sorted(adata.obs['BATCH'].unique())
    cluster_cats = sorted(adata.obs['cluster'].unique(), key=lambda x: int(x))

    # Color palettes
    cell_palette = sc.pl.palettes.default_20 if len(cell_cats) <= 20 else sc.pl.palettes.default_102
    batch_palette = sc.pl.palettes.default_20
    cluster_palette = sc.pl.palettes.default_20 if len(cluster_cats) <= 20 else sc.pl.palettes.default_102

    cell_colors = {cat: cell_palette[i] for i, cat in enumerate(cell_cats)}
    batch_colors = {cat: batch_palette[i] for i, cat in enumerate(batch_cats)}
    cluster_colors = {cat: cluster_palette[i] for i, cat in enumerate(cluster_cats)}

    # Plot celltype
    sc.pl.umap(adata, color='celltype', ax=axes[0], show=False, legend_loc=None,
               frameon=False, title='Cell Type', palette=[cell_colors[c] for c in cell_cats])

    # Plot BATCH
    sc.pl.umap(adata, color='BATCH', ax=axes[1], show=False, legend_loc=None,
               frameon=False, title='Batch', palette=[batch_colors[c] for c in batch_cats])

    # Plot cluster
    sc.pl.umap(adata, color='cluster', ax=axes[2], show=False, legend_loc=None,
               frameon=False, title=f'Leiden Cluster (n={len(cluster_cats)})',
               palette=[cluster_colors[c] for c in cluster_cats])

    # Add legends at bottom
    cell_handles = [Line2D([0], [0], marker='o', color='w', label=cat,
                           markerfacecolor=cell_colors[cat], markersize=6) for cat in cell_cats]
    axes[0].legend(handles=cell_handles, loc='upper center', bbox_to_anchor=(0.5, -0.08),
                   ncol=min(len(cell_cats), 4), fontsize=7, frameon=False)

    batch_handles = [Line2D([0], [0], marker='o', color='w', label=cat,
                            markerfacecolor=batch_colors[cat], markersize=6) for cat in batch_cats]
    axes[1].legend(handles=batch_handles, loc='upper center', bbox_to_anchor=(0.5, -0.08),
                   ncol=min(len(batch_cats), 6), fontsize=7, frameon=False)

    cluster_handles = [Line2D([0], [0], marker='o', color='w', label=f'C{cat}',
                              markerfacecolor=cluster_colors[cat], markersize=6) for cat in cluster_cats]
    axes[2].legend(handles=cluster_handles, loc='upper center', bbox_to_anchor=(0.5, -0.08),
                   ncol=min(len(cluster_cats), 6), fontsize=7, frameon=False)

    plt.suptitle(f'disc={disc}', fontsize=12, y=1.02)
    plt.subplots_adjust(bottom=0.2, wspace=0.15)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved UMAP to {output_path}")


def plot_cluster_heatmap(adata: sc.AnnData, top_markers: list, output_path: Path, disc: float):
    """Create Cluster x Marker heatmap using only top biomarkers."""
    # Filter to top markers
    markers_in_data = [m for m in top_markers if m in adata.var_names]

    X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
    df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    df = df[markers_in_data]

    # Mean expression per cluster
    cluster_expr = df.groupby(adata.obs['cluster'].values).mean()

    # Z-score across clusters
    cluster_expr_z = (cluster_expr - cluster_expr.mean()) / cluster_expr.std()
    cluster_expr_z = cluster_expr_z.fillna(0)

    # Sort clusters numerically
    cluster_expr_z.index = cluster_expr_z.index.astype(int)
    cluster_expr_z = cluster_expr_z.sort_index()

    g = sns.clustermap(
        cluster_expr_z,
        cmap='RdBu_r',
        center=0,
        figsize=(max(12, len(markers_in_data) * 0.4), max(6, len(cluster_expr_z) * 0.5)),
        dendrogram_ratio=(0.1, 0.15),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        xticklabels=True,
        yticklabels=True,
        row_cluster=True,
        col_cluster=True,
    )
    g.ax_heatmap.set_xlabel('Top Biomarkers')
    g.ax_heatmap.set_ylabel('Cluster')
    g.fig.suptitle(f'Cluster x Top Biomarkers (disc={disc}, top{TOP_K_MARKERS}/cluster)', y=1.02)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved cluster heatmap to {output_path}")


def plot_cell_heatmap(adata: sc.AnnData, top_markers: list, output_path: Path, disc: float):
    """Create Cell x Marker heatmap with subsampling and celltype color bar."""
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    # Subsample cells per cluster
    np.random.seed(RANDOM_SEED)
    sampled_idx = []
    for cluster in sorted(adata.obs['cluster'].unique(), key=int):
        cluster_cells = adata.obs[adata.obs['cluster'] == cluster].index.tolist()
        n_sample = min(CELLS_PER_CLUSTER, len(cluster_cells))
        sampled = np.random.choice(cluster_cells, n_sample, replace=False)
        sampled_idx.extend(sampled)

    # Filter to top markers
    markers_in_data = [m for m in top_markers if m in adata.var_names]

    # Create subset with only top markers
    adata_sub = adata[sampled_idx, markers_in_data].copy()

    # Sort cells by cluster
    adata_sub = adata_sub[adata_sub.obs.sort_values('cluster').index]

    # Get expression matrix
    X = adata_sub.X.toarray() if issparse(adata_sub.X) else np.array(adata_sub.X)

    # Create celltype color mapping
    celltypes = adata_sub.obs['celltype'].values
    unique_celltypes = sorted(adata_sub.obs['celltype'].unique())
    celltype_palette = sc.pl.palettes.default_20 if len(unique_celltypes) <= 20 else sc.pl.palettes.default_102
    celltype_colors = {ct: celltype_palette[i] for i, ct in enumerate(unique_celltypes)}
    celltype_numeric = np.array([unique_celltypes.index(ct) for ct in celltypes])

    # Create figure with GridSpec (celltype bar on top)
    fig = plt.figure(figsize=(14, max(10, len(markers_in_data) * 0.35)))
    gs = GridSpec(2, 2, height_ratios=[1, 20], width_ratios=[20, 1], hspace=0.02, wspace=0.02)

    ax_celltype = fig.add_subplot(gs[0, 0])  # Celltype color bar
    ax_heatmap = fig.add_subplot(gs[1, 0])   # Main heatmap
    ax_cbar = fig.add_subplot(gs[1, 1])      # Expression colorbar

    # Plot celltype color bar
    celltype_cmap = ListedColormap([celltype_colors[ct] for ct in unique_celltypes])
    ax_celltype.imshow(celltype_numeric.reshape(1, -1), aspect='auto', cmap=celltype_cmap,
                       vmin=0, vmax=len(unique_celltypes)-1)
    ax_celltype.set_xticks([])
    ax_celltype.set_yticks([0])
    ax_celltype.set_yticklabels(['Cell Type'], fontsize=9)

    # Plot main heatmap
    im = ax_heatmap.imshow(X.T, aspect='auto', cmap='viridis')

    # Set y-axis (markers)
    ax_heatmap.set_yticks(range(len(markers_in_data)))
    ax_heatmap.set_yticklabels(markers_in_data, fontsize=8)

    # Add cluster separators and labels
    cluster_counts = adata_sub.obs['cluster'].value_counts().sort_index()
    cumsum = 0
    cluster_positions = []
    for cluster, count in cluster_counts.items():
        cluster_positions.append(cumsum + count / 2)
        cumsum += count
        ax_heatmap.axvline(x=cumsum - 0.5, color='white', linewidth=1)
        ax_celltype.axvline(x=cumsum - 0.5, color='white', linewidth=1)

    # Set x-axis
    ax_heatmap.set_xticks(cluster_positions)
    ax_heatmap.set_xticklabels([f'C{c}' for c in cluster_counts.index], fontsize=10)
    ax_heatmap.set_xlabel('Cluster')
    ax_heatmap.set_ylabel('Biomarkers')

    # Add expression colorbar
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('Expression')

    # Add celltype legend at bottom
    legend_handles = [Patch(facecolor=celltype_colors[ct], label=ct) for ct in unique_celltypes]
    fig.legend(handles=legend_handles, loc='lower center', ncol=min(len(unique_celltypes), 6),
               fontsize=8, frameon=False, bbox_to_anchor=(0.45, -0.02))

    plt.suptitle(f'Cell x Top Biomarkers (disc={disc}, {CELLS_PER_CLUSTER} cells/cluster)', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved cell heatmap to {output_path}")


def plot_umap_comparison(adatas: dict, output_path: Path):
    """Create UMAP comparison plot for all disc values."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, (disc, adata) in enumerate(adatas.items()):
        ax = axes[idx]
        if 'X_umap' not in adata.obsm:
            sc.tl.umap(adata, random_state=RANDOM_SEED)

        n_clusters = adata.obs['cluster'].nunique()
        sc.pl.umap(adata, color='cluster', ax=ax, show=False, legend_loc='none',
                   frameon=False, title=f'disc={disc} (n={n_clusters})')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"Saved UMAP comparison to {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    adatas = {}
    summary_rows = []

    log("=" * 60)
    log("Loading, clustering, and finding biomarkers...")
    log("=" * 60)

    # Read marker names from original data
    log("Reading marker names from original data...")
    marker_names = get_marker_names()
    log(f"Markers: {marker_names[:5]}... ({len(marker_names)} total)")

    for disc in DISC_VALUES:
        log(f"\n{'='*60}")
        log(f"Processing disc={disc}...")
        log("=" * 60)

        # Create disc-specific output directory
        disc_dir = OUTPUT_DIR / f'disc{disc}'
        disc_dir.mkdir(parents=True, exist_ok=True)

        # Load and cluster
        adata = load_adata(disc, marker_names)
        nmi, ari, n_clusters = cluster_adata(adata)

        # Find biomarkers
        biomarkers_df = find_biomarkers(adata)
        biomarkers_path = disc_dir / 'biomarkers_all.csv'
        biomarkers_df.to_csv(biomarkers_path, index=False)
        log(f"  Saved all biomarkers to {biomarkers_path}")

        # Save top biomarkers summary
        top_df = biomarkers_df[biomarkers_df['rank'] <= TOP_K_MARKERS]
        top_path = disc_dir / f'biomarkers_top{TOP_K_MARKERS}.csv'
        top_df.to_csv(top_path, index=False)
        log(f"  Saved top biomarkers to {top_path}")

        # Get top markers for heatmaps
        top_markers = get_top_markers(biomarkers_df)
        log(f"  Using {len(top_markers)} unique top markers for heatmaps")

        # Generate plots
        plot_umap(adata, disc_dir / 'umap.png', disc)
        plot_cluster_heatmap(adata, top_markers, disc_dir / 'cluster_heatmap.png', disc)
        plot_cell_heatmap(adata, top_markers, disc_dir / 'cell_heatmap.png', disc)

        adatas[disc] = adata
        summary_rows.append({
            'disc': disc,
            'n_clusters': n_clusters,
            'ARI': ari,
            'NMI': nmi,
            'resolution': RESOLUTION,
            'n_top_markers': len(top_markers),
        })

    # Save clustering summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / 'clustering_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    log(f"\nSaved clustering summary to {summary_path}")
    log("\nClustering Summary:")
    log(summary_df.to_string(index=False))

    # Generate combined UMAP comparison
    log("\n" + "=" * 60)
    log("Generating combined UMAP comparison...")
    log("=" * 60)
    plot_umap_comparison(adatas, OUTPUT_DIR / 'umap_comparison.png')

    log("\n" + "=" * 60)
    log("Analysis complete!")
    log(f"Output directory: {OUTPUT_DIR}")
    log("=" * 60)


if __name__ == '__main__':
    main()
