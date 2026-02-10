"""
Tumor biomarker analysis: compare DE results across different batch correction strengths.
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
import warnings

# Configuration
DISC_DIRS = {
    0.05: '20260204_150407_disc_0.05',
    0.1: '20260204_151101_disc_0.1',
    0.3: '20260204_151803_disc_0.3',
}

BASE_DIR = Path('/mnt/iusers01/fatpou01/compsci01/w29632hl/scratch/code/BatchEffect/BioBatchNet/src/saved/immucan')
RAW_DATA_PATH = Path('/mnt/iusers01/fatpou01/compsci01/w29632hl/scratch/code/BatchEffect/BioBatchNet/DATA/IMC/IMMUcan.h5ad')
OUTPUT_DIR = Path('/mnt/iusers01/fatpou01/compsci01/w29632hl/scratch/code/BatchEffect/BioBatchNet/DATA/IMC/tumor_analysis_disc_sweep')

THRESHOLD = 0.5
SEED = 42


def best_leiden_by_nmi(adata: sc.AnnData, label_key: str, resolutions=(0.2, 0.4, 0.6, 0.8, 1.0), random_state=None):
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    best = (-1.0, -1.0, None)
    random_state = random_state if random_state is not None else SEED
    y = adata.obs[label_key].values
    for r in resolutions:
        sc.tl.leiden(adata, key_added='cluster', resolution=r, random_state=random_state)
        pred = adata.obs['cluster'].values
        nmi = normalized_mutual_info_score(y, pred)
        ari = adjusted_rand_score(y, pred)
        if nmi > best[0]:
            best = (nmi, ari, r)
    print(f"  best_leiden_by_nmi: resolution={best[2]}, NMI={best[0]:.4f}, ARI={best[1]:.4f}")
    return best


def plot_umap_with_leiden(adata, save_path, title_prefix, tumor_cluster=None, stroma_cluster=None):
    """Create UMAP visualization with celltype, batch, leiden, and DE clusters."""
    sc.tl.umap(adata, random_state=SEED)
    cols = ['celltype', 'BATCH', 'leiden']
    titles = ['Cell Type', 'Batch', f'Leiden (n={adata.obs["leiden"].nunique()})']
    if tumor_cluster is not None or stroma_cluster is not None:
        adata.obs['de_cluster'] = 'other'
        if tumor_cluster is not None:
            adata.obs.loc[adata.obs['leiden'] == tumor_cluster, 'de_cluster'] = f'Tumor (cluster {tumor_cluster})'
        if stroma_cluster is not None:
            adata.obs.loc[adata.obs['leiden'] == stroma_cluster, 'de_cluster'] = f'Stroma (cluster {stroma_cluster})'
        cols.append('de_cluster')
        titles.append('DE clusters')
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, col, title in zip(axes, cols, titles):
        sc.pl.umap(adata, color=col, ax=ax, show=False, legend_loc='on data' if col == 'leiden' else 'right margin',
                   frameon=False, title=f'{title_prefix} - {title}')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def identify_tumor_stroma_clusters(adata):
    """Identify clusters enriched with Tumor or Stroma cells."""
    ct_cluster = pd.crosstab(adata.obs['leiden'], adata.obs['celltype'], normalize='index')
    tumor_clusters = ct_cluster[ct_cluster['Tumor'] > THRESHOLD].index.tolist()
    stroma_clusters = ct_cluster[ct_cluster['Stroma'] > THRESHOLD].index.tolist()
    return tumor_clusters, stroma_clusters


def run_de_tumor_vs_stroma(adata, tumor_cluster, stroma_cluster):
    """Run DE analysis: one Tumor cluster vs one Stroma cluster."""
    stroma_mask = adata.obs['leiden'] == stroma_cluster
    tumor_mask = adata.obs['leiden'] == tumor_cluster

    adata_subset = adata[tumor_mask | stroma_mask].copy()
    adata_subset.obs['de_group'] = 'Stroma'
    adata_subset.obs.loc[tumor_mask[tumor_mask | stroma_mask], 'de_group'] = 'Tumor'

    sc.tl.rank_genes_groups(adata_subset, groupby='de_group', groups=['Tumor'],
                            reference='Stroma', method='wilcoxon', use_raw=False)

    result = sc.get.rank_genes_groups_df(adata_subset, group='Tumor')
    result = result.rename(columns={'names': 'marker'})
    result['rank'] = range(1, len(result) + 1)
    return result


def process_data(raw_adata, label, corrected_path=None):
    """Process data and run DE analysis."""
    print(f"Processing: {label}")
    if corrected_path is None:
        adata = raw_adata.copy()
        sc.pp.neighbors(adata, use_rep='X', random_state=SEED)
    else:
        adata = sc.read_h5ad(corrected_path)
        sc.pp.neighbors(adata, use_rep='X_biobatchnet', random_state=SEED)

    _, _, best_res = best_leiden_by_nmi(adata, 'celltype', random_state=SEED)
    sc.tl.leiden(adata, key_added='leiden', resolution=best_res, random_state=SEED)

    tumor_clusters, stroma_clusters = identify_tumor_stroma_clusters(adata)
    tumor_cluster = tumor_clusters[0] if tumor_clusters else None
    stroma_cluster = stroma_clusters[0] if stroma_clusters else None
    print(f"  tumor cluster for DE: leiden cluster {tumor_cluster} (of {tumor_clusters})")
    print(f"  stroma cluster for DE: leiden cluster {stroma_cluster} (of {stroma_clusters})")

    plot_umap_with_leiden(adata, OUTPUT_DIR / f'umap_{label}.png', label,
                          tumor_cluster=tumor_cluster, stroma_cluster=stroma_cluster)

    # DE uses raw expression
    if corrected_path is not None:
        raw_copy = raw_adata.copy()
        raw_copy.obs['leiden'] = adata.obs['leiden'].values
        adata = raw_copy

    if tumor_cluster is None or stroma_cluster is None:
        raise ValueError("Need at least one tumor and one stroma cluster for DE.")
    return run_de_tumor_vs_stroma(adata, tumor_cluster, stroma_cluster)


def compare_marker_ranks(all_de, output_dir):
    """Compare marker rankings across conditions."""
    top_k = 15
    all_markers = set()
    for de_df in all_de.values():
        all_markers.update(de_df.head(top_k)['marker'].tolist())

    markers = sorted(all_markers)
    labels = list(all_de.keys())

    rank_matrix = pd.DataFrame(index=markers, columns=labels)
    for label, de_df in all_de.items():
        marker_to_rank = dict(zip(de_df['marker'], de_df['rank']))
        for marker in markers:
            rank_matrix.loc[marker, label] = marker_to_rank.get(marker, len(de_df) + 1)

    rank_matrix = rank_matrix.astype(float)
    rank_matrix = rank_matrix.loc[rank_matrix.mean(axis=1).sort_values().index]

    # Plot
    fig, ax = plt.subplots(figsize=(6, len(markers) * 0.35))
    im = ax.imshow(rank_matrix.values, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(markers)))
    ax.set_yticklabels(rank_matrix.index, fontsize=9)

    for i in range(len(markers)):
        for j in range(len(labels)):
            val = int(rank_matrix.iloc[i, j])
            ax.text(j, i, str(val), ha='center', va='center', fontsize=8,
                    color='white' if val > top_k else 'black')

    plt.colorbar(im, ax=ax, label='Rank')
    ax.set_title('Tumor vs Stroma Marker Rankings')
    plt.tight_layout()
    plt.savefig(output_dir / 'marker_rank_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading raw data: {RAW_DATA_PATH}")
    raw_adata = sc.read_h5ad(RAW_DATA_PATH)
    print(f"  n_obs={raw_adata.n_obs}, n_vars={raw_adata.n_vars}")

    all_de = {'raw': process_data(raw_adata, 'raw')}

    for disc, disc_dir in sorted(DISC_DIRS.items()):
        path = BASE_DIR / disc_dir / 'seed_42' / 'biobatchnet.h5ad'
        all_de[f'disc{disc}'] = process_data(raw_adata, f'disc{disc}', path)

    compare_marker_ranks(all_de, OUTPUT_DIR)

if __name__ == '__main__':
    main()
