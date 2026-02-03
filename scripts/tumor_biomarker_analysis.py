"""
Tumor biomarker analysis: compare DE results before and after batch correction.

Raw (Patient 4 only) vs Corrected (global)
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.sparse import issparse
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from matplotlib.lines import Line2D


def plot_umap_with_leiden(adata, save_path, title_prefix, celltype_key='celltype',
                          batch_key='BATCH', nmi=None, ari=None, seed=42):
    """Create UMAP visualization with celltype and leiden clusters."""
    # Compute UMAP
    sc.tl.umap(adata, random_state=seed)

    n_clusters = adata.obs['leiden'].nunique()

    # Determine if we need batch panel (for global data with multiple batches)
    has_batch = batch_key in adata.obs.columns and adata.obs[batch_key].nunique() > 1

    if has_batch:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Get unique categories and create color maps
    cell_cats = sorted(adata.obs[celltype_key].unique())
    cluster_cats = sorted(adata.obs['leiden'].unique(), key=lambda x: int(x))

    cell_palette = sc.pl.palettes.default_20 if len(cell_cats) <= 20 else sc.pl.palettes.default_102
    cluster_palette = sc.pl.palettes.default_20 if len(cluster_cats) <= 20 else sc.pl.palettes.default_102

    cell_colors = {cat: cell_palette[i] for i, cat in enumerate(cell_cats)}
    cluster_colors = {cat: cluster_palette[i] for i, cat in enumerate(cluster_cats)}

    # Plot celltype
    sc.pl.umap(adata, color=celltype_key, ax=axes[0], show=False, legend_loc=None,
               frameon=False, title=f'{title_prefix} - Cell Type',
               palette=[cell_colors[c] for c in cell_cats])

    # Add celltype legend
    cell_handles = [Line2D([0], [0], marker='o', color='w', label=cat,
                           markerfacecolor=cell_colors[cat], markersize=6) for cat in cell_cats]
    axes[0].legend(handles=cell_handles, loc='upper center', bbox_to_anchor=(0.5, -0.08),
                   ncol=min(len(cell_cats), 4), fontsize=7, frameon=False)

    if has_batch:
        batch_cats = sorted(adata.obs[batch_key].unique())
        batch_palette = sc.pl.palettes.default_20
        batch_colors = {cat: batch_palette[i] for i, cat in enumerate(batch_cats)}

        # Plot batch
        sc.pl.umap(adata, color=batch_key, ax=axes[1], show=False, legend_loc=None,
                   frameon=False, title=f'{title_prefix} - Batch',
                   palette=[batch_colors[c] for c in batch_cats])

        batch_handles = [Line2D([0], [0], marker='o', color='w', label=cat,
                                markerfacecolor=batch_colors[cat], markersize=6) for cat in batch_cats]
        axes[1].legend(handles=batch_handles, loc='upper center', bbox_to_anchor=(0.5, -0.08),
                       ncol=min(len(batch_cats), 4), fontsize=7, frameon=False)

        leiden_ax = axes[2]
    else:
        leiden_ax = axes[1]

    # Plot leiden
    title_str = f'{title_prefix} - Leiden (n={n_clusters}'
    if nmi is not None:
        title_str += f', NMI={nmi:.3f}, ARI={ari:.3f}'
    title_str += ')'

    sc.pl.umap(adata, color='leiden', ax=leiden_ax, show=False, legend_loc=None,
               frameon=False, title=title_str,
               palette=[cluster_colors[c] for c in cluster_cats])

    cluster_handles = [Line2D([0], [0], marker='o', color='w', label=f'C{cat}',
                              markerfacecolor=cluster_colors[cat], markersize=6) for cat in cluster_cats]
    leiden_ax.legend(handles=cluster_handles, loc='upper center', bbox_to_anchor=(0.5, -0.08),
                     ncol=min(len(cluster_cats), 6), fontsize=7, frameon=False)

    plt.subplots_adjust(bottom=0.2, wspace=0.15)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved UMAP: {save_path}")


def best_leiden_by_nmi(adata, label_key, resolutions=(0.2, 0.4, 0.6, 0.8, 1.0)):
    """Find best leiden resolution by NMI with ground truth."""
    best = (-1.0, -1.0, None)
    y = adata.obs[label_key].values
    for r in resolutions:
        sc.tl.leiden(adata, key_added='_cluster_tmp', resolution=r)
        pred = adata.obs['_cluster_tmp'].values
        nmi = normalized_mutual_info_score(y, pred)
        ari = adjusted_rand_score(y, pred)
        if nmi > best[0]:
            best = (nmi, ari, r)
    return best


def identify_tumor_clusters(adata, cluster_key='leiden', celltype_key='celltype', threshold=0.5):
    """Identify clusters enriched with Tumor cells."""
    ct_cluster = pd.crosstab(adata.obs[cluster_key], adata.obs[celltype_key], normalize='index')

    print("\nCluster composition:")
    for cluster in ct_cluster.index:
        top = ct_cluster.loc[cluster].nlargest(2)
        print(f"  Cluster {cluster}: {top.index[0]} ({top.values[0]:.1%}), {top.index[1]} ({top.values[1]:.1%})")

    # Clusters with >threshold Tumor
    if 'Tumor' in ct_cluster.columns:
        tumor_clusters = ct_cluster[ct_cluster['Tumor'] > threshold].index.tolist()
    else:
        tumor_clusters = []

    print(f"\nTumor-enriched clusters (>{threshold:.0%}): {tumor_clusters}")
    return tumor_clusters


def run_de_analysis(adata, tumor_clusters, cluster_key='leiden', celltype_key='celltype'):
    """Run DE: Tumor cluster vs non-Tumor celltypes."""
    # Mark cells
    adata.obs['is_tumor_cluster'] = adata.obs[cluster_key].isin(tumor_clusters)

    # Get non-Tumor cells (by ground truth annotation, excluding Tumor and undefined)
    non_tumor_mask = ~adata.obs[celltype_key].isin(['Tumor', 'undefined'])
    tumor_cluster_mask = adata.obs['is_tumor_cluster']

    # Create comparison groups
    adata.obs['de_group'] = 'Other'
    adata.obs.loc[tumor_cluster_mask, 'de_group'] = 'Tumor_Cluster'
    adata.obs.loc[~tumor_cluster_mask & non_tumor_mask, 'de_group'] = 'Normal_Cells'

    # Subset to only Tumor_Cluster and Normal_Cells
    adata_de = adata[adata.obs['de_group'].isin(['Tumor_Cluster', 'Normal_Cells'])].copy()

    print(f"\nDE analysis: {(adata_de.obs['de_group'] == 'Tumor_Cluster').sum()} Tumor vs "
          f"{(adata_de.obs['de_group'] == 'Normal_Cells').sum()} Normal cells")

    # Run Wilcoxon
    sc.tl.rank_genes_groups(
        adata_de,
        groupby='de_group',
        groups=['Tumor_Cluster'],
        reference='Normal_Cells',
        method='wilcoxon',
        use_raw=False
    )

    # Extract results
    result = sc.get.rank_genes_groups_df(adata_de, group='Tumor_Cluster')
    result = result.rename(columns={'names': 'marker'})

    print(f"\nTop 10 Tumor markers:")
    print(result.head(10)[['marker', 'logfoldchanges', 'pvals_adj']].to_string(index=False))

    return result, adata_de


def process_raw_patient4(adata_path, output_dir, celltype_key='celltype', batch_key='BATCH'):
    """Process Raw data - Patient 4 only."""
    print("\n" + "=" * 60)
    print("Processing: Raw (Patient 4 only)")
    print("=" * 60)

    adata = sc.read_h5ad(adata_path)
    print(f"Loaded: {adata.shape}")

    # Subset to Patient 4
    adata = adata[adata.obs[batch_key] == 'Patient4'].copy()
    print(f"Patient 4 subset: {adata.shape}")

    # Prepare embedding (use X directly)
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    adata.obsm['X_expr'] = X

    # Compute neighbors
    sc.pp.neighbors(adata, use_rep='X_expr')

    # Find best resolution
    nmi, ari, best_r = best_leiden_by_nmi(adata, celltype_key)
    print(f"\nBest resolution: {best_r} (NMI={nmi:.3f}, ARI={ari:.3f})")

    # Cluster with best resolution
    sc.tl.leiden(adata, key_added='leiden', resolution=best_r)
    print(f"Clusters: {adata.obs['leiden'].nunique()}")

    # Plot UMAP
    plot_umap_with_leiden(adata, output_dir / 'umap_raw_patient4.png',
                          'Raw (Patient4)', celltype_key, batch_key, nmi, ari)

    # Identify tumor clusters
    tumor_clusters = identify_tumor_clusters(adata)

    # DE analysis
    de_result, adata_de = run_de_analysis(adata, tumor_clusters)

    return de_result, adata_de, adata, nmi, ari


def process_corrected_global(adata_path, output_dir, celltype_key='celltype', batch_key='BATCH'):
    """Process Corrected data - global (all patients)."""
    print("\n" + "=" * 60)
    print("Processing: Corrected (Global)")
    print("=" * 60)

    adata = sc.read_h5ad(adata_path)
    print(f"Loaded: {adata.shape}")

    # Use X_biobatchnet embedding
    if 'X_biobatchnet' not in adata.obsm:
        raise ValueError("X_biobatchnet not found in obsm")

    # Compute neighbors
    sc.pp.neighbors(adata, use_rep='X_biobatchnet')

    # Find best resolution
    nmi, ari, best_r = best_leiden_by_nmi(adata, celltype_key)
    print(f"\nBest resolution: {best_r} (NMI={nmi:.3f}, ARI={ari:.3f})")

    # Cluster with best resolution
    sc.tl.leiden(adata, key_added='leiden', resolution=best_r)
    print(f"Clusters: {adata.obs['leiden'].nunique()}")

    # Plot UMAP
    plot_umap_with_leiden(adata, output_dir / 'umap_corrected_global.png',
                          'Corrected (Global)', celltype_key, batch_key, nmi, ari)

    # Identify tumor clusters
    tumor_clusters = identify_tumor_clusters(adata)

    # DE analysis
    de_result, adata_de = run_de_analysis(adata, tumor_clusters)

    return de_result, adata_de, adata, nmi, ari


def plot_comparison(results, output_dir):
    """Plot DE comparison: Raw vs Corrected."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge results
    raw_df = results['Raw'].copy()
    raw_df = raw_df.rename(columns={'logfoldchanges': 'logFC_Raw', 'pvals_adj': 'padj_Raw'})

    corr_df = results['Corrected'].copy()
    corr_df = corr_df.rename(columns={'logfoldchanges': 'logFC_Corr', 'pvals_adj': 'padj_Corr'})

    merged = raw_df[['marker', 'logFC_Raw', 'padj_Raw']].merge(
        corr_df[['marker', 'logFC_Corr', 'padj_Corr']], on='marker'
    )

    # Sort by Raw logFC
    merged = merged.sort_values('logFC_Raw', ascending=False)

    print("\n" + "=" * 60)
    print("DE Comparison (Raw vs Corrected)")
    print("=" * 60)
    print(merged.to_string(index=False))

    # Save to CSV
    merged.to_csv(output_dir / 'de_comparison.csv', index=False)

    # Plot 1: Grouped bar plot of logFC
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(merged))
    width = 0.35

    bars1 = ax.bar(x - width/2, merged['logFC_Raw'], width, label='Raw (Patient4)',
                   color='#E64B35', edgecolor='white')
    bars2 = ax.bar(x + width/2, merged['logFC_Corr'], width, label='Corrected (Global)',
                   color='#4DBBD5', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(merged['marker'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Log Fold Change', fontsize=9)
    ax.set_title('Tumor Markers: Raw vs Corrected', fontsize=12)
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'de_logfc_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'de_logfc_comparison.png'}")
    plt.close()

    # Plot 2: Scatter plot logFC Raw vs Corrected
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(merged['logFC_Raw'], merged['logFC_Corr'], s=50, alpha=0.7, c='#3C5488')

    # Add marker labels
    for _, row in merged.iterrows():
        ax.annotate(row['marker'], (row['logFC_Raw'], row['logFC_Corr']),
                    fontsize=7, alpha=0.8)

    # Add diagonal line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)

    ax.set_xlabel('Log FC (Raw - Patient4)', fontsize=9)
    ax.set_ylabel('Log FC (Corrected - Global)', fontsize=9)
    ax.set_title('Tumor Marker Consistency', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'de_logfc_scatter.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'de_logfc_scatter.png'}")
    plt.close()

    return merged


def main():
    # Paths
    raw_path = '/home/w29632hl/code/BatchEffect/BioBatchNet/DATA/IMC/IMMUcan.h5ad'
    corrected_path = '/home/w29632hl/code/BatchEffect/BioBatchNet/src/saved/immucan/20260129_185122_disc0.3/seed_42/adata.h5ad'
    output_dir = Path('/home/w29632hl/code/BatchEffect/BioBatchNet/DATA/IMC/tumor_analysis')

    # Process Raw (Patient 4 only)
    de_raw, adata_de_raw, adata_raw = process_raw_patient4(raw_path)

    # Process Corrected (Global)
    de_corr, adata_de_corr, adata_corr = process_corrected_global(corrected_path)

    # Compare
    results = {'Raw': de_raw, 'Corrected': de_corr}
    merged = plot_comparison(results, output_dir)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
