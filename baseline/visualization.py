import math
import os
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from pathlib import Path

from config import BaselineConfig
from utils import load_adata_from_dir
from utils import logger


OTHER_METHOD_EMBED = {
    "BioBatchNet": "X_biobatchnet",
}


def compute_neighbors(adata_dict, config):
    """
    Compute neighbors for UMAP visualization based on config.

    Args:
        adata_dict: Dictionary of adata objects
        config: BaselineConfig object
    """
    for method, adata in tqdm(adata_dict.items(), desc="Computing neighbors"):
        # Skip if neighbors already computed (e.g., BBKNN)
        if 'connectivities' in adata.obsp and 'distances' in adata.obsp:
            continue
        if method == 'Raw':
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3', subset=True)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.pca(adata)
            sc.pp.neighbors(adata, use_rep='X_pca')
        elif method in config.methods:
            embed = config.get_method(method).embed
            sc.pp.neighbors(adata, use_rep=embed)
        else:
            sc.pp.neighbors(adata, use_rep=OTHER_METHOD_EMBED[method])


def plot_umap(adata_dict, color, save_path=None):
    """
    Plot UMAP for all methods.

    Args:
        adata_dict: Dictionary of adata objects
        color: Column name for coloring ('BATCH' or 'celltype')
        save_path: Path to save the figure
    """
    # Raw first, others by dict order, BioBatchNet last (bottom-right)
    others = [k for k in adata_dict.keys() if k != 'Raw' and k != 'BioBatchNet']
    bio = ['BioBatchNet'] if 'BioBatchNet' in adata_dict else []
    methods = ['Raw'] + others + bio
    n_methods = len(methods)
    n_cols = 6
    n_rows = math.ceil(n_methods / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    # Get unique categories and create color map
    unique_categories = set()
    for method in methods:
        unique_categories.update(adata_dict[method].obs[color].unique())
    unique_categories = sorted(unique_categories)

    palette = sc.pl.palettes.default_20 if len(unique_categories) <= 20 else sc.pl.palettes.default_102
    color_map = {cat: palette[i] for i, cat in enumerate(unique_categories)}

    for i, method in enumerate(tqdm(methods, desc="Plotting UMAP")):
        ax = axes[i]
        adata = adata_dict[method]

        sc.tl.umap(adata)
        sc.pl.umap(
            adata, color=color, ax=ax, show=False,
            legend_loc=None, frameon=False, title=method,
            palette=[color_map[cat] for cat in unique_categories]
        )
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Remove extra axes
    for j in range(n_methods, len(axes)):
        fig.delaxes(axes[j])

    # Add legend
    handles = [Line2D([0], [0], marker='o', color='w', label=cat,
                      markerfacecolor=color_map[cat], markersize=10)
               for cat in unique_categories]
    fig.legend(handles, unique_categories, loc='upper center',
               ncol=min(len(unique_categories), 10), fontsize=12,
               bbox_to_anchor=(0.5, 0.05))

    plt.subplots_adjust(bottom=0.07, top=0.95, hspace=0.1, wspace=0.1)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved plot to {save_path}")

    plt.close()


def visualize(adata_dict, config, save_dir, other_methods=None):
    """
    Run full visualization pipeline.

    Args:
        adata_dict: Dictionary of adata objects
        config: BaselineConfig object
        save_dir: Directory to save plots
        other_methods: Dict of additional methods to include {'method': 'path.h5ad'}
    """
    # Integrate other methods if provided
    if other_methods:
        for method, path in other_methods.items():
            if os.path.exists(path):
                adata_dict[method] = sc.read_h5ad(path)
                logger.info(f"Loaded {method} from {path}")

    # Compute neighbors
    compute_neighbors(adata_dict, config)

    # Plot UMAP
    os.makedirs(save_dir, exist_ok=True)
    plot_umap(adata_dict, 'BATCH', os.path.join(save_dir, 'umap_batch.png'))
    plot_umap(adata_dict, 'celltype', os.path.join(save_dir, 'umap_celltype.png'))

    logger.info(f"Visualization completed. Plots saved to {save_dir}")


def main(config, adata_dir, other_methods=None):
    """
    Load adata from adata_dir and save UMAP plots in the same directory.

    Args:
        config: BaselineConfig object
        adata_dir: Directory containing h5ad files (plots saved here too)
        other_methods: Dict of additional methods {'method': 'path.h5ad'}
    """
    adata_dict = load_adata_from_dir(adata_dir)
    visualize(adata_dict, config, adata_dir, other_methods)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize baseline results')
    parser.add_argument('--adata_dir', '-d', type=str, required=True,
                       help='Directory containing h5ad files; plots saved here')
    args = parser.parse_args()

    logger.info("Visualization script started.")

    config_path = Path(__file__).parent / "config.yaml"
    config = BaselineConfig.load(config_path)

    main(config, args.adata_dir)
    logger.info("Visualization finished.")
