import math
import os
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from pathlib import Path

from config import BaselineConfig
from utils import load_all_adata, get_save_dir
from utils import logger


def compute_neighbors(adata_dict, config):
    """
    Compute neighbors for UMAP visualization based on config.

    Args:
        adata_dict: Dictionary of adata objects
        config: BaselineConfig object
    """
    for method, adata in tqdm(adata_dict.items(), desc="Computing neighbors"):
        if method == 'Raw':
            sc.pp.pca(adata)
            sc.pp.neighbors(adata, use_rep='X_pca')
        elif method in config.methods:
            embed = config.get_method(method).embed
            sc.pp.neighbors(adata, use_rep=embed)
        else:
            # For methods not in config (e.g., BioBatchNet, scDREAMER)
            # Try common embed names
            if f'X_{method.lower()}' in adata.obsm:
                sc.pp.neighbors(adata, use_rep=f'X_{method.lower()}')
            elif 'X_emb' in adata.obsm:
                sc.pp.neighbors(adata, use_rep='X_emb')
            elif 'X_pca' in adata.obsm:
                sc.pp.neighbors(adata, use_rep='X_pca')
            else:
                sc.pp.pca(adata)
                sc.pp.neighbors(adata, use_rep='X_pca')


def plot_umap(adata_dict, color, save_path=None):
    """
    Plot UMAP for all methods.

    Args:
        adata_dict: Dictionary of adata objects
        color: Column name for coloring ('BATCH' or 'celltype')
        save_path: Path to save the figure
    """
    methods = ['Raw'] + [k for k in adata_dict.keys() if k != 'Raw']
    n_methods = len(methods)
    n_rows = 2
    n_cols = math.ceil(n_methods / n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
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


def load_adata_from_dir(save_dir):
    """Load all h5ad files from directory."""
    adata_dict = {}
    for f in Path(save_dir).glob("*.h5ad"):
        method = f.stem
        adata_dict[method] = sc.read_h5ad(f)
        logger.info(f"Loaded {method}")
    return adata_dict


def main(config, save_dir, other_methods=None):
    """
    Load saved adata and visualize.

    Args:
        config: BaselineConfig object
        save_dir: Directory containing h5ad files
        other_methods: Dict of additional methods {'method': 'path.h5ad'}
    """
    # Load saved adata (try adata_results subdir first, then direct)
    adata_results_dir = os.path.join(save_dir, 'adata_results')
    if os.path.exists(adata_results_dir):
        adata_dict = load_all_adata(save_dir)
    else:
        adata_dict = load_adata_from_dir(save_dir)

    # Visualize
    vis_dir = os.path.join(save_dir, 'visualization')
    visualize(adata_dict, config, vis_dir, other_methods)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize baseline results')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='Directory containing h5ad files')
    args = parser.parse_args()

    logger.info("Visualization script started.")

    config_path = Path(__file__).parent / "config.yaml"
    config = BaselineConfig.load(config_path)

    main(config, args.save_dir)
    logger.info("Visualization finished.")
