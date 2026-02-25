"""
Plot UMAP comparison across different disentanglement lambda values.
Creates two figures: one colored by batch, one colored by celltype.
"""
import math
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import argparse

LEGEND_Y = -0.05
SUBPLOTS_BOTTOM = 0.05
TITLE_FONTSIZE = 14


def plot_lambda_comparison(base_dir: str, output_dir: str = None, lambdas: list = None, raw_path: str = None, preprocess: bool = False):
    """
    Plot UMAP for different lambda (disc) values.

    Args:
        base_dir: Directory containing disc* subdirectories
        output_dir: Where to save figures (defaults to base_dir)
        lambdas: Optional list of lambda values to include (filters others out)
        raw_path: Optional path to raw h5ad file (plotted as leftmost panel)
    """
    base_dir = Path(base_dir)
    output_dir = Path(output_dir) if output_dir else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all disc directories and extract lambda values
    disc_dirs = []
    for d in base_dir.iterdir():
        if d.is_dir() and '_disc' in d.name:
            # Extract lambda value from directory name like "20260129_185122_disc0.1"
            lambda_str = d.name.split('_disc')[-1]
            try:
                lambda_val = float(lambda_str)
                if lambdas is not None and lambda_val not in lambdas:
                    continue
                disc_dirs.append((lambda_val, d))
            except ValueError:
                continue

    # Sort by lambda value
    disc_dirs.sort(key=lambda x: x[0])

    if not disc_dirs:
        print(f"No disc directories found in {base_dir}")
        return

    print(f"Found {len(disc_dirs)} lambda values: {[d[0] for d in disc_dirs]}")

    # Load all adata files
    adata_list = []
    lambda_vals = []
    for lambda_val, d in disc_dirs:
        adata_path = d / "seed_42" / "biobatchnet.h5ad"
        if not adata_path.exists():
            adata_path = d / "seed_42" / "adata.h5ad"
        if adata_path.exists():
            adata = sc.read_h5ad(adata_path)
            adata_list.append(adata)
            lambda_vals.append(lambda_val)
            print(f"Loaded λ={lambda_val}")
        else:
            print(f"Warning: {adata_path} not found")

    if not adata_list:
        print("No adata files found")
        return

    # Prepend raw data if provided
    if raw_path:
        from scipy.sparse import issparse
        adata_raw = sc.read_h5ad(raw_path)
        # Apply seq_preprocess for scRNA-seq data (HVG + normalize + log1p)
        if preprocess:
            from sys import path as sys_path
            sys_path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
            from utils.tools import seq_preprocess
            adata_raw = seq_preprocess(adata_raw)
            print("Applied seq_preprocess to raw data")
        X = adata_raw.X.toarray() if issparse(adata_raw.X) else adata_raw.X
        adata_raw.obsm['X_expr'] = X
        sc.pp.neighbors(adata_raw, use_rep='X_expr')
        sc.tl.umap(adata_raw)
        sc.tl.leiden(adata_raw, key_added='leiden_0.6', resolution=0.6, random_state=42)
        adata_list.insert(0, adata_raw)
        lambda_vals.insert(0, None)  # sentinel for raw
        print("Loaded raw data")

    n_plots = len(adata_list)

    # Compute UMAP for all (skip raw, already computed above)
    for i, adata in enumerate(adata_list):
        if lambda_vals[i] is None:
            continue
        if 'X_umap' not in adata.obsm:
            sc.pp.neighbors(adata, use_rep='X_biobatchnet')
            sc.tl.umap(adata)
        sc.tl.leiden(adata, key_added='leiden_0.6', resolution=0.6, random_state=42)

    # Plot function: single row of UMAPs for a given color_key
    def make_figure(color_key: str, save_name: str):
        fig, axes = plt.subplots(
            1, n_plots,
            figsize=(6 * n_plots, 4),
            constrained_layout=False,
        )
        if n_plots == 1:
            axes = [axes]

        # Get consistent color palette across all plots
        unique_categories = set()
        for adata in adata_list:
            unique_categories.update(adata.obs[color_key].unique())
        unique_categories = sorted(unique_categories)

        palette = sc.pl.palettes.default_20 if len(unique_categories) <= 20 else sc.pl.palettes.default_102
        color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(unique_categories)}

        for i, (adata, lambda_val) in enumerate(zip(adata_list, lambda_vals)):
            ax = axes[i]
            title = "Raw" if lambda_val is None else f"λ = {lambda_val}"
            sc.pl.umap(
                adata, color=color_key, ax=ax, show=False,
                legend_loc=None, frameon=False,
                title=title,
                palette=[color_map[cat] for cat in unique_categories],
            )
            ax.set_title(title, fontsize=TITLE_FONTSIZE)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            ax.margins(0.02)

        # Add legend at bottom
        handles = [
            Line2D([0], [0], marker="o", color="w", label=cat,
                   markerfacecolor=color_map[cat], markersize=8)
            for cat in unique_categories
        ]
        fig.legend(
            handles, unique_categories,
            loc="lower center",
            ncol=min(len(unique_categories), 10),
            fontsize=TITLE_FONTSIZE,
            bbox_to_anchor=(0.5, LEGEND_Y),
        )
        plt.subplots_adjust(bottom=SUBPLOTS_BOTTOM, top=0.92, wspace=0.05)

        save_path = output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_path}")
        plt.close(fig)

    # Make separate figures
    make_figure('BATCH', 'lambda_comparison_batch.png')
    make_figure('celltype', 'lambda_comparison_celltype.png')
    make_figure('leiden_0.6', 'lambda_comparison_leiden.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot UMAP comparison across lambda values')
    parser.add_argument('base_dir', type=str, help='Directory containing disc* subdirectories')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output directory (default: base_dir)')
    parser.add_argument('--lambdas', '-l', type=float, nargs='+', default=None, help='Specific lambda values to include')
    parser.add_argument('--raw', '-r', type=str, default=None, help='Path to raw h5ad file (plotted as leftmost panel)')
    parser.add_argument('--preprocess', '-p', action='store_true', help='Apply seq_preprocess (HVG+normalize+log1p) to raw data')
    args = parser.parse_args()

    plot_lambda_comparison(args.base_dir, args.output, args.lambdas, args.raw, args.preprocess)
