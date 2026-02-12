"""
Plot UMAP comparison across different disentanglement lambda values.
Creates two figures: one colored by batch, one colored by celltype.
"""
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_lambda_comparison(base_dir: str, output_dir: str = None, lambdas: list = None):
    """
    Plot UMAP for different lambda (disc) values.

    Args:
        base_dir: Directory containing disc* subdirectories
        output_dir: Where to save figures (defaults to base_dir)
        lambdas: Optional list of lambda values to include (filters others out)
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

    n_plots = len(adata_list)

    # Compute UMAP for all
    for adata in adata_list:
        if 'X_umap' not in adata.obsm:
            sc.pp.neighbors(adata, use_rep='X_biobatchnet')
            sc.tl.umap(adata)

    # Plot function for a given color
    def make_figure(color_key: str, save_name: str):
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
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
            sc.pl.umap(
                adata, color=color_key, ax=ax, show=False,
                legend_loc=None, frameon=False,
                title=f"λ = {lambda_val}",
                palette=[color_map[cat] for cat in unique_categories]
            )
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(f"λ = {lambda_val}", fontsize=12)

        # Add legend at bottom
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker='o', color='w', label=cat,
                          markerfacecolor=color_map[cat], markersize=8)
                   for cat in unique_categories]
        fig.legend(handles, unique_categories, loc='upper center',
                   ncol=min(len(unique_categories), 10), fontsize=10,
                   bbox_to_anchor=(0.5, 0.08))

        plt.subplots_adjust(bottom=0.15, top=0.9, wspace=0.1)

        save_path = output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_path}")
        plt.close()

    # Make both figures
    make_figure('BATCH', 'lambda_comparison_batch.png')
    make_figure('celltype', 'lambda_comparison_celltype.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot UMAP comparison across lambda values')
    parser.add_argument('base_dir', type=str, help='Directory containing disc* subdirectories')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output directory (default: base_dir)')
    parser.add_argument('--lambdas', '-l', type=float, nargs='+', default=None, help='Specific lambda values to include')
    args = parser.parse_args()

    plot_lambda_comparison(args.base_dir, args.output, args.lambdas)
