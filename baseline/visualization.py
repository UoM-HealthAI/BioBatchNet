"""
Single-file UMAP visualization: load h5ad from dir, plot BATCH and celltype per method.
No external project imports; logic inlined from BioBatchNet-main/baseline.
"""
import argparse
import logging
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scanpy as sc
from tqdm import tqdm

# Config: method name -> obsm key for embedding (from baseline config.yaml)
METHOD_EMBED = {
    "Harmony": "X_harmony",
    "BBKNN": "X_pca",
    "Scanorama": "X_scanorama",
    "Combat": "X_combat",
    "SeuratCCA": "X_seurat_cca",
    "SeuratRPCA": "X_seurat_rpca",
    "FastMNN": "X_fastmnn",
    "scVI": "X_scvi",
    "iMAP": "X_imap",
    "MrVI": "X_mrvi",
    "BioBatchNet": "X_biobatchnet",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LEGEND_Y = 0.05
SUBPLOTS_BOTTOM = 0.12
TITLE_FONTSIZE = 14


def load_adata_from_dir(save_dir):
    """Load all h5ad files from directory (from baseline utils)."""
    adata_dict = {}
    for f in Path(save_dir).glob("*.h5ad"):
        method = f.stem
        adata_dict[method] = sc.read_h5ad(f)
        logger.info(f"Loaded {method}")
    return adata_dict


def subsample_adata_dict(adata_dict, fraction=1.0, seed=42):
    """Subsample each adata by fraction (for faster UMAP). fraction=1.0 means no sampling."""
    if fraction >= 1.0:
        return adata_dict
    out = {}
    for method, adata in tqdm(adata_dict.items(), desc="Sampling"):
        out[method] = sc.pp.subsample(adata, fraction=fraction, random_state=seed, copy=True)
    logger.info(f"Subsampled to {fraction*100:.0f}% (seed={seed})")
    return out


def compute_neighbors(adata_dict):
    """Compute neighbors for UMAP: Raw -> PCA+neighbors; others -> use METHOD_EMBED.
    BBKNN (and similar) often have pre-computed obsp connectivities/distances -> skip and use them for UMAP.
    """
    for method, adata in tqdm(adata_dict.items(), desc="Computing neighbors"):
        if "connectivities" in adata.obsp and "distances" in adata.obsp:
            continue  # e.g. BBKNN: already has batch-aware graph
        if method == "Raw":
            sc.pp.pca(adata)
            sc.pp.neighbors(adata, use_rep="X_pca")
        else:
            rep = METHOD_EMBED.get(method, "X_biobatchnet")
            if rep not in adata.obsm.keys():
                rep = "X"  # e.g. iMAP may store embedding in adata.X
                logger.info(f"  {method}: use_rep='X' (obsm has {list(adata.obsm.keys())})")
            else:
                logger.info(f"  {method}: use_rep='{rep}'")
            sc.pp.neighbors(adata, use_rep=rep)


def plot_umap(adata_dict, color, save_path=None):
    """
    Plot one UMAP figure: one subplot per method, colored by `color` (BATCH or celltype).
    Layout: fixed 2 rows, columns = ceil(n_methods / 2), panel size ~ 6x4 (like your other script).
    """
    # Method ordering: Raw first, BioBatchNet last (if present), others in-between (keep current dict order)
    others = [k for k in adata_dict.keys() if k not in ("Raw", "BioBatchNet")]
    methods = (["Raw"] if "Raw" in adata_dict else []) + others + (["BioBatchNet"] if "BioBatchNet" in adata_dict else [])
    n_methods = len(methods)

    # ---- Layout: match "6 4" style ----
    n_rows = 2
    n_cols = math.ceil(n_methods / n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        constrained_layout=False
    )
    axes = axes.flatten()

    # ---- Palette (global, consistent across methods) ----
    unique_categories = set()
    for method in methods:
        unique_categories.update(adata_dict[method].obs[color].unique())
    unique_categories = sorted(unique_categories)

    palette = sc.pl.palettes.default_20 if len(unique_categories) <= 20 else sc.pl.palettes.default_102
    color_map = {cat: palette[i] for i, cat in enumerate(unique_categories)}

    # ---- Plot panels ----
    for i, method in enumerate(tqdm(methods, desc=f"Plotting UMAP ({color})")):
        ax = axes[i]
        adata = adata_dict[method]

        # Compute UMAP if not exists (avoid recompute if already computed)
        if "X_umap" not in adata.obsm_keys():
            sc.tl.umap(adata)

        sc.pl.umap(
            adata,
            color=color,
            ax=ax,
            show=False,
            legend_loc=None,
            frameon=False,
            title=method,
            palette=[color_map[cat] for cat in unique_categories],
        )
        ax.set_title(method, fontsize=TITLE_FONTSIZE)

        # Make panels visually consistent & less "stretched"
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        ax.margins(0.02)

    # Remove unused axes
    for j in range(n_methods, len(axes)):
        fig.delaxes(axes[j])

    # ---- Global legend (bottom, like your current style) ----
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
        bbox_to_anchor=(0.5, LEGEND_Y)
    )
    plt.subplots_adjust(bottom=SUBPLOTS_BOTTOM, top=0.95, hspace=0.10, wspace=0.05)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"Saved plot to {save_path}")

    plt.close(fig)


def visualize(adata_dict, save_dir, other_methods=None, fraction=1.0, seed=42):
    """Load optional other h5ad, optionally subsample, compute neighbors, save umap_batch.png and umap_celltype.png."""
    if other_methods:
        for method, path in other_methods.items():
            if os.path.exists(path):
                adata_dict[method] = sc.read_h5ad(path)
                logger.info(f"Loaded {method} from {path}")

    if fraction < 1.0:
        adata_dict = subsample_adata_dict(adata_dict, fraction=fraction, seed=seed)
    compute_neighbors(adata_dict)
    os.makedirs(save_dir, exist_ok=True)
    plot_umap(adata_dict, "BATCH", os.path.join(save_dir, "umap_batch.png"))
    plot_umap(adata_dict, "celltype", os.path.join(save_dir, "umap_celltype.png"))
    logger.info(f"Visualization completed. Plots saved to {save_dir}")


def main(adata_dir, other_methods=None, fraction=1.0, seed=42):
    """Load adata from adata_dir and save UMAP plots there."""
    adata_dict = load_adata_from_dir(adata_dir)
    visualize(adata_dict, adata_dir, other_methods, fraction=fraction, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize baseline results (all methods UMAP)")
    parser.add_argument("--adata_dir", "-d", type=str, required=True, help="Directory containing h5ad files; plots saved here")
    parser.add_argument("--fraction", "-f", type=float, default=0.3, help="Subsample fraction (0-1), default 1.0 = no sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    args = parser.parse_args()

    logger.info("Visualization script started.")
    main(args.adata_dir, fraction=args.fraction, seed=args.seed)
    logger.info("Visualization finished.")
