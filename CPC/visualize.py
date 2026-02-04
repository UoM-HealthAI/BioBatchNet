import argparse
import os
import scanpy as sc
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Visualize CPC embeddings')
    parser.add_argument('--data_path', '-d', type=str, required=True, help='Path to adata file (.h5ad)')
    parser.add_argument('--label_key', type=str, default='celltype', help='Key in adata.obs for coloring')
    parser.add_argument('--save_path', '-o', type=str, default=None, help='Path to save figure')
    args = parser.parse_args()

    # Default save path: same folder as adata
    if args.save_path is None:
        filename = os.path.basename(args.data_path).replace('.h5ad', '_umap.png')
        args.save_path = os.path.join(os.path.dirname(args.data_path), filename)

    # Load adata
    adata = sc.read_h5ad(args.data_path)

    # Compute UMAP for X_biobatchnet
    print("Computing UMAP for X_biobatchnet...")
    adata_bio = sc.AnnData(adata.obsm['X_biobatchnet'])
    adata_bio.obs = adata.obs.copy()
    sc.pp.neighbors(adata_bio)
    sc.tl.umap(adata_bio)

    # Compute UMAP for X_cpc
    print("Computing UMAP for X_cpc...")
    adata_cpc = sc.AnnData(adata.obsm['X_cpc'])
    adata_cpc.obs = adata.obs.copy()
    sc.pp.neighbors(adata_cpc)
    sc.tl.umap(adata_cpc)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc.pl.umap(adata_bio, color=args.label_key, ax=axes[0], show=False,
               title='Bio embedding before CPC', frameon=False)
    sc.pl.umap(adata_cpc, color=args.label_key, ax=axes[1], show=False,
               title='Bio embedding after CPC', frameon=False)

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {args.save_path}")


if __name__ == '__main__':
    main()
