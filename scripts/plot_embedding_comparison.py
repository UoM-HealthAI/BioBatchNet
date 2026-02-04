#!/usr/bin/env python
"""Plot UMAP comparison of X_batch and X_biobatchnet embeddings."""

import argparse
from pathlib import Path

import anndata as ad
import scanpy as sc


def main():
    parser = argparse.ArgumentParser(description='Plot UMAP comparison')
    parser.add_argument('--h5ad_path', type=str, help='Path to h5ad file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--batch-key', type=str, default='BATCH', help='Batch column name')
    parser.add_argument('--celltype-key', type=str, default='celltype', help='Celltype column name')
    args = parser.parse_args()

    h5ad_path = Path(args.h5ad_path)
    if args.output_dir is None:
        output_dir = h5ad_path.parent
    else:
        output_dir = Path(args.output_dir)

    # Load adata
    adata = sc.read_h5ad(h5ad_path)

    sc.pp.subsample(adata, fraction=0.3)
    batch_key = args.batch_key
    label_key = args.celltype_key

    # Extract embeddings
    X_batch = adata.obsm['X_batch']
    X_biobatchnet = adata.obsm['X_biobatchnet']

    # Create temporary AnnData for each embedding
    adata_batch = ad.AnnData(X_batch)
    adata_batch.obs[batch_key] = adata.obs[batch_key].values
    adata_batch.obs[label_key] = adata.obs[label_key].values

    adata_bio = ad.AnnData(X_biobatchnet)
    adata_bio.obs[batch_key] = adata.obs[batch_key].values
    adata_bio.obs[label_key] = adata.obs[label_key].values

    # Compute neighbors and UMAP
    sc.pp.neighbors(adata_batch)
    sc.tl.umap(adata_batch)
    sc.pp.neighbors(adata_bio)
    sc.tl.umap(adata_bio)

    # Set output directory
    sc.settings.figdir = output_dir

    # Plot batch embedding
    sc.pl.umap(adata_batch, color=[batch_key, label_key], frameon=False,
               title=['Batch latent - Batch', 'Batch latent - Cell Type'],
               save='_batch_embedding.png', show=False)

    # Plot bio embedding
    sc.pl.umap(adata_bio, color=[batch_key, label_key], frameon=False,
               title=['Bio latent - Batch', 'Bio latent - Cell Type'],
               save='_bio_embedding.png', show=False)

    print(f'Saved to {output_dir}')


if __name__ == '__main__':
    main()
