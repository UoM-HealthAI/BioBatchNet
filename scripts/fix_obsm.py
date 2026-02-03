import scanpy as sc
import argparse
from pathlib import Path


def fix_obsm(adata_dir: str):
    """Fix obsm keys to match config expectations."""

    adata_dir = Path(adata_dir)

    # 1. Fix Combat: compute PCA -> X_combat
    combat_path = adata_dir / "Combat.h5ad"
    if combat_path.exists():
        adata = sc.read_h5ad(combat_path)
        if "X_combat" not in adata.obsm:
            sc.pp.pca(adata, n_comps=50)
            adata.obsm["X_combat"] = adata.obsm["X_pca"]
            del adata.obsm["X_pca"]
            adata.write(combat_path)
            print(f"Fixed Combat: added X_combat")

    # 2. Fix iMAP: compute PCA -> X_imap
    imap_path = adata_dir / "iMAP.h5ad"
    if imap_path.exists():
        adata = sc.read_h5ad(imap_path)
        if "X_imap" not in adata.obsm:
            sc.pp.pca(adata, n_comps=50)
            adata.obsm["X_imap"] = adata.obsm["X_pca"]
            del adata.obsm["X_pca"]
            adata.write(imap_path)
            print(f"Fixed iMAP: added X_imap")

    # 3. Fix Harmony: rename X_pca_harmony -> X_harmony
    harmony_path = adata_dir / "Harmony.h5ad"
    if harmony_path.exists():
        adata = sc.read_h5ad(harmony_path)
        if "X_pca_harmony" in adata.obsm and "X_harmony" not in adata.obsm:
            adata.obsm["X_harmony"] = adata.obsm["X_pca_harmony"]
            del adata.obsm["X_pca_harmony"]
            adata.write(harmony_path)
            print(f"Fixed Harmony: renamed X_pca_harmony -> X_harmony")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("adata_dir", type=str, help="Directory containing h5ad files")
    args = parser.parse_args()
    fix_obsm(args.adata_dir)
