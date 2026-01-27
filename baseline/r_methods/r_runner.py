from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path
import scanpy as sc
import pandas as pd


class RBackend:
    """
    Run R-based baselines by:
    1) writing an AnnData to a temp .h5ad
    2) calling Rscript
    3) reading output embedding file and copying it into obsm
    """

    def __init__(self, scripts_dir: str, logger, r_bin: str = "Rscript", conda_env: str | None = None):
        # Use absolute path relative to this file's location
        self.scripts_dir = Path(__file__).parent / scripts_dir if not Path(scripts_dir).is_absolute() else Path(scripts_dir)
        self.logger = logger
        self.r_bin = r_bin
        self.conda_env = conda_env  # if not None, use `conda run -n <env> Rscript ...`

    def _run_rscript(self, script_name: str, in_h5ad: Path, out_path: Path, extra_args=None):
        extra_args = extra_args or []
        script_path = self.scripts_dir / script_name

        if not script_path.exists():
            raise FileNotFoundError(f"R script not found: {script_path}")

        if self.conda_env:
            cmd = ["conda", "run", "-n", self.conda_env, self.r_bin, str(script_path), str(in_h5ad), str(out_path), *extra_args]
        else:
            cmd = [self.r_bin, str(script_path), str(in_h5ad), str(out_path), *extra_args]

        self.logger.info("Running R: " + " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.logger.error(f"R script failed:\n{result.stderr}")
            raise RuntimeError(f"R script failed (exit={result.returncode}): {result.stderr}")

        # Some wrappers may not propagate non-zero exit codes reliably; verify output exists.
        if not out_path.exists():
            msg = (
                f"R script finished but did not create output file: {out_path}\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}\n"
            )
            self.logger.error(msg)
            raise RuntimeError(msg)

    def run_and_copy_obsm(self, adata, script_name: str, out_obsm_key: str, extra_args=None):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            in_path = td / "in.h5ad"
            out_path = td / "out.tsv"
            
            # Clean adata to avoid _index column issue when writing h5ad
            adata.raw = None
            if "_index" in adata.var.columns:
                adata.var = adata.var.drop(columns=["_index"])
            if "_index" in adata.obs.columns:
                adata.obs = adata.obs.drop(columns=["_index"])
            adata.write_h5ad(in_path)

            self._run_rscript(script_name, in_path, out_path, extra_args=extra_args)
            emb_df = pd.read_csv(out_path, sep="\t", index_col=0)

        # Align by cell id (index) to avoid silent mismatches
        emb_df = emb_df.reindex(adata.obs_names)
        if emb_df.isna().any().any():
            missing = emb_df.index[emb_df.isna().any(axis=1)].tolist()[:10]
            raise ValueError(f"R embedding missing rows for some cells (showing up to 10): {missing}")

        adata.obsm[out_obsm_key] = emb_df.to_numpy()
        return adata

    # Convenience wrappers
    def seurat(self, adata, method: str):
        method = method.lower()
        key = f"X_seurat_{method}"
        return self.run_and_copy_obsm(adata, "run_seurat.R", key, extra_args=[method])

    def fastmnn(self, adata):
        return self.run_and_copy_obsm(adata, "run_fastmnn.R", "X_fastmnn")

    def combatseq(self, adata):
        return self.run_and_copy_obsm(adata, "run_combatseq.R", "X_combatseq")
