from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path
import scanpy as sc


class RBackend:
    """
    Run R-based baselines by:
    1) writing an AnnData to a temp .h5ad
    2) calling Rscript
    3) reading output .h5ad and copying specified obsm key back
    """

    def __init__(self, scripts_dir: str, logger, r_bin: str = "Rscript", conda_env: str | None = None):
        self.scripts_dir = Path(scripts_dir)
        self.logger = logger
        self.r_bin = r_bin
        self.conda_env = conda_env  # if not None, use `conda run -n <env> Rscript ...`

    def _run_rscript(self, script_name: str, in_h5ad: Path, out_h5ad: Path, extra_args=None):
        extra_args = extra_args or []
        script_path = self.scripts_dir / script_name

        if self.conda_env:
            cmd = ["conda", "run", "-n", self.conda_env, self.r_bin, str(script_path), str(in_h5ad), str(out_h5ad), *extra_args]
        else:
            cmd = [self.r_bin, str(script_path), str(in_h5ad), str(out_h5ad), *extra_args]

        self.logger.info("Running R: " + " ".join(cmd))
        subprocess.run(cmd, check=True)

    def run_and_copy_obsm(self, adata, script_name: str, out_obsm_key: str, extra_args=None):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            in_path = td / "in.h5ad"
            out_path = td / "out.h5ad"
            adata.write_h5ad(in_path)

            self._run_rscript(script_name, in_path, out_path, extra_args=extra_args)
            out = sc.read_h5ad(out_path)

        if out_obsm_key not in out.obsm:
            raise KeyError(f"R output missing obsm['{out_obsm_key}']")

        adata.obsm[out_obsm_key] = out.obsm[out_obsm_key]
        return adata

    # Convenience wrappers
    def seurat(self, adata, method: str):
        method = method.lower()
        key = f"X_seurat_{method}"
        return self.run_and_copy_obsm(adata, "run_seurat.R", key, extra_args=[method])

    def fastmnn(self, adata):
        return self.run_and_copy_obsm(adata, "run_fastmnn.R", "X_fastmnn")
