# -- author: haiping liu
# -- date: 2025.1.2

import scanpy as sc
from scipy.sparse import issparse
import scvi
import bbknn
import imap
import pandas as pd
import numpy as np
import torch
import gc
import time
from inmoose.pycombat import pycombat_seq

from utils import logger
from config import BaselineConfig
from evaluation import evaluate
from r_methods import RBackend


class RunBaseline:
    def __init__(self, adata, config, dataset_config):
        """
        Run baseline methods for batch effect correction.

        Args:
            adata: AnnData object
            config: BaselineConfig object
            dataset_config: DatasetConfig for the current dataset
        """
        self.adata = adata.copy()
        self.config = config
        self.mode = dataset_config.mode
        self.seed_list = dataset_config.seed_list
        self.sampling_fraction = dataset_config.sampling_fraction
        self.sampling_seed = dataset_config.sampling_seed
        self.timing_results = {}

        self.r_backend = RBackend(
            scripts_dir="scripts_r",
            logger=logger,
            r_bin="Rscript",
            conda_env="r_baselines",   
        )

    def _set_seed(self, seed):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _ensure_counts_layer(self, adata, counts_layer="counts"):
        if counts_layer not in adata.layers:
            adata.layers[counts_layer] = adata.X.copy()
        return adata

    def _seq_process(self, adata):
        """Preprocess scRNA-seq data."""
        adata = adata.copy()
        if issparse(adata.X):
            adata.X = adata.X.toarray()
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='cell_ranger', subset=True)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return adata

    def _run_method(self, method_name, adata_base, seed=None):
        """Run a single method and return result with timing."""
        start_time = time.time()

        if method_name == 'scVI':
            result = self._run_scvi(adata_base.copy())
        elif method_name == 'iMAP':
            adata_imap = adata_base.copy()
            adata_imap.obs['batch'] = adata_imap.obs['BATCH']
            result = self._run_imap(adata_imap, seed)
        elif method_name == 'MRVI':
            result = self._run_mrvi(adata_base.copy())
        elif method_name == 'Harmony':
            result = self._run_harmony(adata_base.copy())
        elif method_name == 'BBKNN':
            result = self._run_bbknn(adata_base.copy())
        elif method_name == 'Scanorama':
            result = self._run_scanorama(adata_base.copy())
        elif method_name == 'Combat':
            result = self._run_combat(adata_base.copy())
        elif method_name == 'CombatSeq':
            result = self._run_combatseq(adata_base.copy())
        elif method_name == "SeuratCCA":
            result = self.r_backend.seurat(adata_base.copy(), method="cca")
        elif method_name == "SeuratRPCA":
            result = self.r_backend.seurat(adata_base.copy(), method="rpca")
        elif method_name == "FastMNN":
            result = self.r_backend.fastmnn(adata_base.copy())
        else:
            raise ValueError(f"Unknown method: {method_name}")

        elapsed = time.time() - start_time
        logger.info(f"{method_name} time: {elapsed:.2f}s")
        return result, elapsed

    def run_one_method(self, method_name, seed):
        """
        Run a single method for one seed without evaluation.
        """
        if method_name not in self.config.methods.keys():
            raise ValueError(f"Method '{method_name}' not found. Available methods: {list(self.config.methods.keys())}")
        
        self._set_seed(seed)
        adata_base = self.adata.copy()
        results = {"Raw": adata_base}

        # Run only the specified method
        adata, elapsed = self._run_method(method_name, adata_base, seed)
        results[method_name] = adata
        self.timing_results[method_name] = elapsed

        gc.collect()
        return results, self.timing_results.copy()

    def run_one_seed(self, seed):
        """
        Run all methods for one seed without evaluation.
        """
        self._set_seed(seed)
        adata_base = self.adata.copy()
        results = {"Raw": adata_base}

        # Run all methods from config
        for method_name in self.config.methods.keys():
            adata, elapsed = self._run_method(method_name, adata_base, seed)
            results[method_name] = adata
            self.timing_results[method_name] = elapsed

        gc.collect()
        return results, self.timing_results.copy()

    def run_all_seeds_and_evaluate(self):
        """
        Run all methods with multiple seeds and evaluate.
        Only keeps first seed's adata for visualization, discards others to save memory.
        
        Returns:
            Tuple of (final_results, timing_stats, first_adata_dict)
        """
        first_adata = None
        aggregated = {}
        timing_aggregated = {m: [] for m in self.config.methods.keys()}

        for i, seed in enumerate(self.seed_list):
            logger.info(f"Running with seed={seed} ({i+1}/{len(self.seed_list)})")

            adata_dict, timing = self.run_one_seed(seed)

            # Only keep first seed's adata for visualization
            if i == 0:
                first_adata = adata_dict

            # Collect timing
            for method, elapsed in timing.items():
                timing_aggregated[method].append(elapsed)

            # Evaluate immediately (don't keep adata for other seeds)
            metrics = evaluate(
                adata_dict, self.config,
                fraction=self.sampling_fraction,
                seed=self.sampling_seed
            )

            # Aggregate metrics
            for method, method_metrics in metrics.items():
                if method not in aggregated:
                    aggregated[method] = {m: [] for m in method_metrics}
                for metric, value in method_metrics.items():
                    aggregated[method][metric].append(value)

        # Calculate final statistics
        final_results = {}
        for method, metric_dict in aggregated.items():
            final_results[method] = {
                metric: {'mean': np.mean(values), 'std': np.std(values)}
                for metric, values in metric_dict.items()
            }

        # Calculate timing stats
        timing_stats = {}
        for method, times in timing_aggregated.items():
            timing_stats[method] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'times': times
            }
            logger.info(f"{method} timing - mean: {np.mean(times):.2f}s, std: {np.std(times):.2f}s")

        return final_results, timing_stats, first_adata

    def _run_scvi(self, adata):
        adata = self.ensure_counts_layer(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", subset=True, layer="counts")
        scvi.model.SCVI.setup_anndata(adata, batch_key="BATCH", layer="counts")
        model = scvi.model.SCVI(adata, gene_likelihood=("normal" if self.mode=="imc" else "zinb"))
        model.train(max_epochs=100)
        adata.obsm["X_scvi"] = model.get_latent_representation()
        return adata
    
    def _run_mrvi(self, adata):
        adata = self.ensure_counts_layer(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3",
                                    subset=True, layer="counts")
        scvi.external.MRVI.setup_anndata(adata, sample_key="BATCH", layer="counts")
        model = scvi.external.MRVI(adata, gene_likelihood=("normal" if self.mode=="imc" else "zinb"))
        model.train(max_epochs=100)
        adata.obsm["X_mrvi"] = model.get_latent_representation()
        return adata

    def _run_imap(self, adata, seed):
        adata = self._seq_process(adata)
        EC, ec_data = imap.stage1.iMAP_fast(adata, key="batch", n_epochs=150, seed=seed)
        output = imap.stage2.integrate_data(adata, ec_data, inc=False, n_epochs=150, seed=seed)
        result = sc.AnnData(output)
        result.obs['celltype'] = adata.obs['celltype'].values
        result.obs['BATCH'] = adata.obs['batch'].values
        sc.pp.pca(result)
        result.obsm['X_imap'] = result.obsm['X_pca']
        return result

    def _run_harmony(self, adata):
        adata = self._seq_process(adata)
        sc.pp.pca(adata)
        sc.external.pp.harmony_integrate(adata, 'BATCH')
        adata.obsm["X_harmony"] = adata.obsm["X_pca_harmony"]
        return adata

    def _run_bbknn(self, adata):
        adata = self._seq_process(adata)
        sc.tl.pca(adata, svd_solver='arpack')
        bbknn.bbknn(adata, batch_key="BATCH")
        adata.obsm['X_bbknn'] = adata.obsm['X_pca']
        return adata

    def _run_scanorama(self, adata):
        adata = self._seq_process(adata)
        adata = adata[adata.obs.sort_values('BATCH').index]
        sc.pp.pca(adata)
        sc.external.pp.scanorama_integrate(adata, key='BATCH')
        return adata

    def _run_combat(self, adata):
        adata = self._seq_process(adata)
        sc.pp.combat(adata, key='BATCH')
        sc.pp.pca(adata)
        adata.obsm['X_combat'] = adata.obsm['X_pca']
        return adata

    def _run_combatseq(
        self,
        adata,
        batch_key="BATCH",
        counts_layer="counts",
        out_rep="X_combatseq",
    ):

        adata = adata.copy()

        tmp = adata.copy()
        tmp.X = tmp.layers[counts_layer]
        sc.pp.highly_variable_genes(tmp, n_top_genes=2000, flavor="seurat_v3", subset=False)
        hvg_genes = tmp.var_names[tmp.var["highly_variable"]]

        # counts (cell×gene) -> (gene×cell) -> combatseq -> corrected (cell×gene)
        ad_hvg = adata[:, hvg_genes].copy()
        X = ad_hvg.layers[counts_layer]
        X = X.toarray() if issparse(X) else np.asarray(X)
        batch = ad_hvg.obs[batch_key].astype(str).to_numpy()

        corrected = np.asarray(pycombat_seq(counts=X.T, batch=batch)).T

        # embedding: normalize+log1p+pca on corrected counts
        work = sc.AnnData(X=corrected, obs=ad_hvg.obs.copy(), var=ad_hvg.var.copy())
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)
        sc.pp.pca(work, n_comps=50)

        adata.obsm[out_rep] = work.obsm["X_pca"]
        return adata
