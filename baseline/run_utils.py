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
import os
from utils import logger
from config import BaselineConfig
from evaluation import evaluate


class RunBaseline:
    def __init__(self, adata, config, dataset_config):
        """
        Run baseline methods for batch effect correction.

        Args:
            adata: AnnData object
            config: BaselineConfig object
            dataset_config: DatasetConfig for the current dataset
        """
        self.raw_adata = adata.copy()
        self.config = config
        self.mode = dataset_config.mode
        self.seed_list = dataset_config.seed_list
        self.sampling_fraction = dataset_config.sampling_fraction
        self.sampling_seed = dataset_config.sampling_seed

        # Preprocess based on mode
        if self.mode == 'imc':
            self.process_adata = self.raw_adata
        else:
            self.process_adata = self._seq_process(self.raw_adata)

        self.features = self.process_adata.X
        self.batch = pd.Categorical(self.process_adata.obs['BATCH'].values)
        self.celltype = pd.Categorical(self.process_adata.obs['celltype'].values)
        self.timing_results = {}

    def _set_seed(self, seed):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _seq_process(self, adata):
        """Preprocess scRNA-seq data."""
        if issparse(adata.X):
            adata.X = adata.X.toarray()

        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='cell_ranger', subset=True)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return adata[:, adata.var['highly_variable']]

    def _create_adata(self):
        """Create AnnData from processed features."""
        features = self.features.toarray() if issparse(self.features) else self.features
        adata = sc.AnnData(features)
        adata.obs['BATCH'] = self.batch
        adata.obs['celltype'] = self.celltype
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
        else:
            raise ValueError(f"Unknown method: {method_name}")

        elapsed = time.time() - start_time
        logger.info(f"{method_name} time: {elapsed:.2f}s")
        return result, elapsed

    def train_all(self, seed):
        """
        Run all baseline methods once.

        Args:
            seed: Random seed for NN methods

        Returns:
            Dictionary of adata results for all methods
        """
        self._set_seed(seed)
        adata_base = self._create_adata()
        results = {"Raw": self.process_adata}

        # Run all methods from config
        for method_name in self.config.methods.keys():
            adata, elapsed = self._run_method(method_name, adata_base, seed)
            results[method_name] = adata
            self.timing_results[method_name] = elapsed

        gc.collect()
        return results

    def run_and_evaluate(self):
        """
        Run all methods with multiple seeds and evaluate.

        Returns:
            Tuple of (final_results, first_adata_dict, timing_results)
        """
        all_adata = None
        aggregated = {}
        timing_aggregated = {m: [] for m in self.config.methods.keys()}

        for i, seed in enumerate(self.seed_list):
            logger.info(f"Running with seed={seed} ({i+1}/{len(self.seed_list)})")

            adata_dict = self.train_all(seed)

            if i == 0:
                all_adata = adata_dict

            # Collect timing
            for method, elapsed in self.timing_results.items():
                timing_aggregated[method].append(elapsed)

            # Evaluate
            metrics = evaluate(
                adata_dict, self.config,
                fraction=self.sampling_fraction,
                seed=self.sampling_seed
            )

            for method, method_metrics in metrics.items():
                if method not in aggregated:
                    aggregated[method] = {m: [] for m in method_metrics}
                for metric, value in method_metrics.items():
                    aggregated[method][metric].append(value)

        # Calculate mean and std
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

        return final_results, all_adata, timing_stats

    def _run_scvi(self, adata):
        scvi.model.SCVI.setup_anndata(adata, batch_key="BATCH")
        likelihood = 'normal' if self.mode == 'imc' else 'zinb'
        model = scvi.model.SCVI(adata, gene_likelihood=likelihood)
        model.train(max_epochs=100)
        adata.obsm["X_scvi"] = model.get_latent_representation()
        return adata

    def _run_imap(self, adata, seed):
        if issparse(adata.X):
            raise ValueError("adata.X must be dense for iMAP")
        EC, ec_data = imap.stage1.iMAP_fast(adata, key="batch", n_epochs=150, seed=seed)
        output = imap.stage2.integrate_data(adata, ec_data, inc=False, n_epochs=150, seed=seed)
        result = sc.AnnData(output)
        result.obs['celltype'] = adata.obs['celltype'].values
        result.obs['BATCH'] = adata.obs['batch'].values
        sc.pp.pca(result)
        result.obsm['X_imap'] = result.obsm['X_pca']
        return result

    def _run_mrvi(self, adata):
        scvi.external.MRVI.setup_anndata(adata, sample_key='BATCH')
        likelihood = 'normal' if self.mode == 'imc' else 'zinb'
        model = scvi.external.MRVI(adata, gene_likelihood=likelihood)
        model.train(max_epochs=100)
        adata.obsm["X_mrvi"] = model.get_latent_representation()
        return adata

    def _run_harmony(self, adata):
        sc.pp.pca(adata)
        sc.external.pp.harmony_integrate(adata, 'BATCH')
        return adata

    def _run_bbknn(self, adata):
        sc.tl.pca(adata, svd_solver='arpack')
        bbknn.bbknn(adata, batch_key="BATCH")
        adata.obsm['X_bbknn'] = adata.obsm['X_pca']
        return adata

    def _run_scanorama(self, adata):
        adata = adata[adata.obs.sort_values('BATCH').index]
        sc.pp.pca(adata)
        sc.external.pp.scanorama_integrate(adata, key='BATCH')
        return adata

    def _run_combat(self, adata):
        sc.pp.combat(adata, key='BATCH')
        sc.pp.pca(adata)
        adata.obsm['X_combat'] = adata.obsm['X_pca']
        return adata
