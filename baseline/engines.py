# -- author: haiping liu
# -- date: 2025.1.2

import scanpy as sc
from scipy.sparse import issparse
import pandas as pd
import numpy as np
import torch
import gc
import time

from utils import logger, save_one_adata, append_method_result
from evaluation import evaluate
from r_methods import RBackend


class RunBaseline:
    def __init__(self, adata, config, dataset_config, save_dir=None):
        """
        Run baseline methods for batch effect correction.

        Args:
            adata: AnnData object
            config: BaselineConfig object
            dataset_config: DatasetConfig for the current dataset
            save_dir: Directory to save adata results (optional)
        """
        self.adata = adata.copy()
        self.config = config
        self.mode = dataset_config.mode
        self.seed_list = dataset_config.seed_list
        self.sampling_fraction = dataset_config.sampling_fraction
        self.sampling_seed = dataset_config.sampling_seed
        self.timing_results = {}
        self.save_dir = save_dir

        self.r_backend = RBackend(
            scripts_dir=".",
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
        elif method_name == 'MrVI':
            result = self._run_mrvi(adata_base.copy())
        elif method_name == 'Harmony':
            result = self._run_harmony(adata_base.copy())
        elif method_name == 'BBKNN':
            result = self._run_bbknn(adata_base.copy())
        elif method_name == 'Scanorama':
            result = self._run_scanorama(adata_base.copy())
        elif method_name == 'Combat':
            result = self._run_combat(adata_base.copy())
        # elif method_name == 'CombatSeq':
        #     result = self.r_backend.combatseq(adata_base.copy())
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
        Run all methods and evaluate.
        - Stochastic methods (neural networks): run multiple seeds, aggregate mean/std
        - Deterministic methods: run once, return results directly

        Returns:
            Tuple of (final_results, timing_stats, all_adata)
        """
        all_adata = {"Raw": self.adata.copy()}
        final_results = {}
        timing_stats = {}

        # Separate stochastic and deterministic methods
        stochastic_methods = []
        deterministic_methods = []
        for method_name, method_cfg in self.config.methods.items():
            if method_cfg.stochastic:
                stochastic_methods.append(method_name)
            else:
                deterministic_methods.append(method_name)

        # 1. Run deterministic methods (single run, evaluate immediately)
        if deterministic_methods:
            logger.info(f"Running deterministic methods: {deterministic_methods}")
            seed = self.seed_list[0]
            self._set_seed(seed)
            adata_base = self.adata.copy()

            for method_name in deterministic_methods:
                adata, elapsed = self._run_method(method_name, adata_base, seed)
                all_adata[method_name] = adata
                timing_stats[method_name] = {'mean': elapsed, 'std': 0.0, 'times': [elapsed]}

                # Evaluate and save immediately
                metrics = evaluate(
                    {"Raw": adata_base, method_name: adata}, self.config,
                    fraction=self.sampling_fraction, seed=self.sampling_seed
                )
                final_results[method_name] = {
                    metric: {'mean': value, 'std': 0.0}
                    for metric, value in metrics[method_name].items()
                }
                if self.save_dir:
                    save_one_adata(adata, method_name, self.save_dir)
                    append_method_result(
                        self.save_dir, method_name, final_results[method_name],
                        timing_stats[method_name]
                    )

        # 2. Run stochastic methods (multiple seeds, aggregate by method)
        if stochastic_methods:
            logger.info(f"Running stochastic methods: {stochastic_methods}")
            adata_base = self.adata.copy()

            for method_name in stochastic_methods:
                logger.info(f"Running {method_name} with {len(self.seed_list)} seeds")
                metrics_list = []
                times_list = []

                for i, seed in enumerate(self.seed_list):
                    logger.info(f"{method_name} seed={seed} ({i+1}/{len(self.seed_list)})")
                    self._set_seed(seed)
                    adata, elapsed = self._run_method(method_name, adata_base.copy(), seed)
                    times_list.append(elapsed)

                    # Evaluate this seed
                    metrics = evaluate(
                        {"Raw": adata_base, method_name: adata}, self.config,
                        fraction=self.sampling_fraction, seed=self.sampling_seed
                    )
                    metrics_list.append(metrics[method_name])

                    # Keep first seed's adata for visualization
                    if i == 0:
                        all_adata[method_name] = adata

                    gc.collect()

                # Calculate mean/std and save immediately
                final_results[method_name] = {
                    metric: {'mean': np.mean([m[metric] for m in metrics_list]),
                             'std': np.std([m[metric] for m in metrics_list])}
                    for metric in metrics_list[0].keys()
                }
                timing_stats[method_name] = {
                    'mean': np.mean(times_list),
                    'std': np.std(times_list),
                    'times': times_list
                }
                logger.info(f"{method_name} timing - mean: {np.mean(times_list):.2f}s, std: {np.std(times_list):.2f}s")

                if self.save_dir:
                    save_one_adata(all_adata[method_name], method_name, self.save_dir)
                    append_method_result(
                        self.save_dir, method_name, final_results[method_name],
                        timing_stats[method_name]
                    )

        return final_results, timing_stats, all_adata

    def _run_scvi(self, adata):
        import scvi
        adata = self._ensure_counts_layer(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", subset=True, layer="counts")
        scvi.model.SCVI.setup_anndata(adata, batch_key="BATCH", layer="counts")
        model = scvi.model.SCVI(adata, gene_likelihood=("normal" if self.mode=="imc" else "zinb"))
        model.train(max_epochs=400)
        adata.obsm["X_scvi"] = model.get_latent_representation()
        return adata
    
    def _run_mrvi(self, adata):
        from scvi.external import MRVI
        adata = self._ensure_counts_layer(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", subset=True, layer="counts")
        MRVI.setup_anndata(adata, sample_key="BATCH", layer="counts")
        model = MRVI(adata)
        model.train(max_epochs=400)
        adata.obsm["X_mrvi"] = model.get_latent_representation()
        return adata

    def _run_imap(self, adata, seed):
        import imap
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
        import harmonypy as hm
        adata = self._seq_process(adata)
        sc.pp.pca(adata)
        ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, 'BATCH')
        adata.obsm["X_harmony"] = ho.Z_corr
        return adata

    def _run_bbknn(self, adata):
        import bbknn
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
