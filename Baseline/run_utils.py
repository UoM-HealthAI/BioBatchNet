# -- author: haiping liu
# -- date: 2025.1.2

import scanpy as sc
from scipy.sparse import issparse
import scvi
import bbknn
import imap
import pandas as pd
import gc
import time
from logger_config import logger

class RunBaseline:
    def __init__(self, adata, mode, seed=None):
        """
        Run baseline methods for IMC and scRNA-seq data
        """
        self.raw_adata = adata.copy()
        self.mode = mode
        self.process_adata = self.raw_adata if self.mode == 'imc' else self.rna_process(self.raw_adata)

        self.features = self.process_adata.X
        self.batch = pd.Categorical(self.process_adata.obs['BATCH'].values)
        self.celltype = pd.Categorical(self.process_adata.obs['celltype'].values)
        self.timing_results = {}
        self.seed = seed

    def train_nn(self):
        """
        Train batch effect correction using neural network (NN)-based methods:
        - scVI
        - iMAP
        - MRVI
        """
        adata_base = RunBaseline.create_adata(self.features, self.batch, self.celltype)    

        # Run and time scVI
        start_time = time.time()
        output_scvi = run_scvi(adata_base.copy(), mode=self.mode)
        end_time = time.time()
        scvi_time = end_time - start_time
        logger.info(f"scVI time: {scvi_time:.2f}s")
        self.timing_results['scVI'] = scvi_time

        # Run and time iMAP
        adata_imap = adata_base.copy()
        adata_imap.obs['batch'] = adata_imap.obs['BATCH']
        
        start_time = time.time()
        output_imap = run_imap(adata_imap, self.seed)
        end_time = time.time()
        imap_time = end_time - start_time
        logger.info(f"iMAP time: {imap_time:.2f}s")
        self.timing_results['iMAP'] = imap_time

        # Run and time MRVI
        start_time = time.time()
        output_mrvi = run_mrvi(adata_base.copy(), mode=self.mode)
        end_time = time.time()
        mrvi_time = end_time - start_time
        logger.info(f"MRVI time: {mrvi_time:.2f}s")
        self.timing_results['MRVI'] = mrvi_time

        gc.collect()
        return {"Raw": self.process_adata,
                "scVI": output_scvi, 
                "iMAP": output_imap,
                "MRVI": output_mrvi}
    
    def train_non_nn(self):
        """
        Run non-NN methods:
        - Harmony
        - BBKNN
        - Scanorama
        - Combat
        """
        adata_base = RunBaseline.create_adata(self.features, self.batch, self.celltype)
        outputs = {"Raw": self.process_adata}
        
        # Run and time Harmony
        start_time = time.time()
        outputs["Harmony"] = run_harmony(adata_base.copy())
        end_time = time.time()
        harmony_time = end_time - start_time
        logger.info(f"Harmony time: {harmony_time:.2f}s")
        self.timing_results['Harmony'] = harmony_time

        # Run and time BBKNN
        start_time = time.time()
        outputs["BBKNN"] = run_bbknn(adata_base.copy())
        end_time = time.time()
        bbknn_time = end_time - start_time
        logger.info(f"BBKNN time: {bbknn_time:.2f}s")
        self.timing_results['BBKNN'] = bbknn_time

        # Run and time Scanorama
        start_time = time.time()
        outputs["Scanorama"] = run_scanorama(adata_base.copy())
        end_time = time.time()
        scanorama_time = end_time - start_time
        logger.info(f"Scanorama time: {scanorama_time:.2f}s")
        self.timing_results['Scanorama'] = scanorama_time

        # Run and time Combat
        start_time = time.time()
        outputs["Combat"] = run_combat(adata_base.copy())
        end_time = time.time()
        combat_time = end_time - start_time
        logger.info(f"Combat time: {combat_time:.2f}s")
        self.timing_results['Combat'] = combat_time

        return outputs

    def get_timing_results(self):
        """
        Get the timing results for all methods
        """
        return self.timing_results

    @staticmethod
    def rna_process(adata):
        if issparse(adata.X):
            adata.X = adata.X.toarray()
        else:
            adata.X = adata.X
            
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='cell_ranger', subset=True)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        processed_adata = adata[:, adata.var['highly_variable']]
        return processed_adata

    @staticmethod
    def create_adata(features, batch, celltype):
        if issparse(features):
            features = features.toarray()
        adata = sc.AnnData(features)
        adata.obs['BATCH'] = batch
        adata.obs['celltype'] = celltype
        return adata

def run_scvi(adata_scvi, mode):
    scvi.model.SCVI.setup_anndata(adata_scvi, batch_key="BATCH")  
    if mode == 'imc':
        model = scvi.model.SCVI(adata_scvi, gene_likelihood='normal')
    else:
        model = scvi.model.SCVI(adata_scvi, gene_likelihood='zinb')
    model.train(max_epochs=100)
    latent = model.get_latent_representation()
    adata_scvi.obsm["X_scvi"] = latent
    return adata_scvi

def run_imap(adata_imap, seed):
    if issparse(adata_imap.X):
        raise ValueError("adata_imap.X is sparse")
    EC, ec_data = imap.stage1.iMAP_fast(adata_imap, key="batch", n_epochs=150, seed=seed) 
    output_results = imap.stage2.integrate_data(adata_imap, ec_data, inc=False, n_epochs=150, seed=seed)
    output_imap = sc.AnnData(output_results)
    output_imap.obs['celltype'] = adata_imap.obs['celltype'].values
    output_imap.obs['BATCH'] = adata_imap.obs['batch'].values
    return output_imap

def run_mrvi(adata_mrvi, mode):
    scvi.external.MRVI.setup_anndata(adata_mrvi, sample_key='BATCH')
    if mode == 'imc':
        model = scvi.external.MRVI(adata_mrvi, gene_likelihood='normal')
    else:
        model = scvi.external.MRVI(adata_mrvi, gene_likelihood='zinb')
    model.train(max_epochs=100)
    latent = model.get_latent_representation()
    adata_mrvi.obsm["X_mrvi"] = latent
    return adata_mrvi

def run_harmony(adata_harm):
    sc.pp.pca(adata_harm)
    sc.external.pp.harmony_integrate(adata_harm, 'BATCH')
    return adata_harm

def run_bbknn(adata_bbknn):
    sc.tl.pca(adata_bbknn, svd_solver='arpack')
    bbknn.bbknn(adata_bbknn, batch_key="BATCH")
    return adata_bbknn

def run_scanorama(adata_scanorama):
    adata_scanorama = adata_scanorama[adata_scanorama.obs.sort_values('BATCH').index]
    sc.pp.pca(adata_scanorama)
    sc.external.pp.scanorama_integrate(adata_scanorama, key='BATCH')
    return adata_scanorama

def run_combat(adata_combat):
    sc.pp.combat(adata_combat, key='BATCH')
    return adata_combat
    
    



