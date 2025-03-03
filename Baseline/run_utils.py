# -- author: haiping liu
# -- date: 2025.1.2

import scanpy as sc
from scipy.sparse import issparse
import scvi
import bbknn
import imap
import pandas as pd
import gc
from logger_config import logger

class RunBaseline:
    def __init__(self, adata, mode):
        """
        Run baseline methods for IMC and scRNA-seq data
        """
        self.raw_adata = adata.copy()
        self.mode = mode
        self.process_adata = self.raw_adata if self.mode == 'imc' else self.rna_process(self.raw_adata)
        self.features = self.process_adata.X
        self.batch = pd.Categorical(self.process_adata.obs['BATCH'].values)
        self.celltype = pd.Categorical(self.process_adata.obs['celltype'].values)

    def run_nn(self):
        """
        Run batch effect correction using neural network (NN)-based methods:
        - scVI
        - iMAP
        """
        adata_base = RunBaseline.create_adata(self.features, self.batch, self.celltype)

        adata_imap = self.raw_adata.copy()
        adata_imap.obs['batch'] = self.batch.astype("category") 
        adata_imap.obs['celltype'] = self.celltype.astype("category")
        
        output_scvi = run_scvi(adata_base, mode=self.mode)
        logger.info("run scvi finished")

        logger.info("begin to run imap")
        output_imap = run_imap(adata_imap, mode=self.mode)
        logger.info("run imap finished")

        gc.collect()
        return {"Raw": self.process_adata,
                "scVI": output_scvi, 
                "iMAP": output_imap}
    
    def run_non_nn(self):
        """
        Run non-NN methods:
        - Harmony
        - BBKNN
        - Scanorama
        - Combat
        """
        adata_base = RunBaseline.create_adata(self.features, self.batch, self.celltype)
        outputs = {
            "Raw": self.process_adata,
            "Harmony": run_harmony(adata_base.copy()),
            "BBKNN": run_bbknn(adata_base.copy()),
            "Scanorama": run_scanorama(adata_base.copy()),
            "Combat": run_combat(adata_base.copy()),
        }
        return outputs

    @staticmethod
    def rna_process(adata):
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='cell_ranger', subset=True)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        processed_adata = adata[:, adata.var['highly_variable']]
        return processed_adata

    @staticmethod
    def create_adata(features, batch, celltype):
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
    model.train(max_epochs=1)
    latent = model.get_latent_representation()
    adata_scvi.obsm["X_scvi"] = latent
    return adata_scvi

def run_imap(adata_imap, mode):
    logger.info(adata_imap)
    logger.info(adata_imap.obs.head())
    logger.info(adata_imap.var.head())

    adata_imap = imap.stage1.data_preprocess(adata_imap, 'batch') if mode == 'rna' else adata_imap
    logger.info("after imap.stage1.data_preprocess")
    EC, ec_data = imap.stage1.iMAP_fast(adata_imap, key="batch", n_epochs=10) 
    output_results = imap.stage2.integrate_data(adata_imap, ec_data, inc=False, n_epochs=10)
    output_imap = sc.AnnData(output_results)
    output_imap.obs['celltype'] = adata_imap.obs['celltype'].values
    output_imap.obs['BATCH'] = adata_imap.obs['batch'].values
    return output_imap

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
    
    



