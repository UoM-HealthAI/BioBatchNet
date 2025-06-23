import scanpy as sc
import scib
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np


def subsample_data(adata, fraction, seed):
    np.random.seed(seed)
    adata = adata.copy()
    sc.pp.subsample(adata, fraction=fraction, random_state=seed)
    return adata

def evaluate_non_nn(adata_dict, fraction, seed):
    results = {}
    batch_key = 'BATCH'
    label_key = 'celltype'

    raw_adata = adata_dict.get('Raw')
    sub_raw_adata = subsample_data(raw_adata, fraction=fraction, seed=seed)
    sc.pp.pca(sub_raw_adata)
    sc.pp.neighbors(sub_raw_adata, use_rep='X_pca')

    for key, adata in adata_dict.items():
        if key == 'Harmony':
            embed = 'X_pca_harmony'
            sub_adata = subsample_data(adata, fraction=fraction, seed=seed)
            sc.pp.neighbors(sub_adata, use_rep=embed)
            results[key] = compute_metrics(sub_raw_adata, sub_adata, batch_key, label_key, type='embed', embed=embed)

        elif key == 'BBKNN':
            embed = 'X_pca'
            sub_adata = subsample_data(adata, fraction=fraction, seed=seed)
            sc.pp.neighbors(sub_adata, use_rep=embed)
            results[key] = compute_metrics(sub_raw_adata, sub_adata, batch_key, label_key, type='embed', embed=embed)
        
        elif key == 'Scanorama':
            embed = 'X_scanorama'
            sub_adata = subsample_data(adata, fraction=fraction, seed=seed)
            sc.pp.neighbors(sub_adata, use_rep=embed)
            results[key] = compute_metrics(sub_raw_adata, sub_adata, batch_key, label_key, type='embed', embed=embed)

        elif key == 'Combat':
            embed = 'X_pca'
            sub_adata = subsample_data(adata, fraction=fraction, seed=seed)
            sc.pp.pca(sub_adata)  
            sc.pp.neighbors(sub_adata, use_rep=embed)  
            results[key] = compute_metrics(sub_raw_adata, sub_adata, batch_key, label_key, type='full', embed=embed)

    return results

def evaluate_nn(adata_dict, fraction, seed):
    """
    Evaluate nn method
    """
    results = {}
    batch_key='BATCH'
    label_key='celltype'

    raw_adata = adata_dict.get('Raw')
    sub_raw_adata = subsample_data(raw_adata, fraction=fraction, seed=seed)
    sc.pp.pca(sub_raw_adata)
    sc.pp.neighbors(sub_raw_adata, use_rep='X_pca')

    for key, adata in adata_dict.items():
        if key == 'scVI':
            embed = 'X_scvi'
            sub_adata = subsample_data(adata, fraction=fraction, seed=seed)
            sc.pp.neighbors(sub_adata, use_rep=embed)  
            results[key] = compute_metrics(sub_raw_adata, sub_adata, batch_key, label_key, type='embed', embed=embed)

        elif key == 'iMAP':
            embed = 'X_pca'
            sub_adata = subsample_data(adata, fraction=fraction, seed=seed)
            sc.pp.pca(sub_adata)
            sc.pp.neighbors(sub_adata, use_rep=embed)  
            results[key] = compute_metrics(sub_raw_adata, sub_adata, batch_key, label_key, type='full', embed=embed)
        
        elif key == 'MRVI':
            embed = 'X_mrvi'
            sub_adata = subsample_data(adata, fraction=fraction, seed=seed)
            sc.pp.neighbors(sub_adata, use_rep=embed)  
            results[key] = compute_metrics(sub_raw_adata, sub_adata, batch_key, label_key, type='embed', embed=embed)
        
        elif key == 'BioBatchNet':
            embed = 'X_biobatchnet'
            sub_adata = subsample_data(adata, fraction=fraction, seed=seed)
            sc.pp.neighbors(sub_adata, use_rep=embed)  
            results[key] = compute_metrics(sub_raw_adata, sub_adata, batch_key, label_key, type='embed', embed=embed)
    
    return results

def compute_metrics(adata_raw, adata, batch_key, label_key, type, embed):
    # Calculate batch effect metrics
    if type == 'full':
        ilisi = scib.metrics.ilisi_graph(adata, batch_key=batch_key, type_=type)
        pcr = scib.me.pcr_comparison(adata_raw, adata, covariate=batch_key)
    else:
        ilisi = scib.metrics.ilisi_graph(adata, batch_key=batch_key, type_=type, use_rep=embed)
        pcr = scib.me.pcr_comparison(adata_raw, adata, covariate=batch_key, embed=embed)

    graph_connectivity = scib.me.graph_connectivity(adata, label_key=label_key)
    asw_batch = scib.me.silhouette_batch(adata, batch_key=batch_key, label_key=label_key, embed=embed)
    
    # Calculate biological conservation metric
    asw_cell = scib.me.silhouette(adata, label_key=label_key, embed=embed)

    # Clustering evaluation
    scib.me.cluster_optimal_resolution(adata, cluster_key="cluster", label_key="celltype")
    ari = scib.me.ari(adata, cluster_key="cluster", label_key="celltype")
    nmi = scib.me.nmi(adata, cluster_key="cluster", label_key="celltype")

    batch_score = (ilisi + graph_connectivity + asw_batch + pcr) / 4
    bio_score = (asw_cell + ari + nmi) / 3
    total_score = (batch_score + bio_score) / 2

    return {
        'iLISI': ilisi,
        'GraphConn': graph_connectivity,
        'ASW_batch': asw_batch,
        'PCR': pcr,
        'BatchScore': batch_score,
        'ASW': asw_cell,
        'ARI': ari,
        'NMI': nmi,
        'BioScore': bio_score,
        'TotalScore': total_score
    }



