import scanpy as sc
import scib
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np


def subsample_data(adata, fraction=0.3, seed=42):
    np.random.seed(seed)
    sc.pp.subsample(adata, fraction=fraction, random_state=seed)
    return adata

def evaluate_non_NN(adata_dict, seed):
    results = {}
    batch_key = 'BATCH'  
    label_key = 'celltype'  

    raw_adata = adata_dict.get('Raw')
    sub_raw_adata = subsample_data(raw_adata, seed=seed)

    for key, adata in adata_dict.items():
        if key == 'Raw':
            sc.pp.subsample(adata, fraction=0.3)
            sc.pp.pca(adata)
            sc.pp.neighbors(adata, use_rep='X_pca')
            continue

        elif key == 'Harmony':
            embed = 'X_pca_harmony'
            sub_adata = subsample_data(adata, seed=seed)
            sc.pp.neighbors(sub_adata, use_rep=embed)
            results[key] = compute_metrics(sub_raw_adata, sub_adata, batch_key, label_key, type='embed', embed=embed)

        elif key == 'BBKNN':
            embed = 'X_pca'
            sub_adata = subsample_data(adata, seed=seed)
            sc.pp.neighbors(sub_adata, use_rep=embed)
            results[key] = compute_metrics(sub_raw_adata, sub_adata, batch_key, label_key, type='embed', embed=embed)
        
        elif key == 'Scanorama':
            embed = 'X_scanorama'
            sub_adata = subsample_data(adata, seed=seed)
            sc.pp.neighbors(sub_adata, use_rep=embed)
            results[key] = compute_metrics(sub_raw_adata, sub_adata, batch_key, label_key, type='embed', embed=embed)

        elif key == 'Combat':
            embed = 'X_pca'
            sub_adata = subsample_data(adata, seed=seed)
            sc.pp.pca(sub_adata)  
            sc.pp.neighbors(adata, use_rep=embed)  
            results[key] = compute_metrics(sub_raw_adata, sub_adata, batch_key, label_key, type='full', embed=embed)

    return results

def evaluate_NN(adata_dict, seed=42):
    """
    Evaluate NN method
    """
    results = {}
    batch_key='BATCH'
    label_key='celltype'

    raw_adata = adata_dict.get('Raw')
    sub_raw_adata = subsample_data(raw_adata, seed=seed)

    for key, adata in adata_dict.items():
        if key == 'scVI':
            embed = 'X_scvi'
            sub_adata = subsample_data(adata, seed=seed)
            sc.pp.neighbors(sub_adata, use_rep=embed)  
            results[key] = compute_metrics(sub_raw_adata, sub_adata, batch_key, label_key, type='embed', embed=embed)

        elif key == 'iMAP':
            embed = 'X_pca'
            sub_adata = subsample_data(adata, seed=seed)
            sc.pp.neighbors(sub_adata, use_rep=embed)  
            results[key] = compute_metrics(sub_raw_adata, sub_adata, batch_key, label_key, type='embed', embed=embed)
    
    return results

def compute_metrics(adata_raw, adata, batch_key, label_key, type, embed):
    # batch effect
    if type == 'full':
        ilisi = scib.metrics.ilisi_graph(adata, batch_key=batch_key, type_=type)
        pcr = scib.me.pcr_comparison(adata_raw, adata, covariate=batch_key)
    else:
        ilisi = scib.metrics.ilisi_graph(adata, batch_key=batch_key, type_=type, use_rep=embed)
        pcr = scib.me.pcr_comparison(adata_raw, adata, covariate=batch_key, embed=embed)

    # kbet = scib.me.kBET(adata, batch_key=batch_key, label_key=label_key, type_=type, embed=embed)
    graph_connectivity = scib.me.graph_connectivity(adata, label_key=label_key)
    asw_batch = scib.me.silhouette_batch(adata, batch_key=batch_key, label_key=label_key, embed=embed)
    
    # biological conservation metric
    asw_cell = scib.me.silhouette(adata, label_key=label_key, embed=embed)

    # Clustering evaluation
    scib.me.cluster_optimal_resolution(adata, cluster_key="cluster", label_key="celltype")
    ari = scib.me.ari(adata, cluster_key="cluster", label_key="celltype")
    nmi = scib.me.nmi(adata, cluster_key="cluster", label_key="celltype")

    return {
        'ilisi': ilisi,
        # 'kBET': kbet,
        'graph_connectivity': graph_connectivity,
        'asw_batch': asw_batch,
        'pcr': pcr,
        'ASW': asw_cell,
        'ARI': ari,
        'NMI': nmi
    }



