import os
import scanpy as sc
from tqdm import tqdm

def save_adata_dict(adata_dict, save_dir, dataset_name):
    dataset_save_dir = os.path.join(save_dir, dataset_name)
    if not os.path.exists(dataset_save_dir):
        os.makedirs(dataset_save_dir)
    
    for key, adata in adata_dict.items():
        save_path = os.path.join(dataset_save_dir, f"{key}.h5ad")
        adata.write_h5ad(save_path)
        print(f"Saved {key} to {save_path}")

def load_h5ad_files(directory):
    file_key_map = {
        'Raw.h5ad': 'Raw',
        'scVI.h5ad': 'scVI',
        'Harmony.h5ad': 'Harmony',
        'BBKNN.h5ad': 'BBKNN',
        'Scanorama.h5ad': 'Scanorama',
        'Combat.h5ad': 'Combat',
        'iMAP.h5ad': 'iMAP',
        'scDREAMER.h5ad': 'scDREAMER',
        'scDML.h5ad': 'scDML',
        'BioBatchNet.h5ad': 'BioBatchNet',
    }

    result_dict = {}    
    for file_name, key in tqdm(file_key_map.items(), desc="loadding files", unit='file'):
        file_path = os.path.join(directory, file_name)
        print(file_path)
        if os.path.exists(file_path):  
            result_dict[key] = sc.read_h5ad(file_path)
        else:
            print(f"File {file_name} not found in directory {directory}")
    return result_dict

def sampling(adata_dict, fraction=0.3):
    sampling_adata_dict = {}
    for key, adata in tqdm(adata_dict.items(), desc="samlping adata", unit="adata"):
        sc.pp.subsample(adata, fraction=fraction)
        sampling_adata_dict[key] =  adata
    return sampling_adata_dict

