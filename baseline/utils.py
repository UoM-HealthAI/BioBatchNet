import os
import scanpy as sc
from tqdm import tqdm
import logging
from datetime import datetime


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "../Logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = os.path.join(log_dir, f"baseline_evaluation_{timestamp}.log")

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


def save_adata_dict(adata_dict, save_dir, dataset_name):
    """Save adata dict to directory with dataset name subfolder."""
    dataset_save_dir = os.path.join(save_dir, dataset_name)
    if not os.path.exists(dataset_save_dir):
        os.makedirs(dataset_save_dir)

    for key, adata in adata_dict.items():
        save_path = os.path.join(dataset_save_dir, f"{key}.h5ad")
        adata.write_h5ad(save_path)
        print(f"Saved {key} to {save_path}")


def load_h5ad_files(directory):
    """Load h5ad files from directory with predefined file-key mapping."""
    file_key_map = {
        'Raw.h5ad': 'Raw',
        'scVI.h5ad': 'scVI',
        'Harmony.h5ad': 'Harmony',
        'BBKNN.h5ad': 'BBKNN',
        'Scanorama.h5ad': 'Scanorama',
        'Combat.h5ad': 'Combat',
        'iMAP.h5ad': 'iMAP',
        'CombatSeq.h5ad': 'CombatSeq',
        'SeuratRPCA.h5ad': 'Seurat',
        'FastMNN.h5ad': 'FastMNN',
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
    """Subsample adata dict with given fraction."""
    sampling_adata_dict = {}
    for key, adata in tqdm(adata_dict.items(), desc="samlping adata", unit="adata"):
        adata = sc.pp.subsample(adata, fraction=fraction).copy()
        sampling_adata_dict[key] = adata.copy()
    return sampling_adata_dict


def save_all_adata(adata_dict, save_dir):
    """Save all adata results to h5ad files."""
    adata_dir = os.path.join(save_dir, 'adata_results')
    os.makedirs(adata_dir, exist_ok=True)

    for method, adata in adata_dict.items():
        save_path = os.path.join(adata_dir, f'{method}.h5ad')
        adata.write(save_path)
        logger.info(f"Saved {method} to {save_path}")

    return adata_dir


def load_all_adata(save_dir):
    """Load all adata results from h5ad files."""
    adata_dir = os.path.join(save_dir, 'adata_results')

    if not os.path.exists(adata_dir):
        raise FileNotFoundError(f"Adata directory not found: {adata_dir}")

    adata_dict = {}
    for filename in os.listdir(adata_dir):
        if filename.endswith('.h5ad'):
            method = filename.replace('.h5ad', '')
            adata_dict[method] = sc.read_h5ad(os.path.join(adata_dir, filename))
            logger.info(f"Loaded {method}")

    return adata_dict


def get_save_dir(dataset_config, dataset_name):
    """Get save directory based on dataset mode."""
    if dataset_config.mode == 'imc':
        return f"../Results/IMC/{dataset_name}"
    return f"../Results/scRNA-seq/{dataset_name}"

