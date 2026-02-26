import os
import scanpy as sc
from tqdm import tqdm
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


def load_adata_from_dir(save_dir):
    """Load all h5ad files from directory."""
    adata_dict = {}
    for f in Path(save_dir).glob("*.h5ad"):
        method = f.stem
        adata_dict[method] = sc.read_h5ad(f)
        logger.info(f"Loaded {method}")
    return adata_dict


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
    for file_name, key in tqdm(file_key_map.items(), desc="loading files", unit='file'):
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
    for key, adata in tqdm(adata_dict.items(), desc="sampling adata", unit="adata"):
        adata = sc.pp.subsample(adata, fraction=fraction).copy()
        sampling_adata_dict[key] = adata.copy()
    return sampling_adata_dict


def save_one_adata(adata, method_name, save_dir):
    """Save single adata result to h5ad file."""
    adata_dir = os.path.join(save_dir, 'adata_results')
    os.makedirs(adata_dir, exist_ok=True)
    save_path = os.path.join(adata_dir, f'{method_name}.h5ad')
    adata.write(save_path)
    logger.info(f"Saved {method_name} to {save_path}")
    return save_path


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


def append_method_result(save_dir, method_name, method_metrics, timing_row=None):
    """Append one method's evaluation result (and optional timing) to CSV. Safer: one method at a time."""
    metrics = sorted(method_metrics.keys())
    row = {'method': method_name}
    for m in metrics:
        row[f'{m}_mean'] = method_metrics[m]['mean']
        row[f'{m}_std'] = method_metrics[m]['std']
    results_path = os.path.join(save_dir, 'results.csv')
    df = pd.DataFrame([row])
    if os.path.exists(results_path):
        df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        df.to_csv(results_path, index=False)
    logger.info(f"Results appended for {method_name} -> {results_path}")
    if timing_row is not None:
        timing_path = os.path.join(save_dir, 'timing_results.csv')
        tdf = pd.DataFrame([{'method': method_name, 'mean': timing_row['mean'], 'std': timing_row['std']}])
        if os.path.exists(timing_path):
            tdf.to_csv(timing_path, mode='a', header=False, index=False)
        else:
            tdf.to_csv(timing_path, index=False)


def get_save_dir(dataset_config, dataset_name):
    """Get save directory based on dataset mode."""
    if dataset_config.mode == 'imc':
        return f"../Results/IMC/{dataset_name}"
    return f"../Results/scRNA-seq/{dataset_name}"

