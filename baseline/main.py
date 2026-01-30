# NumPy 兼容性补丁 (np.bool8 在 NumPy 1.24+ 中已移除)
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import scanpy as sc
import pandas as pd
import os
from pathlib import Path

from config import BaselineConfig
from engines import RunBaseline
from utils import load_all_adata, get_save_dir, save_one_adata, logger


def main(config, dataset_name):
    """
    Run baseline evaluation for a dataset.

    Args:
        config: BaselineConfig object
        dataset_name: Name of the dataset
    """
    dataset_config = config.get_dataset(dataset_name)

    logger.info(f"Loading data from {dataset_config.path}")
    adata = sc.read_h5ad(dataset_config.path)
    logger.info(f"Data shape: {adata.shape}")

    save_dir = get_save_dir(dataset_config, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, 'results.csv')
    if os.path.exists(results_path):
        os.remove(results_path)

    # Save Raw first
    save_one_adata(adata, "Raw", save_dir)

    # Run and evaluate (save_dir enables incremental saving; results/timing written per method in engines)
    rb = RunBaseline(adata, config, dataset_config, save_dir=save_dir)
    results, timing_stats, all_adata = rb.run_all_seeds_and_evaluate()

    if save_dir:
        logger.info(f"Results saved to {save_dir}/results.csv")

    return results, all_adata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run baseline methods')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                       help='Specific dataset to process (default: all)')
    args = parser.parse_args()

    logger.info("Script started.")

    config_path = Path(__file__).parent / "config.yaml"
    config = BaselineConfig.load(config_path)

    datasets = [args.dataset] if args.dataset else list(config.datasets.keys())

    for dataset_name in datasets:
        logger.info(f"Processing: {dataset_name}")
        main(config, dataset_name)
        logger.info(f"{dataset_name} finished.")
