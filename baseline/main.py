import scanpy as sc
import pandas as pd
import os
from pathlib import Path

from config import BaselineConfig
from engines import RunBaseline
from utils import save_all_adata, load_all_adata, get_save_dir, logger


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

    # Run and evaluate
    rb = RunBaseline(adata, config, dataset_config)
    results, timing_stats, all_adata = rb.run_all_seeds_and_evaluate()

    # Save evaluation results
    results_path = os.path.join(save_dir, 'results.csv')
    pd.DataFrame(results).to_csv(results_path, index=True)
    logger.info(f"Results saved to {results_path}")

    # Save timing results
    timing_df = pd.DataFrame.from_dict(timing_stats, orient='index')
    timing_path = os.path.join(save_dir, 'timing_results.csv')
    timing_df.to_csv(timing_path)
    logger.info(f"Timing saved to {timing_path}")

    # Save adata for visualization
    save_all_adata(all_adata, save_dir)

    return results, all_adata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run baseline methods')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset to process (default: all)')
    args = parser.parse_args()

    logger.info("Script started.")

    config_path = Path(__file__).parent / "baseline_config.yaml"
    config = BaselineConfig.load(config_path)

    datasets = [args.dataset] if args.dataset else list(config.datasets.keys())

    for dataset_name in datasets:
        logger.info(f"Processing: {dataset_name}")
        main(config, dataset_name)
        logger.info(f"{dataset_name} finished.")
