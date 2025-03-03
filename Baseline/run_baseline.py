import numpy as np
import torch
import scanpy as sc
from tqdm import tqdm
import pandas as pd
import os
from run_utils import RunBaseline
from evaluation import evaluate_nn, evaluate_non_nn
from logger_config import logger
from pathlib import Path

class BaselineEvaluator:
    def __init__(self, adata, mode, sampling_seed=42, seed_list=[42, 52, 62]):
        """
        Initialize the BaselineEvaluator class.
        :param adata: Original AnnData object.
        :param mode: Mode for running baseline (e.g., 'rna' or 'imc').
        :param seed_list: List of random seeds for experiments.
        """
        self.adata = adata
        self.seed_list = seed_list
        self.mode = mode
        self.sampling_seed = sampling_seed
        
        logger.info("Starting BaselineEvaluator initialization...")
        logger.info(f"Initialized BaselineEvaluator with mode={mode}, seeds={seed_list}")
        logger.info("Finished BaselineEvaluator initialization.")

    def run_single_nn(self, seed):
        """
        Generate experimental results for different seeds.
        """
        logger.info(f"Running experiment with seed={seed}")

        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rb = RunBaseline(self.adata, mode=self.mode)
        adata_dict = rb.run_nn()  # Run neural network methods
        logger.info(f"Finished running NN methods for seed={seed}")
        return adata_dict

    def run_multiple_nn(self):  
        """
        Evaluate experiments under multiple random seeds.
        Returns the mean and standard deviation of each metric for each method across multiple runs.
        """
        logger.info("Evaluating NN Methods across multiple runs...")
        aggregated_results = {}
        
        for seed in tqdm(self.seed_list, desc="Evaluating NN Methods"):
            adata_dict = self.run_single_nn(seed)
            metrics_run = evaluate_nn(adata_dict, fraction=0.01, seed=42)  # Fixed seed for sampling
            
            for method, metrics in metrics_run.items():
                if method not in aggregated_results:
                    aggregated_results[method] = {metric: [] for metric in metrics}
                for metric, value in metrics.items():
                    aggregated_results[method][metric].append(value)

        final_results = {}
        for method, metric_dict in aggregated_results.items():
            final_results[method] = {
                metric: {'mean': np.mean(values), 'std': np.std(values)}
                for metric, values in metric_dict.items()
            }

        return final_results

    def run_non_nn(self):
        """
        Evaluate non-NN methods once.
        """
        logger.info("Evaluating Non-NN Methods...")
        rb = RunBaseline(self.adata, mode=self.mode)
        logger.info("Run Baseline method finished")
        adata_dict = rb.run_non_nn()  # Run non-NN methods
        metrics_run = evaluate_non_nn(adata_dict, seed=42)  # Fixed seed for sampling
        
        logger.info(f"Non-NN Methods Evaluation: {metrics_run}")
        return metrics_run

def main(adata_dir, save_dir):
    logger.info(f"Loading AnnData from {adata_dir}")
    adata = sc.read_h5ad(adata_dir)
    logger.info("AnnData loaded successfully.")
    evaluator = BaselineEvaluator(adata, mode='rna', sampling_seed=42, seed_list=[42])

    # evaluate NN methods
    logger.info("Starting evaluation of NN methods...")
    final_evaluation_nn = evaluator.run_multiple_nn()    
    pd.DataFrame(final_evaluation_nn).to_csv(save_dir / 'results_nn.csv', index=True)
    logger.info(f"NN Methods Evaluation Results saved to {save_dir}/results_nn.csv")

    # evaluate non-NN methods
    # logger.info("Starting evaluation of non-NN methods...")
    # final_evaluation_non_nn = evaluator.run_non_nn()
    # logger.info(final_evaluation_non_nn)
    # save_path = Path(save_dir) / 'results_non_nn.csv'
    # pd.DataFrame(final_evaluation_non_nn).to_csv(save_path, index=True)
    # logger.info(f"Non-NN Methods Evaluation Results saved to {save_dir}/results_non_nn.csv")

if __name__ == "__main__":
    logger.info("Script execution started.")
    data_name = "macaque"
    adata_dir = f"../Data/scRNA-seq/{data_name}.h5ad"
    save_dir = f"../Results/scRNA-seq/{data_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger.info(f"Results directory created at {save_dir}")
    main(adata_dir, save_dir)
    logger.info("Script execution finished.")
