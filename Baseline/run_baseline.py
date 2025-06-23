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
    def __init__(self, adata, mode, sampling_fraction, sampling_seed, seed_list):
        """
        Initialize the BaselineEvaluator class.
        :param adata: Original AnnData object.
        :param mode: Mode for running baseline (e.g., 'rna' or 'imc').
        :param sampling_fraction: Fraction of data to use for evaluation (default: 0.5)
        :param sampling_seed: Seed for sampling
        :param seed_list: List of random seeds for experiments.
        """
        self.adata = adata
        self.seed_list = seed_list
        self.mode = mode
        self.sampling_fraction = sampling_fraction
        self.sampling_seed = sampling_seed
        self.timing_results = {
            'scVI': {'times': [], 'mean': None, 'std': None},
            'iMAP': {'times': [], 'mean': None, 'std': None},
            'MRVI': {'times': [], 'mean': None, 'std': None},
            'Harmony': None,
            'BBKNN': None,
            'Scanorama': None,
            'Combat': None
        }
        
        logger.info("Starting BaselineEvaluator initialization...")
        logger.info(f"Initialized BaselineEvaluator with mode={mode}, seeds={seed_list}, sampling_fraction={sampling_fraction}")
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

        rb = RunBaseline(self.adata, mode=self.mode, seed=seed)
        adata_dict = rb.train_nn()  # Run neural network methods
        
        # Collect timing results
        timing_results = rb.get_timing_results()
        for method in ['scVI', 'iMAP', 'MRVI']:
            self.timing_results[method]['times'].append(timing_results[method])
        
        logger.info(f"Finished running NN methods for seed={seed}")
        return adata_dict

    def run_multiple_nn(self):  
        """
        Evaluate experiments under multiple random seeds.
        Returns the mean and standard deviation of each metric for each method across multiple runs.
        """
        logger.info("Evaluating NN Methods across multiple runs...")
        aggregated_results = {}
        
        for seed in self.seed_list:
            adata_dict = self.run_single_nn(seed)
            metrics_run = evaluate_nn(adata_dict, fraction=self.sampling_fraction, seed=self.sampling_seed)
            
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

        # Calculate mean and std for timing results
        for method in ['scVI', 'iMAP', 'MRVI']:
            times = self.timing_results[method]['times']
            self.timing_results[method]['mean'] = np.mean(times)
            self.timing_results[method]['std'] = np.std(times)
            logger.info(f"{method} timing - mean: {np.mean(times):.2f}s, std: {np.std(times):.2f}s")

        return final_results

    def run_single_non_nn(self):
        """
        Evaluate non-NN methods once.
        """
        logger.info("Evaluating Non-NN Methods...")
        rb = RunBaseline(self.adata, mode=self.mode)
        logger.info("Run Baseline method finished")
        adata_dict = rb.train_non_nn()  # Run non-NN methods
        
        # Collect timing results
        timing_results = rb.get_timing_results()
        for method in ['Harmony', 'BBKNN', 'Scanorama', 'Combat']:
            self.timing_results[method] = timing_results[method]
        
        metrics_run = evaluate_non_nn(adata_dict, fraction=self.sampling_fraction, seed=self.sampling_seed)  # Fixed seed for sampling
        logger.info(f"Non-NN Methods Evaluation: {metrics_run}")
        return metrics_run

    def get_timing_results(self):
        """
        Get all timing results in a formatted dictionary
        """
        formatted_results = {}
        
        # Format NN methods results
        for method in ['scVI', 'iMAP', 'MRVI']:
            formatted_results[method] = {
                'mean_time': self.timing_results[method]['mean'],
                'std_time': self.timing_results[method]['std'],
                'all_times': self.timing_results[method]['times'],
                'type': 'Neural Network'
            }
        
        # Format non-NN methods results
        for method in ['Harmony', 'BBKNN', 'Scanorama', 'Combat']:
            formatted_results[method] = {
                'mean_time': self.timing_results[method],
                'std_time': None,
                'all_times': [self.timing_results[method]],
                'type': 'Non-Neural Network'
            }
        
        return formatted_results

def main(adata_dir, 
         save_dir, 
         mode, 
         sampling_fraction, 
         sampling_seed, 
         seed_list):
    
    logger.info(f"Loading AnnData from {adata_dir}")
    adata = sc.read_h5ad(adata_dir)
    logger.info("AnnData loaded successfully.")

    evaluator = BaselineEvaluator(adata, 
                                mode=mode, 
                                sampling_fraction=sampling_fraction,
                                sampling_seed=sampling_seed, 
                                seed_list=seed_list)

    # evaluate NN methods
    logger.info("Starting evaluation of NN methods...")
    final_evaluation_nn = evaluator.run_multiple_nn()    
    results_nn_path = os.path.join(save_dir, 'results_nn.csv')
    pd.DataFrame(final_evaluation_nn).to_csv(results_nn_path, index=True)
    logger.info(f"NN Methods Evaluation Results saved to {save_dir}/results_nn.csv")

    # evaluate non-NN methods
    logger.info("Starting evaluation of non-NN methods...")
    final_evaluation_non_nn = evaluator.run_single_non_nn()
    logger.info(final_evaluation_non_nn)
    results_non_nn_path = os.path.join(save_dir, 'results_non_nn.csv')
    pd.DataFrame(final_evaluation_non_nn).to_csv(results_non_nn_path, index=True)
    logger.info(f"Non-NN Methods Evaluation Results saved to {save_dir}/results_non_nn.csv")

    # Save timing results in a single CSV
    timing_results = evaluator.get_timing_results()
    timing_df = pd.DataFrame.from_dict(timing_results, orient='index')
    timing_df = timing_df.reset_index()
    timing_df.columns = ['Method', 'Mean Time (s)', 'Std Time (s)', 'All Times (s)', 'Type']
    timing_path = os.path.join(save_dir, 'timing_results.csv')
    timing_df.to_csv(timing_path, index=False)
    logger.info(f"Timing results saved to {timing_path}")

if __name__ == "__main__":
    logger.info("Script execution started.")

    data_config = {
        'subDamond_full': {
            "mode": "imc",
            "sampling_fraction": 0.2,
            "sampling_seed": 42,
            "seed_list": [42, 52, 62, 72, 82]
        },
        'HochSchulz': {
            "mode": "imc",
            "sampling_fraction": 0.1,
            "sampling_seed": 42,
            "seed_list": [42, 52, 62, 72, 82]
        },
        "IMMUcan_batch": {
            "mode": "imc",
            "sampling_fraction": 1,
            "sampling_seed": 42,
            "seed_list": [42, 52, 62, 72, 82]
        },
        'pancreas':{
            "mode": "rna",
            "sampling_fraction": 1,
            "sampling_seed": 42,
            "seed_list": [42, 52, 62, 72, 82]
        },
        'macaque':{
            "mode": "rna",
            "sampling_fraction": 1,
            "sampling_seed": 42,
            "seed_list": [42, 52, 62, 72, 82]
        },
        'SubMouseBrain':{
            "mode": "rna",
            "sampling_fraction": 1,
            "sampling_seed": 42,
            "seed_list": [42, 52, 62, 72, 82]
        }
    }

    for data_name, config in data_config.items():
        mode = config["mode"]
        sampling_fraction = config["sampling_fraction"]
        sampling_seed = config["sampling_seed"]
        seed_list = config["seed_list"]

        if mode == 'imc':
            adata_dir = f"../Data/IMC/h5ad_format/{data_name}.h5ad"
            save_dir = f"../Results/IMC/{data_name}"
        elif mode == 'rna':
            adata_dir = f"../Data/scRNA-seq/{data_name}.h5ad"
            save_dir = f"../Results/scRNA-seq/{data_name}"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logger.info(f"Results directory created at {save_dir}")

        main(adata_dir, 
             save_dir, 
             mode=mode, 
             sampling_fraction=sampling_fraction, 
             sampling_seed=sampling_seed, 
             seed_list=seed_list)
        
        logger.info(f"{data_name}'s task finished.")
