import numpy as np
import torch
import scanpy as sc
from tqdm import tqdm

from run_utils import RunBaseline
from evaluation import evaluate_NN, evaluate_non_NN

class BaselineEvaluator:
    def __init__(self, adata, mode, seed_list=[42, 52, 62]):
        """
        Initialize the BaselineEvaluator class.
        :param adata: Original AnnData object.
        :param mode: Mode for running baseline (e.g.,]] 'rna' or 'imc').
        :param seed_list: List of random seeds for experiments.
        """
        self.adata = adata
        self.seed_list = seed_list
        self.mode = mode

    def adata_dict_generator(self, seed):
        """
        Generate experimental results for different seeds.
        For example, you can call the run_nn method of RunBaseline here.
        """
        # Set the seed for model initialization
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rb = RunBaseline(self.adata, mode=self.mode)
        adata_dict = rb.run_nn()  # Run neural network methods
        return adata_dict

    def evaluate_multiple_runs(self):
        """
        Evaluate experiments under multiple random seeds.
        Returns the mean and standard deviation of each metric for each method across multiple runs.
        """
        aggregated_results = {}
        
        for seed in tqdm(self.seed_list, desc="Evaluating NN Methods"):
            adata_dict = self.adata_dict_generator(seed)
            # Use a fixed seed for data sampling in evaluate_NN
            metrics_run = evaluate_NN(adata_dict, seed=42)  # Fixed seed for sampling
            
            for method, metrics in metrics_run.items():
                if method not in aggregated_results:
                    aggregated_results[method] = {metric: [] for metric in metrics}
                for metric, value in metrics.items():
                    aggregated_results[method][metric].append(value)
        
        final_results = {}
        for method, metric_dict in aggregated_results.items():
            final_results[method] = {metric: {'mean': np.mean(values), 'std': np.std(values)}
                                     for metric, values in metric_dict.items()}
        
        return final_results

    def evaluate_non_nn(self):
        """
        Evaluate non-NN methods once.
        Returns the metrics for each method.
        """
        print("Evaluating Non-NN Methods...")
        rb = RunBaseline(self.adata, mode=self.mode)
        adata_dict = rb.run_non_nn()  # Run non-NN methods
        # Use a fixed seed for data sampling in evaluate_non_NN
        metrics_run = evaluate_non_NN(adata_dict, seed=42)  # Fixed seed for sampling
        return metrics_run

def main(adata_dir):
    adata = sc.read_h5ad(adata_dir)
    evaluator = BaselineEvaluator(adata, mode='rna')
    final_evaluation_nn = evaluator.evaluate_multiple_runs()
    print("NN Methods Evaluation:", final_evaluation_nn)
    
    final_evaluation_non_nn = evaluator.evaluate_non_nn()
    print("Non-NN Methods Evaluation:", final_evaluation_non_nn)

if __name__ == "__main__":
    adata_dir = "../Data/scRNA-seq"
    main(adata_dir)

