import numpy as np
import torch
import scanpy as sc
import logging
from tqdm import tqdm

from run_utils import RunBaseline
from evaluation import evaluate_NN, evaluate_non_NN

# 配置 logging
logging.basicConfig(
    filename="baseline_evaluation.log",  # 日志文件
    level=logging.INFO,  # 设置日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class BaselineEvaluator:
    def __init__(self, adata, mode, seed_list=[42, 52, 62]):
        """
        Initialize the BaselineEvaluator class.
        :param adata: Original AnnData object.
        :param mode: Mode for running baseline (e.g., 'rna' or 'imc').
        :param seed_list: List of random seeds for experiments.
        """
        self.adata = adata
        self.seed_list = seed_list
        self.mode = mode
        logger.info(f"Initialized BaselineEvaluator with mode={mode}, seeds={seed_list}")

    def adata_dict_generator(self, seed):
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

    def evaluate_multiple_runs(self):
        """
        Evaluate experiments under multiple random seeds.
        Returns the mean and standard deviation of each metric for each method across multiple runs.
        """
        logger.info("Evaluating NN Methods across multiple runs...")
        aggregated_results = {}
        
        for seed in tqdm(self.seed_list, desc="Evaluating NN Methods"):
            adata_dict = self.adata_dict_generator(seed)
            metrics_run = evaluate_NN(adata_dict, seed=42)  # Fixed seed for sampling
            
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

        logger.info(f"Final NN Methods Evaluation: {final_results}")
        return final_results

    def evaluate_non_nn(self):
        """
        Evaluate non-NN methods once.
        """
        logger.info("Evaluating Non-NN Methods...")
        rb = RunBaseline(self.adata, mode=self.mode)
        logger.info("Run Baseline method finished")
        adata_dict = rb.run_non_nn()  # Run non-NN methods
        metrics_run = evaluate_non_NN(adata_dict, seed=42)  # Fixed seed for sampling
        
        logger.info(f"Non-NN Methods Evaluation: {metrics_run}")
        return metrics_run

def main(adata_dir):
    logger.info(f"Loading AnnData from {adata_dir}")
    adata = sc.read_h5ad(adata_dir)
    evaluator = BaselineEvaluator(adata, mode='rna')

    # 评估 NN 方法
    # final_evaluation_nn = evaluator.evaluate_multiple_runs()
    # logger.info(f"NN Methods Evaluation Results: {final_evaluation_nn}")

    # 评估非 NN 方法
    final_evaluation_non_nn = evaluator.evaluate_non_nn()
    logger.info(f"Non-NN Methods Evaluation Results: {final_evaluation_non_nn}")

if __name__ == "__main__":
    adata_dir = "../Data/scRNA-seq/macaque_raw.h5ad"
    main(adata_dir)
