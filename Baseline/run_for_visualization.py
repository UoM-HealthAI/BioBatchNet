import numpy as np
import torch
import scanpy as sc
import pandas as pd
import os
from run_utils import RunBaseline
from logger_config import logger
from visulization import plot_batch_and_celltype, cal_nn

class VisualizationRunner:
    def __init__(self, adata, mode, seed=42):
        """
        Initialize the VisualizationRunner
        :param adata: Original AnnData object
        :param mode: Running mode ('rna' or 'imc')
        :param seed: Random seed for reproducibility
        """
        self.adata = adata
        self.mode = mode
        self.seed = seed
        logger.info(f"Initialized VisualizationRunner with mode={mode}, seed={seed}")

    def run_single_for_visualization(self):
        """
        Run all methods once for visualization and save adata results
        """
        logger.info("Running all methods once for visualization...")
        
        # Set fixed seed for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Run NN methods (with single seed)
        rb_nn = RunBaseline(self.adata, mode=self.mode, seed=self.seed)
        nn_results = rb_nn.train_nn()
        logger.info("NN methods completed")
        
        # Run non-NN methods
        rb_non_nn = RunBaseline(self.adata, mode=self.mode)
        non_nn_results = rb_non_nn.train_non_nn()
        logger.info("Non-NN methods completed")
        
        # Merge all results
        all_results = {**nn_results, **non_nn_results}
        logger.info(f"All baseline methods completed. Methods: {list(all_results.keys())}")
        
        return all_results

    def save_all_adata(self, adata_dict, save_dir):
        """Save all method adata results"""
        adata_dir = os.path.join(save_dir, 'adata_results')
        os.makedirs(adata_dir, exist_ok=True)
        
        for method, adata in adata_dict.items():
            save_path = os.path.join(adata_dir, f'{method}.h5ad')
            adata.write(save_path)
            logger.info(f"Saved {method} to {save_path}")
        
        logger.info(f"All adata saved to {adata_dir}")

    def load_baseline_adata(self, baseline_dir):
        """Load saved baseline adata results"""
        adata_dict = {}
        adata_dir = os.path.join(baseline_dir, 'adata_results')
        
        if not os.path.exists(adata_dir):
            raise FileNotFoundError(f"Baseline adata directory not found: {adata_dir}")
        
        for method_file in os.listdir(adata_dir):
            if method_file.endswith('.h5ad'):
                method_name = method_file.replace('.h5ad', '')
                file_path = os.path.join(adata_dir, method_file)
                adata_dict[method_name] = sc.read_h5ad(file_path)
                logger.info(f"Loaded {method_name} from {file_path}")
        
        return adata_dict

    def integrate_other_methods(self, baseline_dict, other_methods_paths):
        """
        Integrate other method results
        :param baseline_dict: Baseline methods adata dictionary
        :param other_methods_paths: Other methods path dictionary {'method_name': 'path'}
        """
        all_methods = baseline_dict.copy()
        
        for method, path in other_methods_paths.items():
            if os.path.exists(path):
                all_methods[method] = sc.read_h5ad(path)
                logger.info(f"Integrated {method} from {path}")
            else:
                logger.warning(f"Path not found for {method}: {path}")
        
        logger.info(f"Total methods for visualization: {list(all_methods.keys())}")
        return all_methods

    def run_and_visualize(self, save_dir, other_methods_paths=None, run_baseline=True):
        """
        Complete running and visualization pipeline
        :param save_dir: Save directory
        :param other_methods_paths: Other methods path dictionary
        :param run_baseline: Whether to run baseline (False to load existing results)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if run_baseline:
            # Run baseline methods
            baseline_results = self.run_single_for_visualization()
            # Save adata results
            self.save_all_adata(baseline_results, save_dir)
        else:
            # Load existing baseline results
            baseline_results = self.load_baseline_adata(save_dir)
        
        # Integrate other methods
        if other_methods_paths:
            all_results = self.integrate_other_methods(baseline_results, other_methods_paths)
        else:
            all_results = baseline_results
        
        # Calculate neighbors for visualization
        logger.info("Calculating neighbors for UMAP visualization...")
        cal_nn(all_results)
        
        # Unified visualization
        batch_save_path = os.path.join(save_dir, 'batch_all_methods.png')
        celltype_save_path = os.path.join(save_dir, 'celltype_all_methods.png')
        
        plot_batch_and_celltype(all_results, batch_save_path, celltype_save_path)
        
        logger.info(f"Visualization completed. Plots saved to {save_dir}")
        return all_results


def main_visualization(adata_path, save_dir, mode, other_methods_paths=None, run_baseline=True):
    """
    Main visualization function
    :param adata_path: Original data path
    :param save_dir: Save directory
    :param mode: Running mode
    :param other_methods_paths: Other methods path dictionary
    :param run_baseline: Whether to run baseline
    """
    logger.info(f"Starting visualization pipeline for {adata_path}")
    
    # Load data
    adata = sc.read_h5ad(adata_path)
    logger.info(f"Loaded data: {adata.shape}")
    
    # Create visualization runner
    runner = VisualizationRunner(adata, mode=mode)
    
    # Run complete pipeline
    results = runner.run_and_visualize(save_dir, other_methods_paths, run_baseline)
    
    logger.info("Visualization pipeline completed")
    return results


def batch_visualization(dataset_configs, base_data_dir="../Data", base_save_dir="../Results/Visualization", base_other_methods_dir="../Results"):
    """
    Batch visualization for multiple datasets
    :param dataset_configs: Dictionary containing dataset configurations
    :param base_data_dir: Base directory for data files
    :param base_save_dir: Base directory for saving results
    :param base_other_methods_dir: Base directory for other methods results
    """
    logger.info(f"Starting batch visualization for {len(dataset_configs)} datasets")
    
    all_results = {}
    
    for dataset_name, config in dataset_configs.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'='*50}")
        
        try:
            # Build paths
            mode = config["mode"]
            if mode == 'imc':
                adata_path = os.path.join(base_data_dir, "IMC", f"{dataset_name}.h5ad")
            else:  # mode == 'rna'
                adata_path = os.path.join(base_data_dir, "scRNA-seq", f"{dataset_name}.h5ad")
            
            save_dir = os.path.join(base_save_dir, dataset_name)
            
            # Build other methods paths if specified
            other_methods_paths = None
            if config.get("other_methods"):
                other_methods_paths = {}
                for method, filename in config["other_methods"].items():
                    method_path = os.path.join(base_other_methods_dir, method, filename)
                    other_methods_paths[method] = method_path
            
            # Run visualization for this dataset
            results = main_visualization(
                adata_path=adata_path,
                save_dir=save_dir,
                mode=mode,
                other_methods_paths=other_methods_paths,
                run_baseline=config.get("run_baseline", True)
            )
            
            all_results[dataset_name] = results
            logger.info(f"✅ Successfully processed {dataset_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {dataset_name}: {str(e)}")
            continue
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Batch visualization completed")
    logger.info(f"Successfully processed: {list(all_results.keys())}")
    logger.info(f"{'='*50}")
    
    return all_results


def get_default_dataset_configs():
    """
    Get default dataset configurations for all datasets
    Note: Only scRNA-seq datasets are configured since IMC data is not found in Data directory.
    To add IMC datasets, put IMC h5ad files in Data/IMC/h5ad_format/ and uncomment the IMC configs below.
    """
    return {
        # IMC datasets (uncomment and adjust paths if IMC data is available)
        # 'subDamond_full': {
        #     "mode": "imc",
        #     "run_baseline": True,
        #     "other_methods": {
        #         'BioBatchNet': "subDamond_full_biobatchnet.h5ad",
        #         'scDREAMER': "subDamond_full_scdreamer.h5ad",
        #         'scDML': "subDamond_full_scdml.h5ad"
        #     }
        # },
        
        # scRNA-seq datasets (available in Data/scRNA-seq/)
        'pancreas': {
            "mode": "rna",
            "run_baseline": True,
            "other_methods": {
                'BioBatchNet': "pancreas_biobatchnet.h5ad",
                'scDREAMER': "pancreas_scdreamer.h5ad",
                'scDML': "pancreas_scdml.h5ad"
            }
        },
        'macaque': {
            "mode": "rna",
            "run_baseline": True,
            "other_methods": {
                'BioBatchNet': "macaque_biobatchnet.h5ad",
                'scDREAMER': "macaque_scdreamer.h5ad",
                'scDML': "macaque_scdml.h5ad"
            }
        },

        'mousebrain': {
            "mode": "rna",
            "run_baseline": True,
            "other_methods": {
                'BioBatchNet': "mousebrain_biobatchnet.h5ad",
                'scDREAMER': "mousebrain_scdreamer.h5ad",
                'scDML': "mousebrain_scdml.h5ad"
            }
        }
    }


if __name__ == "__main__":
    # Single dataset example
    single_config = {
        'adata_path': "../Data/scRNA-seq/pancreas.h5ad",
        'save_dir': "../Results/Visualization/pancreas",
        'mode': "rna",
        'other_methods_paths': {
            'BioBatchNet': "../Results/BioBatchNet/pancreas_biobatchnet.h5ad",
            'scDREAMER': "../Results/scDREAMER/pancreas_scdreamer.h5ad",
            'scDML': "../Results/scDML/pancreas_scdml.h5ad"
        },
        'run_baseline': True
    }
    
    # Uncomment to run single dataset
    # main_visualization(**single_config)
    
    # Batch processing for all datasets
    dataset_configs = get_default_dataset_configs()
    
    # You can customize which datasets to process
    # For example, only IMC datasets:
    # imc_configs = {k: v for k, v in dataset_configs.items() if v["mode"] == "imc"}
    # batch_visualization(imc_configs)
    
    # Or only specific datasets:
    # selected_configs = {k: dataset_configs[k] for k in ['subDamond_full', 'pancreas']}
    # batch_visualization(selected_configs)
    
    # Process all datasets
    batch_visualization(dataset_configs) 