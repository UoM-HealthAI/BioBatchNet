# -- author: haiping liu
# -- date: 2025.1.2

import scanpy as sc
import os
import pandas as pd
from evaluation import evaluate  
from run_utils import run_methods
from visulization import plot_umap_batch_and_celltype
from utils import save_adata_dict

def main():
    dataset_dir = '/root/autodl-tmp/Gene_data'
    datasets = {
        'human_immune': os.path.join(dataset_dir, 'Immune_ALL_human.h5ad'),
        'mouse_brain': os.path.join(dataset_dir, 'SubMouseBrain_raw.h5ad'),
        'macaque': os.path.join(dataset_dir, 'macaque_raw.h5ad'),
        'pancreas': os.path.join(dataset_dir, 'pancreas_raw.h5ad')
    }

    berd_dir = '/root/autodl-tmp/berd_embed'
    berd_embedding = {
        'human_immune': os.path.join(berd_dir, 'human_berd.csv'),
        'mouse_brain': os.path.join(berd_dir, 'sub_mousebrain_berd.csv'),
        'macaque': os.path.join(berd_dir, 'macauqe_berd.csv'),
        'pancreas': os.path.join(berd_dir, 'pancreas_berd.csv ')
    }

    scdreamer_dir = '/root/autodl-tmp/scdreamer_embed'
    scdreamer_embedding = {
        'human_immune': os.path.join(scdreamer_dir, 'human_scdreamer.csv'),
        'mouse_brain': os.path.join(scdreamer_dir, 'sub_mousebrain_scdreamer.csv'),
        'macaque': os.path.join(scdreamer_dir, 'macaque_scdreamer.csv'),
        'pancreas': os.path.join(scdreamer_dir, 'pancreas_scdreamer.csv ')
    }

    scdml_dir = '/root/autodl-tmp/scDML'
    scdml_embedding = {
        'human_immune': os.path.join(scdml_dir, 'scdml_human.h5ad'),
        'mouse_brain': os.path.join(scdml_dir, 'scdml_submouse.h5ad'),
        'macaque': os.path.join(scdml_dir, 'scdml_macaque.h5ad'),
        'pancreas': os.path.join(scdml_dir, 'scdml_pancreas.h5ad')
    }

    
    results_dir = os.path.join(dataset_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    for dataset_name, dataset_path in datasets.items():
        dataset_results_dir = os.path.join(results_dir, dataset_name)
        if not os.path.exists(dataset_results_dir):
            os.makedirs(dataset_results_dir)

        adata_raw = sc.read_h5ad(dataset_path)

        # berd scdreamer scdml
        berd_latent = pd.read_csv(berd_embedding[dataset_name])
        scdreamer_latent = pd.read_csv(scdreamer_embedding[dataset_name], header=None)
        scdml_latent = sc.read_h5ad(scdml_embedding[dataset_name])

        # run methods
        adata_dict = run_methods(dataset_path, berd_latent, scdreamer_latent, scdml_latent)
        adata_dict['raw_data'] = adata_raw
        save_adata_dict(adata_dict, dataset_results_dir, dataset_name)
        print("Baseline running finished")
    
        # Evaluate the baseline methods
        results = evaluate(adata_dict)
        results_df = pd.DataFrame(results).transpose()
        result_file = os.path.join(dataset_results_dir, f'{dataset_name}_evaluation_results.csv')
        results_df.to_csv(result_file)
        print(f'Evaluation results saved to: {result_file}')
        
        # visualization 
        save_dir_batch = os.path.join(dataset_results_dir, 'umap_batch.png')
        save_dir_celltype = os.path.join(dataset_results_dir, 'umap_cell.png')
        plot_umap_batch_and_celltype(adata_dict, save_dir_batch, save_dir_celltype)
        print(f"UMAP plots saved for {dataset_name}")

if __name__ == '__main__':
    main()
