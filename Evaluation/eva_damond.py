import pandas as pd
import scanpy as sc
from run_utils import run_methods
from evaluation import evaluate
import pandas as pd
import os
from visulization import plot_batch_and_celltype
from utils import save_adata_dict, load_h5ad_files, sampling

dataset_dir = '/mnt/iusers01/fatpou01/compsci01/w29632hl/scratch/code/Data/IMC_data/adatas'
datasets = {
    'immu': os.path.join(dataset_dir, 'immu_adata.h5ad'),
    'hoch': os.path.join(dataset_dir, 'hoch_adata.h5ad'),
    'damond': os.path.join(dataset_dir, 'damond_adata.h5ad'),
}

berd_dir = '/mnt/iusers01/fatpou01/compsci01/w29632hl/scratch/code/Data/IMC_data/embed'
berd_embedding = {
    'immu': os.path.join(berd_dir, 'immu_berd_embed.csv'),
    'hoch': os.path.join(berd_dir, 'hoch_berd_embed.csv'),
    'damond': os.path.join(berd_dir, 'damond_berd_embed.csv'),
}

scdreamer_dir = '/mnt/iusers01/fatpou01/compsci01/w29632hl/scratch/code/Data/IMC_data/scDreamer_IMC'
scdreamer_embedding = {
    'immu': os.path.join(scdreamer_dir, 'immu_scdreamer.csv'),
    'hoch': os.path.join(scdreamer_dir, 'hoch_scdreamer.csv'),
    'damond': os.path.join(scdreamer_dir, 'damond_scdreamer.csv'),
}

def main(dataset_name):
    berd_latent = pd.read_csv(berd_embedding[dataset_name])
    scdreamer_latent = pd.read_csv(scdreamer_embedding[dataset_name], header=None)

    adata_dir = datasets[dataset_name]
    
    print("begin to run methods")
    adata_dict = run_methods(adata_dir, berd_latent, scdreamer_latent)
    adata_dict = sampling(adata_dict)

    print("begin to evaluate")
    results = evaluate(adata_dict)
    results_df = pd.DataFrame(results).transpose()
    results_df.to_csv('/mnt/iusers01/fatpou01/compsci01/w29632hl/scratch/code/haiping/scExperiment_IMC/results/damond/evaluation_results.csv')
    save_adata_dict(adata_dict, '/mnt/iusers01/fatpou01/compsci01/w29632hl/scratch/code/haiping/scExperiment_IMC/results', 'damond')

if __name__=="__main__":
    dataset_name = 'damond'
    main(dataset_name)