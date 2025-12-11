import math
import matplotlib.pyplot as plt
import scanpy as sc
from matplotlib.lines import Line2D
from matplotlib import cm
from utils import save_adata_dict, load_h5ad_files, sampling
import os
from evaluation import evaluate
from tqdm import tqdm
import pandas as pd

def cal_nn(adata_dict):
    results = {}

    for key, adata in tqdm(adata_dict.items(), desc="calculation nn for each adata", unit="adata"):
        if key == 'Raw':
            sc.pp.pca(adata)
            sc.pp.neighbors(adata, use_rep='X_pca')
            continue

        elif key == 'Harmony':
            embed = 'X_pca_harmony'
            sc.pp.neighbors(adata, use_rep=embed)

        elif key == 'BBKNN':
            embed = 'X_pca'
        
        elif key ==  'scVI':
            embed = 'X_scvi'
            sc.pp.neighbors(adata, use_rep=embed)  

        elif key ==  'scDREAMER':
            embed = 'X_scdreamer'
            sc.pp.neighbors(adata, use_rep=embed)  
        
        elif key ==  'BioBatchNet':
            embed = 'X_biobatchnet'
            sc.pp.neighbors(adata, use_rep=embed)  

        elif key ==  'scDML':
            embed = 'X_emb'
            sc.pp.neighbors(adata, use_rep=embed)  
        
        elif key == 'Scanorama':
            embed = 'X_scanorama'
            sc.pp.neighbors(adata, use_rep=embed)  

        else:
            # imap combat
            embed = 'X_pca'
            sc.pp.pca(adata)  
            sc.pp.neighbors(adata, use_rep=embed)  

    return results

def plot_umap(adata_dict, color, save_dir=None):

    methods = ['Raw'] + [k for k in adata_dict.keys() if k != 'Raw']
    n_methods = len(methods)

    n_rows = 2
    n_cols = math.ceil(n_methods / n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), constrained_layout=False)
    axes = axes.flatten()

    unique_categories = set()
    for method in methods:
        adata = adata_dict[method]
        unique_categories.update(adata.obs[color].unique())
    unique_categories = sorted(unique_categories)
    n_categories = len(unique_categories)
    
    handles = []
    labels = []
    
    if n_categories <= 20:
        palette = sc.pl.palettes.default_20
    else:
        palette = sc.pl.palettes.default_102
    category_color_map = {cat: palette[i] for i, cat in enumerate(unique_categories)}
    
    for i, method in enumerate(tqdm(methods, desc="Processing methods", unit="method")):
        ax = axes[i]
        adata = adata_dict[method]

        sc.tl.umap(adata)
        sc.pl.umap(
            adata,
            color=color,
            ax=ax,
            show=False,
            legend_loc=None, 
            frameon=False,
            title=f"{method}",
            palette=[category_color_map[cat] for cat in unique_categories]  
        )
        ax.set_xlabel('')
        ax.set_ylabel('')

    for j in range(n_methods, len(axes)):
        fig.delaxes(axes[j])
        
    for category in unique_categories:
        handles.append(Line2D([0], [0], marker='o', color='w', label=category,
                                markerfacecolor=category_color_map[category], markersize=10))
        labels.append(category)
    
    fig.legend(
        handles, labels, loc='upper center', ncol=min(len(labels), 10),
        fontsize=12, bbox_to_anchor=(0.5, 0.05)
    )
    
    plt.subplots_adjust(bottom=0.07, top=0.95, hspace=0.1, wspace=0.1)
    plt.show()
    if save_dir is not None:
        plt.savefig(save_dir, bbox_inches='tight')   

 
def plot_batch_and_celltype(adata_dict, save_dir_batch, save_dir_celltype):
    plot_umap(adata_dict, color='BATCH', save_dir=save_dir_batch)
    plot_umap(adata_dict, color='celltype', save_dir=save_dir_celltype)

def plot_main(directory, data_name):
    base_save_dir = '/mnt/iusers01/fatpou01/compsci01/w29632hl/scratch/code/haiping/scExperiment_IMC/results'
    save_dir = os.path.join(base_save_dir, data_name)
    os.makedirs(save_dir, exist_ok=True)
    # load adata_dict
    adata_dict = load_h5ad_files(directory)

    # sampling
    sampling_adata_dict = sampling(adata_dict)

    # evaluate 
    # results = evaluate(sampling_adata_dict)
    # results_df = pd.DataFrame(results).transpose()
    # results_df.to_csv(os.path.join(base_save_dir, 'evaluation.csv'))

    # calculate nn
    # cal_nn(sampling_adata_dict)

    # plot
    plot_batch_and_celltype(sampling_adata_dict,
                            os.path.join(save_dir, 'batch.png'),
                            os.path.join(save_dir, 'celltype.png'))

if __name__ == "__main__":
    imc_base_dir = '/mnt/iusers01/fatpou01/compsci01/w29632hl/scratch/Data/IMC_data'
    imc_dataname = 'Damond'

    gene_base_dir = '/mnt/iusers01/fatpou01/compsci01/w29632hl/scratch/Data/Gene results'
    gene_dataname = 'mouse_brain'

    directory = os.path.join(gene_base_dir, gene_dataname)
    plot_main(directory, gene_dataname)
    