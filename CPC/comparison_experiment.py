import argparse
import collections
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from utils import generate_random_pair, transitive_closure
from CPC.CPC_trainer import Trainer
from model import DEC
from dataset import GeneralDataset, MLDataset, CLDataset
import os
import json
import wandb  
import random
import scanpy as sc  

def run_experiment(data_path, cell_type_col, input_dim, output_dir, num_constraints_list, wandb_run, num_runs=1):
    results = {
        "ACC": [],
        "ARI": [],
        "NMI": []
    }

    for num_constraints in num_constraints_list:
        print(f"Running {num_runs} runs for num_constraints = {num_constraints}")
        
        acc_list = []
        ari_list = []
        nmi_list = []
        
        for run_idx in range(1, num_runs + 1):
            print(f"Run {run_idx}/{num_runs} for num_constraints = {num_constraints}")
            
            # fix random seed
            run_seed = 50 
            torch.manual_seed(run_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(run_seed)
            random.seed(run_seed)

            # load data
            table = pd.read_csv(data_path)
            table = table[table[cell_type_col] != 'undefined']
    
            data = table.iloc[:, 0:input_dim].values
            cell_type = table[cell_type_col].values  
            
            label_encoder = LabelEncoder()
            cell_type = label_encoder.fit_transform(cell_type)
            cell_type_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}

            # generate pairs
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = generate_random_pair(cell_type, num_constraints*2)
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, data.shape[0])
    
            ml_ind1 = ml_ind1[:num_constraints]
            ml_ind2 = ml_ind2[:num_constraints]
            cl_ind1 = cl_ind1[:num_constraints]
            cl_ind2 = cl_ind2[:num_constraints]

            # dataset
            train_dataset = GeneralDataset(data, cell_type)
            ml_dataset = MLDataset(ml_ind1, ml_ind2, data)
            cl_dataset = CLDataset(cl_ind1, cl_ind2, data)
    
            dec_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=256)
            eva_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=256)
            ml_dataloader = DataLoader(ml_dataset, shuffle=True, batch_size=128)
            cl_dataloader = DataLoader(cl_dataset, shuffle=True, batch_size=128)

            # model
            idea_model = DEC(input_dim=input_dim, latent_dim=10, n_clusters=7)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            idea_model = idea_model.to(device)
    
            trainable_params = list(filter(lambda p: p.requires_grad, idea_model.autoencoder.parameters())) 
            pretrain_optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
            cluster_optimizer = torch.optim.Adam(idea_model.parameters(), lr=1e-4)
    
            trainer = Trainer(
                idea_model, 
                cluster_optimizer, 
                pretrain_optimizer, 
                dec_dataloader, 
                ml_dataloader, 
                cl_dataloader, 
                eva_dataloader,
                device
            )
            
            trainer.train()
            
            predicted_labels, true_labels, acc, ari, nmi = trainer.evaluate()
    
            acc_list.append(acc)
            ari_list.append(ari)
            nmi_list.append(nmi)
            
        avg_acc = np.mean(acc_list)
        avg_ari = np.mean(ari_list)
        avg_nmi = np.mean(nmi_list)
    
        results["ACC"].append(avg_acc)
        results["ARI"].append(avg_ari)
        results["NMI"].append(avg_nmi)
    
        wandb_run.log({
            "num_constraints": num_constraints,
            "ACC": avg_acc,
            "ARI": avg_ari,
            "NMI": avg_nmi
        })
    
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f)
        
        print(f"Experiment completed. Results saved to {output_dir}/results.json")

    return results

def plot_results(results_processed, results_original, results_scvi, num_constraints_list, output_path, wandb_run):
    metrics = ["ACC", "ARI", "NMI"]
    colors = {
        "processed": "blue",
        "original": "red",
        "scVI": "green"
    }
    markers = {
        "processed": "o",
        "original": "s",
        "scVI": "d"
    }

    plt.figure(figsize=(18, 5))

    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        plt.plot(num_constraints_list, results_processed[metric], label='Batch Effect Removed', color=colors["processed"], marker=markers["processed"])
        plt.plot(num_constraints_list, results_original[metric], label='Original Data', color=colors["original"], marker=markers["original"])
        plt.plot(num_constraints_list, results_scvi[metric], label='scVI Data', color=colors["scVI"], marker=markers["scVI"])
        plt.title(f'{metric} vs Number of Constraints')
        plt.xlabel('Number of Constraints')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    wandb_run.log({"performance_comparison": wandb.Image(output_path)})

def main():
    SEED = 50
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    num_runs = 1

    num_constraints_list = list(range(100, 6101, 1000))

    experiments = [
        {
            "name": "BERD embed",
            "data_path": "data/IMMU_embedding.csv",
            "cell_type_col": "cell type",  
            "input_dim": 20,
            "output_dir": "results/processed"
        },
        {
            "name": "Original Data",
            "data_path": "data/IMMUcan_batch.csv",
            "cell_type_col": "cell_type",  
            "input_dim": 40,
            "output_dir": "results/original"
        },
        {
            "name": "scVI embed",
            "data_path": "data/IMMU_scvi_embeddings.csv",
            "cell_type_col": "cell_type",  
            "input_dim": 20,
            "output_dir": "results/scvi"
        }
    ]

    all_results = {}

    for exp in experiments:
        print(f"Starting experiment: {exp['name']}")
        with wandb.init(project="C-DEC", name=exp["name"], reinit=True) as run:
            run.config.update({
                "data_path": exp["data_path"],
                "cell_type_col": exp["cell_type_col"],
                "input_dim": exp["input_dim"],
                "output_dir": exp["output_dir"],
                "num_constraints_list": num_constraints_list,
                "num_runs_per_constraint": num_runs,
                "random_seed": SEED
            })

            results = run_experiment(
                data_path=exp["data_path"],
                cell_type_col=exp["cell_type_col"],
                input_dim=exp["input_dim"],
                output_dir=exp["output_dir"],
                num_constraints_list=num_constraints_list,
                wandb_run=run,
                num_runs=num_runs
            )
            all_results[exp["name"]] = results

    with open("results/processed/results.json", "r") as f:
        results_processed = json.load(f)
    with open("results/original/results.json", "r") as f:
        results_original = json.load(f)
    with open("results/scvi/results.json", "r") as f:
        results_scvi = json.load(f)

    with wandb.init(project="C-DEC", name="Performance Comparison", reinit=True) as run_plot:
        plot_results(results_processed, results_original, results_scvi, num_constraints_list, "results/performance_comparison.png", run_plot)

    wandb.finish()

if __name__ == "__main__":
    main()
