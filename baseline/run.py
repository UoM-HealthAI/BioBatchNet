from argparse import ArgumentParser
import os
import scanpy as sc

from engines import RunBaseline
from utils import save_adata_dict
from config import BaselineConfig

def main():
    parser = ArgumentParser(description='Run baseline methods')
    parser.add_argument('--config_path', type=str, default=None, required=True, help='Config file')
    parser.add_argument('--dataset', type=str, default=None, required=True, help='Dataset name') 
    parser.add_argument('--save_dir', type=str, default='baseline/output', help='Save directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--methods', type=str, default=None, choices=['scVI', 'iMAP', 'MRVI', 'Harmony', 'BBKNN', 'Scanorama', 'Combat'],
                       help='Methods to run (default: all methods)')
    args = parser.parse_args()

    config = BaselineConfig.load(args.config_path)
    dataset_config = config.get_dataset(args.dataset)
    adata = sc.read_h5ad(dataset_config.path)

    rb = RunBaseline(adata, config, dataset_config)
    
    # Parse methods list
    if args.methods:
        methods_to_run = [m.strip() for m in args.methods.split(',')]
        adata_dict, timing = rb.run_selected_methods(methods_to_run, args.seed)
    else:
        adata_dict, timing = rb.run_one_seed(args.seed)

    save_adata_dict(adata_dict, args.save_dir, args.dataset)

if __name__ == "__main__":
    main()