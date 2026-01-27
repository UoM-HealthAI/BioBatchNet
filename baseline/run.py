from argparse import ArgumentParser
import scanpy as sc
from joblib import Parallel, delayed

from engines import RunBaseline
from utils import save_adata_dict
from config import BaselineConfig


def run_single_method(rb, method, seed):
    """Run a single method, return (method_name, adata, timing)"""
    result, t = rb.run_one_method(method, seed)
    return method, result[method], t[method]


def main():
    parser = ArgumentParser(description='Run baseline methods')
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--dataset', type=str, default=None, required=True, help='Dataset name')
    parser.add_argument('--save_dir', type=str, default='output', help='Save directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--methods', type=str, default=None,
                       help='Comma-separated methods to run (default: all methods)')
    parser.add_argument('--n_jobs', type=int, default=1,
                       help='Number of parallel jobs (default: 1, use -1 for all CPUs)')
    args = parser.parse_args()

    config = BaselineConfig.load(args.config_path)
    dataset_config = config.get_dataset(args.dataset)
    adata = sc.read_h5ad(dataset_config.path)

    rb = RunBaseline(adata, config, dataset_config)

    # Parse methods list
    if args.methods:
        methods_to_run = [m.strip() for m in args.methods.split(',')]
    else:
        methods_to_run = list(config.methods.keys())

    # Run methods (parallel or sequential)
    if args.n_jobs == 1:
        # Sequential
        adata_dict = {"Raw": rb.adata.copy()}
        timing = {}
        for method in methods_to_run:
            result, t = rb.run_one_method(method, args.seed)
            adata_dict[method] = result[method]
            timing.update(t)
    else:
        # Parallel
        results = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(run_single_method)(rb, method, args.seed)
            for method in methods_to_run
        )
        adata_dict = {"Raw": rb.adata.copy()}
        timing = {}
        for method, adata_result, elapsed in results:
            adata_dict[method] = adata_result
            timing[method] = elapsed

    save_adata_dict(adata_dict, args.save_dir, args.dataset)

if __name__ == "__main__":
    main()


"""
# Sequential
python run.py --dataset pancreas --methods CombatSeq

# Parallel (3 jobs)
python run.py --dataset pancreas --methods CombatSeq,FastMNN,SeuratRPCA --n_jobs 3

# Parallel (all CPUs)
python run.py --dataset pancreas --n_jobs -1
"""