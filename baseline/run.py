from argparse import ArgumentParser
import scanpy as sc
import pandas as pd

from engines import RunBaseline
from utils import save_adata_dict, logger
from config import BaselineConfig
from evaluation import evaluate
from visualization import visualize


def main():
    parser = ArgumentParser(description='Run baseline methods')
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--dataset', type=str, default=None, required=True, help='Dataset name')
    parser.add_argument('--save_dir', type=str, default='output', help='Save directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--methods', type=str, default=None,
                       help='Comma-separated methods to run (default: all methods)')
    parser.add_argument('--eval', action='store_true', help='Run evaluation after methods')
    parser.add_argument('--vis', action='store_true', help='Run visualization after methods')
    parser.add_argument('--sample_frac', type=float, default=1.0,
                       help='Sampling fraction for evaluation (default: 1.0)')
    args = parser.parse_args()

    config = BaselineConfig.load(args.config_path)
    dataset_config = config.get_dataset(args.dataset)
    adata = sc.read_h5ad(dataset_config.path)

    rb = RunBaseline(adata, config, dataset_config)

    if args.methods:
        methods_to_run = [m.strip() for m in args.methods.split(',')]
    else:
        methods_to_run = list(config.methods.keys())

    adata_dict = {"Raw": rb.adata.copy()}
    timing = {}
    for method in methods_to_run:
        result, t = rb.run_one_method(method, args.seed)
        adata_dict[method] = result[method]
        timing.update(t)

    save_adata_dict(adata_dict, args.save_dir, args.dataset)

    # Evaluation
    if args.eval:
        logger.info("Running evaluation...")
        metrics = evaluate(adata_dict, config, fraction=args.sample_frac, seed=args.seed)
        df = pd.DataFrame(metrics).T
        df.to_csv(f"{args.save_dir}/{args.dataset}/metrics.csv")
        logger.info(f"Metrics:\n{df.to_string()}")

    # Visualization
    if args.vis:
        logger.info("Running visualization...")
        vis_dir = f"{args.save_dir}/{args.dataset}/visualization"
        visualize(adata_dict, config, vis_dir)

if __name__ == "__main__":
    main()


"""
python run.py --dataset lung --methods Harmony
python run.py --dataset macaque --methods Harmony
"""