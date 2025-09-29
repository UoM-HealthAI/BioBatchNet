import yaml
import json
import torch
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import random

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def read_yaml(file_path):
    """Read YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def write_yaml(data, file_path):
    """Write data to a YAML file."""
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys):
        self.data = {key: 0 for key in keys}
        self.counts = {key: 0 for key in keys}

    def update(self, key, value, count=1):
        self.data[key] += value * count
        self.counts[key] += count

    def update_batch(self, metrics_dict, count=1):
        for key, value in metrics_dict.items():
            self.update(key, value, count)

    def avg(self, key):
        return self.data[key] / self.counts[key] if self.counts[key] > 0 else 0

    def reset(self):
        for key in self.data:
            self.data[key] = 0
            self.counts[key] = 0

    def result(self):
        return {key: self.avg(key) for key in self.data}

    def log_metrics(self, extra_metrics=None):
        """Log metrics (placeholder; extend with your logger of choice)."""
        metrics = self.result()
        if extra_metrics:
            metrics.update(extra_metrics)
        return metrics  # Disabled for API usage
        

def visualization(save_dir, adata, emb, epoch):
    sc.pp.subsample(adata, fraction=0.3)
    sc.pp.neighbors(adata, use_rep=emb)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=['BATCH', 'celltype'], frameon=False)
    plt.savefig(f'{save_dir}/{emb}_{epoch}_umap.png')

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
