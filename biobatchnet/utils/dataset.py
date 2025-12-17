"""Dataset classes for batch effect correction."""
import torch
from torch.utils.data import Dataset
import numpy as np


class BBNDataset(Dataset):
    """Simple dataset for BioBatchNet."""

    def __init__(self, data, batch_labels, cell_types=None):
        self.data = torch.tensor(np.array(data), dtype=torch.float32)
        self.batch_labels = torch.tensor(np.array(batch_labels), dtype=torch.long)
        self.cell_types = torch.tensor(np.array(cell_types), dtype=torch.long) if cell_types is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.cell_types is not None:
            return self.data[idx], self.batch_labels[idx], self.cell_types[idx]
        return self.data[idx], self.batch_labels[idx]


class MLDataset(Dataset):
    """Must-link pairs dataset."""
    def __init__(self, ml_ind1, ml_ind2, data):
        self.ml_pairs = list(zip(ml_ind1, ml_ind2))
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.ml_pairs)

    def __getitem__(self, idx):
        i, j = self.ml_pairs[idx]
        return self.data[i], self.data[j]


class CLDataset(Dataset):
    """Cannot-link pairs dataset."""
    def __init__(self, cl_ind1, cl_ind2, data):
        self.cl_pairs = list(zip(cl_ind1, cl_ind2))
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.cl_pairs)

    def __getitem__(self, idx):
        i, j = self.cl_pairs[idx]
        return self.data[i], self.data[j]
