import torch
from torch.utils.data import Dataset
import numpy as np

class UserIMCDataset(Dataset):
    """Simple dataset for user-provided IMC data"""
    def __init__(self, data, batch_labels):
        self.data = torch.FloatTensor(data) if not isinstance(data, torch.Tensor) else data
        self.batch_labels = torch.LongTensor(batch_labels) if not isinstance(batch_labels, torch.Tensor) else batch_labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.batch_labels[idx]