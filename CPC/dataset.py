import torch
import numpy as np
from torch.utils.data import Dataset


class GeneralDataset(Dataset):
    def __init__(self, adata, label_key, obsm_key='X_biobatchnet'):
        """
        Args:
            adata: AnnData object
            label_key: key in adata.obs for labels
            obsm_key: key in adata.obsm for embeddings
        """
        super().__init__()
        data = adata.obsm[obsm_key]
        self.data = torch.tensor(data, dtype=torch.float32)

        # Get labels
        labels = adata.obs[label_key]
        if hasattr(labels, 'cat'):
            # Categorical type
            labels = labels.cat.codes.values
        elif not np.issubdtype(labels.dtype, np.number):
            from sklearn.preprocessing import LabelEncoder
            labels = LabelEncoder().fit_transform(labels.values)
        else:
            labels = labels.values
        self.label = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class MLDataset(Dataset):
    def __init__(self, ml_ind1, ml_ind2, adata, obsm_key='X_biobatchnet'):
        """
        Args:
            ml_ind1, ml_ind2: must-link pair indices
            adata: AnnData object
            obsm_key: key in adata.obsm for embeddings
        """
        self.ml_pairs = list(zip(ml_ind1, ml_ind2))
        data = adata.obsm[obsm_key]
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.ml_pairs)

    def __getitem__(self, idx):
        i, j = self.ml_pairs[idx]
        return self.data[i], self.data[j]


class CLDataset(Dataset):
    def __init__(self, cl_ind1, cl_ind2, adata, obsm_key='X_biobatchnet'):
        """
        Args:
            cl_ind1, cl_ind2: cannot-link pair indices
            adata: AnnData object
            obsm_key: key in adata.obsm for embeddings
        """
        self.cl_pairs = list(zip(cl_ind1, cl_ind2))
        data = adata.obsm[obsm_key]
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.cl_pairs)

    def __getitem__(self, idx):
        i, j = self.cl_pairs[idx]
        return self.data[i], self.data[j]
