import numpy as np
from torch.utils.data import Dataset

class SCData(Dataset):
    """
    This is a simple pytorch dataset class for batch training.
    Each sample represents a cell. Each dimension represents a single gene.
    """
    def __init__(self, D, weight=None):
        """
        D: [N x G] cell by gene data matrix
        weight: (optional) [N x 1] training weight of each sample
        """
        self.M, self.N = D.shape[0], D.shape[1]//2
        self.data = D
        self.weight = np.ones((self.M, self.N)) if weight is None else weight
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.weight[idx], idx

class SCLabeledData(Dataset):
    """
    This is a simple pytorch dataset class for batch training.
    Each sample represents a cell. Each dimension represents a single gene.
    THe dataset also contains the cell labels (types).
    """
    def __init__(self, D, labels, weight=None):
        """
        D: [N x G] cell by gene data matrix
        labels: [N x 1] cell type information
        weight: (optional) [N x 1] training weight of each sample
        """
        self.M, self.N = D.shape[0], D.shape[1]//2
        self.data = D
        self.labels=labels.astype(int)
        self.weight = np.ones((self.M, self.N)) if weight is None else weight
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.weight[idx], idx
