import numpy as np
from torch.utils.data import Dataset


class SCData(Dataset):
    """This is a simple pytorch dataset class for batch training.
    Each sample represents a cell. Each dimension represents a single gene.
    The dataset also contains the cell labels (types).
    """
    def __init__(self, D, labels, u0=None, s0=None, t0=None, weight=None):
        """Class constructor

        Arguments
        ---------

        D : `numpy array`
            Cell by gene data matrix, (N,G)
        labels : `numpy array`
            Cell type information, (N,1)
        u0, s0 : `numpy array`, optional
            Cell-specific initial condition, (N,G)
        t0 : `numpy array`, optional
            Cell-specific initial time, (N,1)
        weight : `numpy array`, optional
            Training weight of each sample.
        """
        self.N, self.G = D.shape[0], D.shape[1]//2
        self.data = D
        self.labels = labels
        self.u0 = u0
        self.s0 = s0
        self.t0 = t0
        self.weight = np.ones((self.N, self.G)) if weight is None else weight

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.u0 is not None and self.s0 is not None and self.t0 is not None:
            return (self.data[idx],
                    self.labels[idx],
                    self.weight[idx],
                    idx,
                    self.u0[idx],
                    self.s0[idx],
                    self.t0[idx])

        return (self.data[idx],
                self.labels[idx],
                self.weight[idx],
                idx)


class SCTimedData(Dataset):
    """
    This class is almost the same as SCData. The only difference is the addition
    of cell time. This is used for training the branching ODE.
    """
    def __init__(self, D, labels, t, u0=None, s0=None, t0=None, weight=None):
        """Class constructor

        Arguments
        ---------

        D : `numpy array`
            Cell by gene data matrix, (N,G)
        labels : `numpy array`
            Cell type information, (N,1)
        t : `numpy array`
            Cell time, (N,1)
        u0, s0 : `numpy array`, optional
            Cell-specific initial condition, (N,G)
        t0 : `numpy array`, optional
            Cell-specific initial time, (N,1)
        weight : `numpy array`, optional
            Training weight of each sample.
        """
        self.N, self.G = D.shape[0], D.shape[1]//2
        self.data = D
        self.labels = labels
        self.time = t.reshape(-1, 1)
        self.u0 = u0
        self.s0 = s0
        self.t0 = t0
        self.weight = np.ones((self.N, self.G)) if weight is None else weight

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.u0 is not None and self.s0 is not None and self.t0 is not None:
            return (self.data[idx],
                    self.labels[idx],
                    self.time[idx],
                    self.weight[idx],
                    idx,
                    self.u0[idx],
                    self.s0[idx],
                    self.t0[idx])

        return (self.data[idx],
                self.labels[idx],
                self.time[idx],
                self.weight[idx],
                idx)
