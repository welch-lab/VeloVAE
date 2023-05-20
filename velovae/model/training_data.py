import numpy as np
from torch.utils.data import Dataset


class SCData(Dataset):
    """This is a simple pytorch dataset class for batch training.
    Each sample represents a cell. Each dimension represents an 
    unspliced or spliced count number of a single gene.
    The dataset also contains the cell labels (types).
    """
    def __init__(self, D, labels, u0=None, s0=None, t0=None, weight=None):
        """Constructor

        Args:
            D (:class:`numpy array`):
                Cell-by-gene data matrix, (N, G)
            labels (:class:`numpy array`):
                Cell type annotation, (N, 1)
            u0 (:class:`numpy array`, optional):
                Cell-by-gene unspliced initial condition, (N, G). Defaults to None.
            s0 (:class:`numpy array`, optional):
                Cell-by-gene spliced initial condition, (N, G). Defaults to None.
            t0 (:class:`numpy array`, optional): 
                Time at the initial condition for each cell. Defaults to None.
            weight (:class:`numpy array`, optional):
                Sample weight. Defaults to None.
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
        """Constructor

        Args:
            D (:class:`numpy array`):
                Cell-by-gene data matrix, (N, G)
            labels (:class:`numpy array`):
                Cell type annotation, (N, 1)
            t (:class:`numpy array`):
                Cell time, (N, 1)
            u0 (:class:`numpy array`, optional):
                Cell-by-gene unspliced initial condition, (N, G). Defaults to None.
            s0 (:class:`numpy array`, optional):
                Cell-by-gene spliced initial condition, (N, G). Defaults to None.
            t0 (:class:`numpy array`, optional): 
                Time at the initial condition for each cell. Defaults to None.
            weight (:class:`numpy array`, optional):
                Sample weight. Defaults to None.
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
