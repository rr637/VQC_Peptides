import numpy as np
import torch
from torch.utils.data import Dataset

class PeptideDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('./data_parsed/data.csv', delimiter=",", dtype=np.float32)
        self.x = torch.from_numpy(xy[:, :12])
        self.y = torch.from_numpy(xy[:, [-1]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
class ToyPeptideDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('./data_parsed/6AA_data_neg_norm.csv', delimiter=",", dtype=np.float32)
        self.x = torch.from_numpy(xy[:, :6])
        self.y = torch.from_numpy(xy[:, [-1]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples