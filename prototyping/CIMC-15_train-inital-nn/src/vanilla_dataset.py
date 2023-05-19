import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class VanillaDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.len = self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.len

