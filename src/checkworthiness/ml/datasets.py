import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class VanillaDataset(Dataset):
    def __init__(self, features, labels, tweet_ids=None):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.tweet_ids = (
            torch.from_numpy(tweet_ids.astype(int)) if tweet_ids is not None else None
        )
        self.len = self.features.shape[0]

    def __getitem__(self, index):
        if self.tweet_ids is not None:
            return self.features[index], self.labels[index], self.tweet_ids[index]
        else:
            return self.features[index], self.labels[index]

    def __len__(self):
        return self.len
