import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


def custom_dataloader(dataset, split, batch_size):
    if split != 'train':
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    labels = dataset.get_labels()
    labels = labels.long()

    class_counts = torch.bincount(labels)

    class_weights = 1.0 / class_counts.float()

    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(sample_weights, len(dataset), replacement=True)

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
