import torch
import numpy as np
import itertools
from sklearn.metrics import classification_report


def evaluate_model(network, dev_dataloader):
    """
    Used to evaluate the trained models on the fly, using the dev set.
    :param network:
    :param dev_dataloader:
    :return:
    """
    predictions = []
    actual_labels = []
    total = 0
    correct = 0
    with torch.no_grad():
        for input_features, label in dev_dataloader:
            outputs = network(input_features)
            predicted = np.where(outputs < 0.5, 0, 1)
            predicted = list(itertools.chain(*predicted))
            predictions.append(predicted)
            actual_labels.append(label)
            total += label.size(0)
            correct += (predicted == label.numpy()).sum().item()

    predictions = list(itertools.chain(*predictions))
    actual_labels = list(itertools.chain(*actual_labels))

    return classification_report(actual_labels, predictions)
