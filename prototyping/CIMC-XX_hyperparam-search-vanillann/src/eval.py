import torch
import numpy as np
import itertools
from sklearn.metrics import classification_report


def evaluate_model(model, dataloader, confidence):
    """
    Evaluates a model on a given data set.
    :param model: Trained model
    :param dataloader: Dataloder of the evaluation set
    :param confidence: Binary prediction threshold for the positive class
    :return:
    """
    predictions = []
    actual_labels = []
    total = 0
    correct = 0
    with torch.no_grad():
        for input_features, label in dataloader:
            outputs = model(input_features)
            predicted = np.where(outputs < confidence, 0, 1)
            predicted = list(itertools.chain(*predicted))
            predictions.append(predicted)
            actual_labels.append(label)
            total += label.size(0)
            correct += (predicted == label.numpy()).sum().item()

    predictions = list(itertools.chain(*predictions))
    actual_labels = list(itertools.chain(*actual_labels))

    scores = classification_report(actual_labels, predictions, output_dict=True)

    return scores


def pretty_print_metrics(scores_dict, threshold):
    """
    Pretty prints the most important metrics from scikit-learns report dict.
    """
    string = f"Threshold={threshold}\n"
    for cls, metrics in list(scores_dict.items())[:3]:
        string += f"{cls}: {metrics}\n"
    return string


def write_eval_to_file(file_path, report_string):
    """
    Writes evaluation metrics to a file.
    """
    with open(file_path, 'w') as f:
        f.write(report_string)
        return report_string

