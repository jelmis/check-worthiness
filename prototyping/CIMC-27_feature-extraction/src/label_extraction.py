import os
import json
import numpy as np

# Constant dict keys
LABEL = "class_label"
YES = "Yes"
NO = "No"


def get_labels_from_dataset(split_json_file):
    """
    Extracts the features from a split and returns them as np.array
    :param split_json_file: path to a split's JSON file
    :return: np.array containing the labels ("No" = 0, "Yes" = 1)
    """
    # Initialize label list
    label_list = []

    # Extract labels from JSON file
    with open(split_json_file, "r") as f:
        for line in f:
            line = json.loads(line)
            label = 1 if line[LABEL] == YES else 0
            label_list.append(label)

    # Convert label list to np.array
    labels = np.array(label_list)

    # All labels of the split
    return labels
