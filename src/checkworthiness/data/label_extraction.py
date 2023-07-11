import os
import json
import numpy as np

# Constant dict keys
LABEL = "class_label"
YES = "Yes"
NO = "No"
ID = "tweet_id"


def get_labels_from_dataset(split_json_file):
    """
    Extracts the labels from a split and returns them as np.array
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


def get_tweet_ids_from_dataset(split_json_file):
    """
    Extracts the tweet ids from a split and returns them as np.array
    :param split_json_file: path to a split's JSON file
    :return: np.array containing the tweet ids
    """
    # Initialize label list
    id_list = []

    # Extract labels from JSON file
    with open(split_json_file, "r") as f:
        for line in f:
            line = json.loads(line)
            id = line[ID] 
            id_list.append(id)

    # Convert label list to np.array
    ids = np.array(id_list)

    # All labels of the split
    return ids