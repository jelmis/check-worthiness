import numpy as np
import pickle
import os
from feature_extraction import TRAIN, DEV, TEST, TXT, IMG


def get_embeddings_from_pickle_file(path_to_split):
    """
    Takes the path to a file with pickled embeddings of a certain split.
    Converts the pickled embeddings of the whole split
    into two NumPy arrays, one containing all text embeddings, one the image embeddings.
    :param path_to_split: File path to a pickle file of a certain split
    :return: txt_embeddings, img_embeddings
    """
    # Load embedding dict from pickle file
    embedding_dict = np.load(path_to_split, allow_pickle=True)

    # Convert embeddings to NumPy arrays
    txt_embeddings = np.array(embedding_dict[TXT])
    img_embeddings = np.array(embedding_dict[IMG])

    # Return text and image embeddings in separate NumPy arrays
    return txt_embeddings, img_embeddings


def table_embeddings_dims_per_split(embeddings_dict):
    """
    Convenience function to print a table of text and image
    embedding dimensions per split.
    :param embeddings_dict: e.g. embeddings[TRAIN][IMG]
    :return:
    """
    table = f"Split\ttxt\t\t\timg\n" \
            f"Tr\t\t{embeddings_dict[TRAIN][TXT].shape}\t{embeddings_dict[TRAIN][IMG].shape}\n" \
            f"De\t\t{embeddings_dict[DEV][TXT].shape}\t{embeddings_dict[DEV][IMG].shape}\n" \
            f"Te\t\t{embeddings_dict[TEST][TXT].shape}\t{embeddings_dict[TEST][TXT].shape}" \

    return table


def table_feature_dims_per_split(split_to_features):
    """
    Convenience function to print a table of
    the input feature dimensions per split.
    :param split_to_features: Dictionary that maps a split to its feature matrix
    :return: UTF-8 table that shows the input dimensions per split
    """
    return f"Split\tShape\n" \
           f"Tr\t\t{split_to_features[TRAIN].shape}\n" \
           f"De\t\t{split_to_features[DEV].shape}\n" \
           f"Te\t\t{split_to_features[TEST].shape}"


def pickle_features_or_labels(features, pickle_file):
    """
    Takes the feature matrix of one split and pickles it.
    :param features: np.array holding all input features/labels
    :param pickle_file: Path to pickle file
    :return:
    """
    with open(pickle_file, 'wb') as handle:
        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Pickled: {pickle_file}")


def pickle_all_splits(split_to_features, directory, feature_method, dataset_version, reload_and_check=False):
    """
    Takes a dictionary that maps a split to its feature matrix and pickles every split's
    feature matrix in a separate file.
    :param split_to_features: Dictionary that maps a split to its feature matrix
    :param directory: Directory where every pickle file is stored
    :param feature_method: Specifies the feature extraction method in the file name
    :param dataset_version: Which dataset version is used?
    :param reload_and_check: Shall every pickled matrix be reloaded for a sanity check?
    :return:
    """
    # Pickle feature matrix for every split
    for split, features in split_to_features.items():

        # Pickle current split's features
        pickle_file = f"{directory}/features/{feature_method}/{feature_method}_{split}_{dataset_version}.pickle"
        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
        pickle_features_or_labels(features, pickle_file)

        # Check if pickled and initial feature matrix are the same
        if reload_and_check:
            print(f"Pickled and initial feature matrix same? "
                  f"{np.array_equal(features, np.load(pickle_file, allow_pickle=True))}\n")
