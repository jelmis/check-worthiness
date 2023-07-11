import pickle
import numpy as np
import os
import json
from PIL import Image

##############################
# STUFF FOR DATASET CREATIONS FROM RAW DATA
##############################


def load_dataset(dataset_directory):
    raw_dataset = {"train": [], "dev": [], "test": []}
    texts = {"train": [], "dev": [], "test": []}
    imgs = {"train": [], "dev": [], "test": []}
    tweet_ids = {"train": [], "dev": [], "test": []}

    train_set = []
    for split in ["train", "dev", "test"]:
        if split == "test":
            split_jsonl_file = os.path.join(dataset_directory, f"CT23_1A_checkworthy_multimodal_english_dev_{split}.jsonl")
        else:
            split_jsonl_file = os.path.join(dataset_directory, f"CT23_1A_checkworthy_multimodal_english_{split}.jsonl")
        with open(split_jsonl_file, "r") as f:
            for line in f:
                raw_dataset[split].append(json.loads(line))
                line = json.loads(line)
                img_path = os.path.join(dataset_directory, line["image_path"])
                imgs[split].append(Image.open(img_path))
                texts[split].append(line["tweet_text"])
                tweet_ids[split].append(line["tweet_id"])

    print("Sizes of train/test/dev txt and img arrays respectively: ")
    print(len(texts["train"]), len(imgs["train"]))
    print(len(texts["dev"]), len(imgs["dev"]))
    print(len(texts["test"]), len(imgs["test"]))

    return raw_dataset, texts, imgs, tweet_ids


def load_data_splits_with_gold_dataset(dataset_directory, version, **kwargs):
    splits = kwargs.get("selected_split", None)
    if splits is None:
        splits = ["train", "dev", "test", "gold"]
    if not isinstance(splits, list):
        splits = [splits]

    raw_dataset = {"train": [], "dev": [], "test": [], "gold": []}
    texts = {"train": [], "dev": [], "test": [], "gold": []}
    imgs = {"train": [], "dev": [], "test": [], "gold": []}
    tweet_ids = {"train": [], "dev": [], "test": [], "gold": []}

    for split in splits:
        if split == "gold":
            data_dir = f"{dataset_directory}_test_gold"
            split_jsonl_file = f"{dataset_directory}_test_gold/CT23_1A_checkworthy_multimodal_english_test_gold.jsonl"
        else:
            data_dir = f"{dataset_directory}_{version}"
            split_name = split if split != "test" else "dev_test"
            split_jsonl_file = f"{data_dir}/CT23_1A_checkworthy_multimodal_english_{split_name}.jsonl"
        with open(split_jsonl_file, "r") as f:
            for idx, line in enumerate(f):
                raw_dataset[split].append(json.loads(line))
                line = json.loads(line)
                img_path = os.path.join(data_dir, line["image_path"])
                imgs[split].append(Image.open(img_path))
                texts[split].append(line["tweet_text"])
                tweet_ids[split].append(line["tweet_id"])

    print("Sizes of train/test/dev/gold txt and img arrays respectively: ")
    print(len(texts["train"]), len(imgs["train"]))
    print(len(texts["dev"]), len(imgs["dev"]))
    print(len(texts["test"]), len(imgs["test"]))
    print(len(texts["gold"]), len(imgs["gold"]))

    return raw_dataset, texts, imgs, tweet_ids


##############################
# BEYOND THIS POINT: FEATURE EXTRACTION DATA UTILS
##############################


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
    table = f"Split\ttxt\t\timg\n" \
            f"Tr\t{embeddings_dict['train']['txt'].shape}\t{embeddings_dict['train']['img'].shape}\n" \
            f"De\t{embeddings_dict['dev']['txt'].shape}\t{embeddings_dict['dev']['img'].shape}\n" \
            f"Te\t{embeddings_dict['test']['txt'].shape}\t{embeddings_dict['test']['txt'].shape}\n" \
            f"Go\t{embeddings_dict['gold']['txt'].shape}\t{embeddings_dict['gold']['txt'].shape}"

    return table


def table_feature_dims_per_split(split_to_features):
    """
    Convenience function to print a table of
    the input feature dimensions per split.
    :param split_to_features: Dictionary that maps a split to its feature matrix
    :return: UTF-8 table that shows the input dimensions per split
    """
    return f"Split\tShape\n" \
           f"Tr\t{split_to_features['train'].shape}\n" \
           f"De\t{split_to_features['dev'].shape}\n" \
           f"Te\t{split_to_features['test'].shape}\n" \
           f"Go\t{split_to_features['gold'].shape}"


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
        if split == GOLD:
            pickle_file = f"{directory}/features/{feature_method}/{feature_method}_{split}.pickle"
        else:
            pickle_file = f"{directory}/features/{feature_method}/{feature_method}_{split}_{dataset_version}.pickle"
        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
        pickle_features_or_labels(features, pickle_file)

        # Check if pickled and initial feature matrix are the same
        if reload_and_check:
            print(f"Pickled and initial feature matrix same? "
                  f"{np.array_equal(features, np.load(pickle_file, allow_pickle=True))}\n")
