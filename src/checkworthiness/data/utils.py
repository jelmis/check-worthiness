import pickle
import re
import numpy as np
import pandas as pd
import os
import torch
import json
from PIL import Image

##############################
# STUFF FOR DATASET CREATIONS FROM RAW DATA
##############################

# Constant dict keys
TRAIN = "train"
DEV = "dev"
TEST = "test"
TXT = "txt"
IMG = "img"
GOLD = "gold"
SPLITS = [TRAIN, DEV, TEST, GOLD]


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


def load_data_splits_with_gold_dataset(dataset_directory, version):
    raw_dataset = {"train": [], "dev": [], "test": [], "gold": []}
    texts = {"train": [], "dev": [], "test": [], "gold": []}
    imgs = {"train": [], "dev": [], "test": [], "gold": []}
    tweet_ids = {"train": [], "dev": [], "test": [], "gold": []}
    ocr_texts = {"train": [], "dev": [], "test": [], "gold": []}
    tweet_concat_ocr = {"train": [], "dev": [], "test": [], "gold": []}

    for split in ["train", "dev", "test", "gold"]:
        if split == "gold":
            data_dir = f"{dataset_directory}_test_gold"
            split_jsonl_file = f"{dataset_directory}_test_gold/CT23_1A_checkworthy_multimodal_english_test_gold.jsonl"
            with open(split_jsonl_file, "r") as f:
                for line in f:
                    raw_dataset[split].append(json.loads(line))
                    line = json.loads(line)
                    img_path = os.path.join(data_dir, line["image_path"])
                    imgs[split].append(Image.open(img_path))
                    texts[split].append(line["tweet_text"])
                    tweet_ids[split].append(line["tweet_id"])
                    ocr_texts[split].append(line["ocr_text"])
                    tweet_concat_ocr[split].append(str(line["tweet_text"] + "\n" + line["ocr_text"]))
        else:
            data_dir = f"{dataset_directory}_{version}"
            split_name = split if split != "test" else "dev_test"
            split_jsonl_file = f"{data_dir}/CT23_1A_checkworthy_multimodal_english_{split_name}.jsonl"
            with open(split_jsonl_file, "r") as f:
                for line in f:
                    raw_dataset[split].append(json.loads(line))
                    line = json.loads(line)
                    img_path = os.path.join(data_dir, line["image_path"])
                    imgs[split].append(Image.open(img_path))
                    texts[split].append(line["tweet_text"])
                    tweet_ids[split].append(line["tweet_id"])
                    ocr_texts[split].append(line["ocr_text"])
                    tweet_concat_ocr[split].append(str(line["tweet_text"] + "\n" + line["ocr_text"]))

    print("Sizes of txt, img, ocr, txt+ocr arrays in train, test, dev, gold:")
    print(len(texts["train"]), len(imgs["train"]), len(ocr_texts["train"]), len(tweet_concat_ocr["train"]))
    print(len(texts["dev"]), len(imgs["dev"]), len(ocr_texts["train"]), len(tweet_concat_ocr["train"]))
    print(len(texts["test"]), len(imgs["test"]), len(ocr_texts["train"]), len(tweet_concat_ocr["train"]))
    print(len(texts["gold"]), len(imgs["gold"]), len(ocr_texts["train"]), len(tweet_concat_ocr["train"]))

    return raw_dataset, texts, imgs, tweet_ids, ocr_texts, tweet_concat_ocr


##############################
# EMBEDDING UTILS
##############################

def get_samples_with_excess_tokens(token_limit, split_to_tokenized_texts, padding_token):
    # Save examples with excess tokens here
    split_to_examples_with_excess_tokens = {split: [] for split in SPLITS}
    
    # Find examples with excess tokens
    for split in SPLITS:
        for idx, item in enumerate(split_to_tokenized_texts[split]):
            seq_length = (item != padding_token).nonzero(as_tuple=True)[0].shape[0]
            if seq_length > token_limit:
                split_to_examples_with_excess_tokens[split].append((idx, seq_length, seq_length - token_limit)) #record how many tokens would be truncated from which sample
        split_to_examples_with_excess_tokens[split].sort(key=lambda x: x[1])

    # Return dictionary that contains excess examples with excess tokens for every split
    return split_to_examples_with_excess_tokens


def further_normalize_samples_with_excess_tokens(token_limit, split_to_normalized_texts, split_to_samples_with_excess_tokens, split_to_tokenized, tokenizer, padding_token):
    # Copy old texts
    final_split_to_normalized_texts = split_to_normalized_texts.copy()
    
    # Remove emoji explanations and HTTPURL/@USER tokens from tweets with excess tokens
    for split, excess_examples in split_to_samples_with_excess_tokens.items():
        for excess_example in excess_examples:
            
            # Original (normalized) text
            idx, length, num_excess = excess_example
            text = split_to_normalized_texts[split][idx]

            # Remove emojis and tokeníze
            further_norm_text = re.sub("(^|\s):\S.*?\S:", "", text)
            further_norm_tokens = tokenizer(further_norm_text, padding=True, return_tensors="pt")["input_ids"][0]
            
            # New sequence length
            seq_length = (further_norm_tokens != padding_token).nonzero(as_tuple=True)[0].shape[0]
            
            # Remove multiple occurences of @USER and HTTPURL
            if seq_length > token_limit:
                # @USER
                words = np.array(further_norm_text.split())
                if "@USER" in words:
                    words = words[~(words == "@USER")].astype(str).tolist()
                    further_norm_text = " ".join(words + ["@USER"])
                # @HTTP
                words = np.array(further_norm_text.split())
                if "HTTPURL" in words:
                    words = words[~(words == "HTTPURL")].tolist()
                further_norm_text = " ".join(words + ["HTTPURL"])
                # Tokenize
                further_norm_tokens = tokenizer(further_norm_text, padding=True, return_tensors="pt")["input_ids"][0]

            # Final text the tokenizer receives
            seq_length = (further_norm_tokens != padding_token).nonzero(as_tuple=True)[0].shape[0]
            if seq_length > token_limit:
                num_excess_tokens = seq_length - token_limit
                final_split_to_normalized_texts[split][idx] = " ".join(further_norm_text.split()[:-num_excess_tokens])
            else: 
                final_split_to_normalized_texts[split][idx] = further_norm_text
    
    # Return final tokens
    return final_split_to_normalized_texts


def embed_and_pickle_split_with_bertweet(bertweet_model, pickle_dir, split_name, encoded_split, with_ocr=False, batch_size=8, dataset_version="v2", device="cpu"):
    # Gather information about the split
    ocr_method_string = f"bertweet_embeddings_with_ocr_{split_name}_{dataset_version}" if with_ocr else f"bertweet_embeddings_{split_name}_{dataset_version}"
    pickle_file = f"{pickle_dir}/{ocr_method_string}.pickle" 
    num_samples = encoded_split.shape[0]
    total_num_steps = int(num_samples / batch_size)
    
    # Show overview info
    print(f"Split: {split_name}")
    print(f"Num samples: {num_samples}")
    print(f"Num batches: {total_num_steps}")

    # Collect batch-wise embeddings here
    embedding_tensors_per_batch = []

    # Set model to eval and device
    bertweet_model.eval()
    bertweet_model.to(device)

    # Push encoded batches through the model
    for idx in range(0, num_samples, batch_size):
        print(f"{split_name} batch {int((idx+1) / batch_size)}/{total_num_steps}")
        encoded_batch = encoded_split[idx:min(num_samples, idx + batch_size)]
        encoded_batch = encoded_batch.to(device)
        with torch.no_grad():
            output = bertweet_model(encoded_batch).pooler_output
        embedding_tensors_per_batch.append(output)

    # Make one big tensor out of all batches
    all_embeddings_tensor = torch.cat(embedding_tensors_per_batch, dim=0)
    print(f"\n{ocr_method_string}: {all_embeddings_tensor.shape}")

    # Convert to Numpy array
    all_embeddings = all_embeddings_tensor.numpy()

    # Pickle the tensor with all embeddings
    with open(pickle_file, 'wb') as handle:
        pickle.dump(all_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Pickled the embeddings: {pickle_file}")

    # Return the Numpy array with all embeddings
    return all_embeddings


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
            f"Tr\t{embeddings_dict[TRAIN][TXT].shape}\t{embeddings_dict[TRAIN][IMG].shape}\n" \
            f"De\t{embeddings_dict[DEV][TXT].shape}\t{embeddings_dict[DEV][IMG].shape}\n" \
            f"Te\t{embeddings_dict[TEST][TXT].shape}\t{embeddings_dict[TEST][TXT].shape}\n" \
            f"Go\t{embeddings_dict[GOLD][TXT].shape}\t{embeddings_dict[GOLD][TXT].shape}"

    return table


def table_feature_dims_per_split(split_to_features):
    """
    Convenience function to print a table of
    the input feature dimensions per split.
    :param split_to_features: Dictionary that maps a split to its feature matrix
    :return: UTF-8 table that shows the input dimensions per split
    """
    return f"Split\tShape\n" \
           f"Tr\t{split_to_features[TRAIN].shape}\n" \
           f"De\t{split_to_features[DEV].shape}\n" \
           f"Te\t{split_to_features[TEST].shape}\n" \
           f"Go\t{split_to_features[GOLD].shape}"


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
