import numpy as np

# Constant dict keys
TRAIN = "train"
DEV = "dev"
TEST = "test"
TXT = "txt"
IMG = "img"
SPLITS = [TRAIN, DEV, TEST]


def cosine(txt_emb, img_emb):
    """
    Takes a text embedding an image embedding and computes their cosine similarity
    :param txt_emb: np.Array
    :param img_emb: np.Array
    :return: cosine similarity
    """
    cos = np.dot(txt_emb, img_emb) / (np.linalg.norm(txt_emb) * np.linalg.norm(img_emb))
    return cos


def compute_cosine_array(txt_embs, img_embs):
    """
    Takes a matrix of corresponding text-image-embeddings and computes
    an array that holds the cosine similarity for every embedding.
    :param txt_embs: embedding matrix
    :param img_embs: image matrix
    :return: array with cosine similarity for every text-image pair
    """
    # Compute cosine sim for every pair
    cosine_list = []
    for i in range(len(txt_embs)):
        txt, img = txt_embs[i], img_embs[i]
        cosine_list.append(cosine(txt, img))

    # Array with cosine similarities between every text-image pair
    return np.array(cosine_list)


def add_feature_dim(feature_matrix, new_feature_array):
    """
    Takes a feature matrix and a 1D-array and appends the array
    as a new feature dimension to the feature matrix.
    :param feature_matrix: (num_examples, num_features)
    :param new_feature_array: (1, num_examples)
    :return:
    """
    # Copy feature matrix to preserve old version
    feat_matrix = feature_matrix.copy()

    # Fit to-be-added array
    new_dim = new_feature_array.reshape(len(feat_matrix), 1)

    # Append new dimension
    expanded_feat_matrix = np.concatenate((feat_matrix, new_dim), axis=1)

    return expanded_feat_matrix


def add_feature_dim_to_all_splits(split_to_features, split_to_new_dim):
    """
    Takes a dictionary that maps every split to its feature matrix and a dictionary
    that maps every split to its new feature 1D-array.
    Appends the array as a new feature dimension to the feature matrix of every split.
    :param split_to_features: split_to_features[TRAIN] is of shape (num_examples, num_features)
    :param split_to_new_dim: split_to_new_dim[TRAIN] is of shape (1, num_examples)
    :return: Dictionary with appended cosine sim dimension for every split's matrix
    """
    # Copy dict to preserve old dict
    split_to_feats = split_to_features.copy()

    # Add cosine dim to every split
    for split in [TRAIN, DEV, TEST]:
        feature_matrix = split_to_features[split]
        new_feature_array = split_to_new_dim[split]
        split_to_feats[split] = add_feature_dim(feature_matrix, new_feature_array)

    return split_to_feats
