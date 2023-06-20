import numpy as np

# Constant dict keys
TRAIN = "train"
DEV = "dev"
TEST = "test"
TXT = "txt"
IMG = "img"
SPLITS = [TRAIN, DEV, TEST]


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
