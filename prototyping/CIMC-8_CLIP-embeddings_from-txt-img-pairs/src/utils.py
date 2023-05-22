import os
import json
from PIL import Image


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
