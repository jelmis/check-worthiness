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


def load_data_splits_with_gold_dataset(dataset_directory, version):
        raw_dataset = {"train": [], "dev": [], "test": [], "gold": []}
        texts = {"train": [], "dev": [], "test": [], "gold": []}
        imgs = {"train": [], "dev": [], "test": [], "gold": []}
        tweet_ids = {"train": [], "dev": [], "test": [], "gold": []}

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

        print("Sizes of train/test/dev/gold txt and img arrays respectively: ")
        print(len(texts["train"]), len(imgs["train"]))
        print(len(texts["dev"]), len(imgs["dev"]))
        print(len(texts["test"]), len(imgs["test"]))
        print(len(texts["gold"]), len(imgs["gold"]))

        return raw_dataset, texts, imgs, tweet_ids