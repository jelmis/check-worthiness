import json

"""
This file enables to compute stats about the dataset:
1) class label distribution
2) fraction of examples with ocr text
"""

# Constant JSON keys/values
KEY_LABEL = "class_label"
KEY_OCR = "ocr_text"
LABEL_YES = "Yes"

# Read json file
file = "/home/jockl/CT23_1A_checkworthy_multimodal_english_v1-20230428T150127Z-001/CT23_1A_checkworthy_multimodal_english_v1/CT23_1A_checkworthy_multimodal_english_dev_test.jsonl"
json_objects = [json.loads(line) for line in open(file, "r")]

# Compute fraction of examples with positive label
total_num_examples = len(json_objects)
num_yes = sum([1 if example[KEY_LABEL] == LABEL_YES else 0 for example in json_objects])
fraction_yes = num_yes / total_num_examples

# Compute fraction of examples with OCR text
num_ocr = sum([1 if example[KEY_OCR] != "" else 0 for example in json_objects])
fraction_ocr = num_ocr / total_num_examples

# Report stats
print(f"{file}\n"
      f"Total examples: {total_num_examples}\n"
      f"Pos. examples: {num_yes} ({fraction_yes})\n"
      f"OCR examples: {num_ocr} ({fraction_ocr})")