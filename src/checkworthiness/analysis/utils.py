import pandas as pd


def extract_datapoints(
    df_dict: dict[pd.DataFrame],
    raw_dataset: dict,
    texts: dict,
    tweet_ids: dict,
) -> pd.DataFrame:
    df_data = {
        "tweet_id": [],
        "tweet": [],
        "true_label": [],
        "type": [],
        # "model_pred": [],
        "image": [],
    }
    for df_name, df in df_dict.items():
        # model_prediction = model_prediction_dict[idx + 1]
        for _, row in df.iterrows():
            split = row["split"]
            datapoint_id = row["datapoint_id"]
            actual_label = row["actual_label"]
            text = texts[split][datapoint_id]
            img_url = raw_dataset[split][datapoint_id]["image_url"]
            tweet_id = tweet_ids[split][datapoint_id]
            df_data["tweet_id"].append(tweet_id)
            df_data["tweet"].append(text)
            df_data["true_label"].append(actual_label)
            df_data["type"].append(df_name)
            # df_data["model_pred"].append(model_prediction)
            df_data["image"].append(img_url)
    return pd.DataFrame(df_data)
