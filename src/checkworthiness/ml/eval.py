import torch
import csv
import numpy as np
import itertools
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, SubsetRandomSampler


def get_output_scores(model, dataloader):
    """
    Gets the last layer scores from a model on a given data set.
    :param model: Trained model
    :param dataloader: Dataloder of the evaluation set
    :return:
    """
    output_scores = []
    actual_labels = []

    with torch.no_grad():
        for input_features, label in dataloader:
            outputs = model(input_features).flatten().tolist()
            output_scores.append(outputs)
            actual_labels.append(label.flatten().tolist())

    output_scores = list(itertools.chain(*output_scores))
    actual_labels = list(itertools.chain(*actual_labels))

    return output_scores, actual_labels


def evaluate_model(model, dataloader, confidence):
    """
    Evaluates a model on a given data set.
    :param model: Trained model
    :param dataloader: Dataloader of the evaluation set
    :param confidence: Binary prediction threshold for the positive class
    :return:
    """
    predictions = []
    actual_labels = []
    total = 0
    correct = 0
    with torch.no_grad():
        for input_features, label in dataloader:
            outputs = model(input_features)
            predicted = np.where(outputs < confidence, 0, 1)
            predicted = list(itertools.chain(*predicted))
            predictions.append(predicted)
            actual_labels.append(label)
            total += label.size(0)
            correct += (predicted == label.numpy()).sum().item()

    predictions = list(itertools.chain(*predictions))
    actual_labels = list(itertools.chain(*actual_labels))

    scores = classification_report(actual_labels, predictions, output_dict=True)

    return scores


def cross_validate_thresholds(model, cross_val_dataset, splits, thresholds, batch_size):
    """
    Performs cross-validation of a model for different threshold
    :param model: Trained Model
    :param cross_val_dataset: PyTorch dataset
    :param splits: sklearn KFold object
    :param thresholds: List of threshold to validate on
    :return: Tuple: (String that reports best average thresholds for every split, best mean threshold)
    """
    # Collect best f1 score and corresponding threshold for every split
    report_string = ""
    best_f1_list, best_threshold_list = [], []
    for num_fold, (_, val_idx) in enumerate(splits.split(np.arange(len(cross_val_dataset)))):

        # Init best f_1 and best_threshold for current split
        best_f1, best_threshold = 0, 0
        best_metrics_string = ""

        # Get left-out fold to validate on
        val_sampler = SubsetRandomSampler(val_idx)
        val_loader = DataLoader(cross_val_dataset, batch_size=batch_size, sampler=val_sampler)

        # Compute metrics for every threshold on current left-out
        for threshold in thresholds:
            scores_dict = evaluate_model(model=model, dataloader=val_loader, confidence=threshold)
            f1 = scores_dict["1.0"]["f1-score"]
            metrics_string = pretty_print_metrics(scores_dict, threshold) + "\n"
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics_string = metrics_string

        # Save best f1 score for current split
        best_f1_list.append(best_f1)
        best_threshold_list.append(best_threshold)
        report_string += f"Best on fold {num_fold}: {best_metrics_string}"
        # print(f"Best on fold {num_fold}: {best_metrics_string}")

    # Compute best f1/threshold on average
    mean_best_f1 = np.mean(np.array(best_f1_list))
    mean_best_threshold = np.mean(np.array(best_threshold_list))
    report_string += f"Mean of best thresholds: {mean_best_threshold}\nMean of best f1-scores: {mean_best_f1}"

    # Return
    return report_string, mean_best_threshold


def pretty_print_metrics(scores_dict, threshold):
    """
    Pretty prints the most important metrics from scikit-learns report dict.
    """
    string = f"Threshold={threshold}\n"
    for cls, metrics in list(scores_dict.items())[:3]:
        string += f"{cls}: {metrics}\n"
    return string


def write_eval_to_file(file_path, report_string):
    """
    Writes evaluation metrics to a file.
    """
    with open(file_path, 'w') as f:
        f.write(report_string)
        return report_string

def get_tweet_ids_and_pred_from_model(model, dataloader, confidence):
    """
    Runs a model on a given data set.
    :param model: Trained model
    :param dataloader: Dataloader of the evaluation set
    :param confidence: Binary prediction threshold for the positive class
    :return: zip(tweet_ids, predictions)
    """
    predictions = []
    tweet_ids = []
    with torch.no_grad():
        for input_features, label, tweet_id in dataloader:
            outputs = model(input_features)
            predicted = np.where(outputs < confidence, 0, 1)
            predicted = list(itertools.chain(*predicted))
            predictions.append(predicted)
            tweet_ids.append(tweet_id)

    predictions = list(itertools.chain(*predictions))
    predictions = ["Yes" if pred == 1 else "No" for pred in predictions]
    tweet_ids = [int(id.item()) for id in list(itertools.chain(*tweet_ids))]

    return zip(tweet_ids, predictions)


def create_submission_file_from_model(model, dataloader, confidence, file_path, run_id="our_model"):
    """
    Writes the models prediction's to a .tsv submission file in the specified format:
    1235648554338791427	No  our_model
    1235287380292235264	Yes  our_model
    1236020820947931136	No  our_model
    30313	No  Model_1
    :param model: Trained model
    :param dataloader: Dataloader of the evaluation set
    :param confidence: Binary prediction threshold for the positive class
    :param confidence: Path to submission file
    :param run_id: Our choice of how to name the model in the submission file
    :return: 
    """
    # Get the ids with preds
    list_of_id_pred_tuples = list(get_tweet_ids_and_pred_from_model(model, dataloader, confidence))

    # Write to tsv file
    with open(file_path, "wt") as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow("")
        for id, pred in list_of_id_pred_tuples:
            id, pred = str(id), str(pred)
            tsv_writer.writerow([id, pred, run_id])

    # Status print
    print(f"Created submission file at: {file_path}")