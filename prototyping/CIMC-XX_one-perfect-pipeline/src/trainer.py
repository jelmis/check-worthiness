import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from itertools import product
import random
from torch.utils.data import DataLoader


class Trainer():
    """
    Each instance of this class has one model architecture and dataset.
    It can perform hyperparameter search, and deep training on provided hyperparameters.
    """

    def __init__(self, model, model_init_params, dataset_dict):
        """
        :param dataset_dict:  
        """
        self.model = model
        self.model_init_params = model_init_params
        self.dataset_dict = dataset_dict
        self.trained_models = []

    def hyperparameter_search(self, **kwargs):
        """
        Determines the best hyperparameter combination for the model to further train the model
        :param random: if None, grid search will performed. Otherwise, pass a float between 0 to 1
        to determine number of hyperparameter combinations to be randomly selected out of all
        combinations.
        """

        loss_fn = kwargs.get("loss_fn", nn.BCELoss())
        optimizer_type = kwargs.get("optimizer", torch.optim.Adam)
        num_workers = kwargs.get("num_workers", 8)
        num_epochs = kwargs.get("num_epochs", 50)
        device = kwargs.get("device", "cpu")
        early_quit_thresh = kwargs.get("early_quit_thresh", 10)
        balance_classes = kwargs.get("balance_classes", False)
        hyparam_combinations = self._get_hyparam_combinations(kwargs)

        # Perform hyperparameter search
        # TODO: Allow hyperparameter search on model architecture
        for lr, batch_size, shuffle in hyparam_combinations:
            dataloaders = {}
            if not balance_classes:
                for split_name, split in self.dataset_dict.items():
                    dataloaders[split_name] = DataLoader(split, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            else:
                pass  # TODO: Add balanced dataloader support
            model = self.model(self.model_init_params)
            model.to(device)
            tb = SummaryWriter()
            optimizer = optimizer_type(model.parameters(), lr=lr)
            self.train_model(tb, num_epochs, model, device, dataloaders["train"], dataloaders["val"], optimizer, loss_fn, early_quit_thresh)
            self.trained_models.append(model)
            # self._evaluate_model(model)  # log
            # self._name_the_model()
        return

    def _get_hyparam_combinations(self, hyperparams):

        random = hyperparams.get("random", None)
        learning_rates = hyperparams.get("learning_rates", [1e-5])
        batch_size_list = hyperparams.get("batch_sizes", [16])
        shuffle_list = hyperparams.get("shuffle_list", [True])
        hyperparams_to_search = product(learning_rates, batch_size_list, shuffle_list)
        if random is None:
            # Perform grid search
            return hyperparams_to_search
        else:
            # TODO: allow random search
            # random.sample(list, n)
            return hyperparams_to_search

    def train_model(self, tb, num_epochs, model, device, train_dataloader, val_dataloader, optimizer, loss_fn, early_quit_thresh=None):
        """
        Takes in a model, trains it on the specified device for num_epochs, 
        until early quit conditions are reached.
        """

        accum_train_losses, accum_val_losses = [], []
        for epoch in tqdm(range(num_epochs)):

            epoch_train_loss, epoch_val_loss = self._train_single_epoch(model, device, train_dataloader, val_dataloader, optimizer, loss_fn)
            accum_train_losses.append(epoch_train_loss)
            accum_val_losses.append(epoch_val_loss)

            # Log to tensorboard
            tb.add_scalar("Mean Loss/train", epoch_train_loss, epoch)
            tb.add_scalar("Mean Loss/val", epoch_val_loss, epoch)
            precision, recall = self._evaluate_model(model, val_dataloader)
            tb.add_scalar("Precision for Positive Class on valset:", precision)
            tb.add_sacalar("Recall Positive Class on valset:", recall)

            if early_quit_thresh:
                if early_quit_thresh > epoch+1:
                    idx = early_quit_thresh+1
                    last_epoch_val_loss = accum_val_losses[-1]
                    recent_mean_loss = np.mean(accum_val_losses[-idx:-1])
                    if recent_mean_loss < last_epoch_val_loss:
                        tb.close()
                        break
        return

    def _train_single_epoch(model, device, train_dataloader, val_dataloader, optimizer, loss_fn):
        """
        Conducts the training loop and returns a list of all best losses.
        :param train_dataloader:
        :param val_dataloader:
        :param optimizer:
        :param network:
        :param loss_fn:
        :param num_epochs:
        :return: list of batch losses
        """

        train_losses = []
        for x, y in train_dataloader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            scores = model(x)
            loss = loss_fn(scores, y.unsqueeze(-1))
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        val_losses = []
        with torch.no_grad():
            for x, y in val_dataloader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out, y.unsqueeze(-1))
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        return avg_train_loss, avg_val_loss

    def _evaluate_model(model, dataloader, confidence=0.5):

        pred = []
        gt = []

        with torch.no_grad():
            for x, y in dataloader:
                scores = model(x)
                y_pred = np.where(scores < confidence, 0, 1).tolist()
                pred.extend(y_pred)
                gt.extend(y.tolist())
        pred = np.array(pred)
        gt = np.array(gt)
        TP = sum(gt[np.where(pred == 1)])
        FP = sum(pred[np.where(gt == 0)])
        FN = -sum(pred[np.where(gt == 1)] - 1)

        return TP/(TP+FP), TP/(TP+FN)
