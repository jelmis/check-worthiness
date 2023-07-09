import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from itertools import product
import random
from torch.utils.data import DataLoader
from datetime import date, datetime


class Trainer():
    """
    Each instance of this class has one model architecture and dataset.
    It can perform hyperparameter search, and deep training on provided hyperparameters.
    """

    def __init__(self, model, dataset_method, dataset_dict):
        """
        :param dataset_dict:  
        """
        self.model = model
        self.dataset_method = dataset_method
        self.dataset_dict = dataset_dict
        self.input_dim = len(dataset_dict["train"][0][0])
        self.output_dim = 1
        self.trained_models = {}

    def hyperparameter_search(self, **kwargs):
        """
        Determines the best hyperparameter combination for the model to further train the model
        :param random: if None, grid search will performed. Otherwise, pass a float between 0 to 1
        to determine number of hyperparameter combinations to be randomly selected out of all
        combinations.
        """

        self.loss_fn = kwargs.get("loss_fn", nn.BCELoss())
        self.optimizer_type = kwargs.get("optimizer", torch.optim.Adam)
        self.num_workers = kwargs.get("num_workers", 0)
        self.num_epochs = kwargs.get("num_epochs", 50)
        self.device = kwargs.get("device", "cpu")
        self.early_quit_thresh = kwargs.get("early_quit_thresh", 10)
        self.balance_classes = kwargs.get("balance_classes", False)
        self.model_init_params = [self.input_dim] + kwargs.get("model_init_params", [64, 32]) + [self.output_dim]
        self.hyparam_combinations = self._get_hyparam_combinations(kwargs)

        # TODO: Add scheduler for learning rate decay

        # Perform hyperparameter search
        # TODO: Allow hyperparameter search on model architecture
        for lr, batch_size, shuffle in self.hyparam_combinations:
            self.lr, self.batch_size, self.shuffle = lr, batch_size, shuffle
            dataloaders = {}
            if not self.balance_classes:
                for split_name, split in self.dataset_dict.items():
                    dataloaders[split_name] = DataLoader(split, batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers)
            else:
                pass  # TODO: Add balanced dataloader support
            model = self.model(self.model_init_params)
            model.to(self.device)
            model_metadata, model_metadata_w_time = self._get_model_metadata(lr, batch_size, shuffle)
            tb = SummaryWriter(comment=model_metadata)
            optimizer = self.optimizer_type(model.parameters(), lr=lr)
            f1 = self.train_model(tb, self.num_epochs, model, self.device, dataloaders["train"], dataloaders["dev"], optimizer, self.loss_fn, early_quit_thresh=self.early_quit_thresh)
            self.trained_models[model_metadata_w_time] = model, f1
        return

    def _get_hyparam_combinations(self, hyperparams):

        random = hyperparams.get("random", None)
        learning_rates = hyperparams.get("learning_rates", [1e-5])
        batch_size_list = hyperparams.get("batch_sizes", [16])
        shuffle_list = hyperparams.get("shuffle_list", [True])
        hyperparam_combinations = product(learning_rates, batch_size_list, shuffle_list)
        if random is None:
            # Perform grid search
            return hyperparam_combinations
        else:
            # TODO: allow random search
            # random.sample(list, n)
            return hyperparam_combinations

    def _get_model_metadata(self, lr, batch_size, shuffle):

        time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        model_architecture = "x".join(str(size) for size in self.model_init_params[1:-1])
        shuffle = "shuffled" if shuffle else "unshuffled"
        model_metadata = "_".join([self.dataset_method, model_architecture, "lr_"+str(lr), "batch-size_"+str(batch_size), shuffle])
        model_metadata_w_time = time + "_" + model_metadata
        return model_metadata, model_metadata_w_time

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
            precision, recall, f1 = self._evaluate_model(model, val_dataloader)
            tb.add_scalar("Scores on Valset Positive Samples/precision", precision, epoch)
            tb.add_scalar("Scores on Valset Positive Samples/recall", recall, epoch)
            tb.add_scalar("Scores on Valset Positive Samples/f1", f1, epoch)

            if early_quit_thresh:
                if early_quit_thresh < epoch+1:
                    idx = early_quit_thresh+1
                    last_epoch_val_loss = accum_val_losses[-1]
                    recent_mean_loss = np.mean(accum_val_losses[-idx:-1])
                    if recent_mean_loss < last_epoch_val_loss:
                        print("Early quitting at epoch:", epoch)
                        break

        lr = self._get_lr(optimizer)
        tb.add_hparams(
            {
                "model_architecture": "x".join(str(size) for size in self.model_init_params[1:-1]),
                "dataset_method": self.dataset_method,
                "lr": lr, "bsize": self.batch_size, "shuffle": self.shuffle
            },
            {
                "f1_score": f1,
                "loss": epoch_val_loss,
            },
        )
        tb.close()
        return f1

    def _train_single_epoch(self, model, device, train_dataloader, val_dataloader, optimizer, loss_fn):
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

    def _evaluate_model(self, model, dataloader, confidence=0.5):

        pred = []
        gt = []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                scores = model(x).cpu()
                y_pred = np.where(scores < confidence, 0, 1).reshape(-1).tolist()
                pred.extend(y_pred)
                gt.extend(y.tolist())
        pred = np.array(pred)
        gt = np.array(gt)
        TP = sum(gt[np.where(pred == 1)])
        FP = sum(pred[np.where(gt == 0)])
        FN = -sum(pred[np.where(gt == 1)] - 1)
        precision = TP/(TP+FP) if (TP+FP) != 0 else 0
        recall = TP/(TP+FN) if (TP+FN) != 0 else 0
        f1 = 2*(precision*recall)/(precision+recall) if (precision+recall) != 0 else 0
        return precision, recall, f1

    def _get_lr(self, optimizer):
        """
        If lr decay is used, the lr will differ at the end of training.
        Get the lr here.
        """
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def save_trained_models(self, dir, to_be_saved_models=None):
        """
        Saves trained models in a directory.
        By default, all models trained by the trainer are saved.
        :param to_be_saved_models: List of model names in the form model_metadata_w_time
        :param dir: Directory where the models are saved
        :return:
        """
        # Save all models
        if to_be_saved_models == None:
            for model_name in self.trained_models.keys():
                model, f1 = self.trained_models[model_name]
                path = f"{dir}/{model_name}_f1_{f1:.2f}.pt"
                torch.save(model.state_dict(), path)

        # Save specified models
        else:
            for model_name in to_be_saved_models:
                model, f1 = self.trained_models[model_name]
                path = f"{dir}/{model_name}_f1_{f1:.2f}.pt"
                torch.save(model.state_dict(), path)
