import numpy as np
import matplotlib.pyplot as plt
import torch


def training_loop(train_dataloader, val_dataloader, optimizer, network, loss_fn, num_epochs):
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
    tot_train_losses = []
    tot_val_losses = []

    for epoch in range(num_epochs):
        train_losses = []
        for features, labels in train_dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            out = network(features)

            # Compute loss
            loss = loss_fn(out, labels.unsqueeze(-1))
            train_losses.append(loss.item())

            # Backward pass
            loss.backward()

            # Optimize weights
            optimizer.step()
        # Validation
        val_losses = []
        with torch.no_grad():
            for val_features, val_labels in val_dataloader:
                val_out = network(val_features)
                val_loss = loss_fn(val_out, val_labels.unsqueeze(-1))
                val_losses.append(val_loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        tot_train_losses.append(avg_train_loss)
        tot_val_losses.append(avg_val_loss)

    return tot_train_losses, tot_val_losses


def plot_losses(num_epochs, train_losses, val_losses, fe_method):
    """
    Plots the batch-wise losses over
    all epochs. Uses fe_method as labeling.
    :param num_epochs:
    :param train_losses:
    :param val_losses:
    :param fe_method:
    :return:
    """
    steps = np.linspace(0, num_epochs, len(train_losses))

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(steps, np.array(train_losses), label='Training Loss')
    plt.plot(steps, np.array(val_losses), label='Validation Loss')
    plt.title(f"Step-wise Loss ({fe_method})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
