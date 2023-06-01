import numpy as np
import matplotlib.pyplot as plt

def training_loop(train_dataloader, optimizer, network, loss_fn, num_epochs):
    """
    Conducts the training loop and returns a list of all best losses.
    :param train_dataloader:
    :param optimizer:
    :param network:
    :param loss_fn:
    :param num_epochs:
    :return: list of batch losses
    """
    losses = []
    for epoch in range(num_epochs):
        for features, labels in train_dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            out = network(features)

            # Compute loss
            loss = loss_fn(out, labels.unsqueeze(-1))
            losses.append(loss.item())

            # Backward pass
            loss.backward()

            # Optimize weights
            optimizer.step()

    return losses


def plot_losses(num_epochs, losses, fe_method):
    """
    Plots the batch-wise losses over
    all epochs. Uses fe_method as labeling.
    :param num_epochs:
    :param losses:
    :param fe_method:
    :return:
    """
    step = np.linspace(0, num_epochs, len(losses))

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(step, np.array(losses))
    plt.title(f"Step-wise Loss ({fe_method})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
