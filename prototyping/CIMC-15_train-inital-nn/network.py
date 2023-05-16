import torch
from torch import nn
from torch import optim


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        predictions = torch.sigmoid(self.layer_1(x))
        return predictions
