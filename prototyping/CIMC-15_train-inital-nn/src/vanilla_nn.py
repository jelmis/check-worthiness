import torch
from torch import nn
from torch import optim
from torch.nn import ReLU


class VanillaNN(nn.Module):
    def __init__(self, layer_sizes):
        super(VanillaNN, self).__init__()

        self.hidden_layers = []
        for layer_in, layer_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.hidden_layers.append(nn.Linear(layer_in, layer_out))
            self.hidden_layers.append(ReLU())
        # Remove non-linearity in the final layer to replace it with softmax in forward pass
        del self.hidden_layers[-1]

        self.sequence = nn.Sequential(
            *self.hidden_layers
        )

    def forward(self, x):
        x = self.sequence(x)
        out = torch.sigmoid(x)

        return out


"""
class VanillaNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VanillaNN, self).__init__()
        self.linear_one = nn.Linear(input_dim, hidden_dim)
        self.linear_two = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_one_out = torch.nn.functional.relu(self.linear_one(x))
        h_two_out = self.linear_two(h_one_out)
        out = torch.sigmoid(h_two_out)

        return out
"""
