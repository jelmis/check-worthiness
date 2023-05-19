import torch
from torch import nn
from torch import optim


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
