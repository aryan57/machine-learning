import os
import torch
from torch import nn


# No hidden layers
class MLP_0(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 7)
        )

    def forward(self, x):
        return self.layers(x)


# 1 hidden layer
class MLP_1(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 7),
        )

    def forward(self, x):
        return self.layers(x)


# 2 hidden layers
class MLP_2(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 7),
        )

    def forward(self, x):
        return self.layers(x)