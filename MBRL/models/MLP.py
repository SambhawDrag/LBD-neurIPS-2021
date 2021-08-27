import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dims, 16),
            nn.ReLU(),
            nn.Linear(16, out_dims)
        )

    def forward(self, x):
        return self.layers(x)
