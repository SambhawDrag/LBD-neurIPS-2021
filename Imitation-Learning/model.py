import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any
import torch.nn.functional as F

import os
import wandb
#import torch


class Net(nn.Module):
    def __init__(self, in_dims, out_dims, layer_1, layer_2, layer_3, layer_4, layer_5):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dims, layer_1),
            nn.ReLU(),
            nn.Linear(layer_1, layer_2),
            nn.ReLU(),
            nn.Linear(layer_2, layer_3),
            nn.ReLU(),
            nn.Linear(layer_3, layer_4),
            nn.ReLU(),
            nn.Linear(layer_4, layer_5),
            nn.ReLU(),
            nn.Linear(layer_5, out_dims)
        )

    def forward(self, x):
        return self.layers(x)


class LitMLP(pl.LightningModule):
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        layer_1: int,
        layer_2: int,
        layer_3: int,
        layer_4: int,
        layer_5: int,
        lr: float = 1e-3,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.model = Net(in_dims, out_dims, layer_1, layer_2, layer_3, layer_4, layer_5)
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
