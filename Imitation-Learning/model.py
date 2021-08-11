import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, out_dims)
        )

    def forward(self, x):
        return self.layers(x)


class LitMLP(pl.LightningModule):
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        lr: float = 1e-3,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.model = Net(in_dims, out_dims)
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
