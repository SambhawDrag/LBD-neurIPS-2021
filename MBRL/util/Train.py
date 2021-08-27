import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List


@torch.enable_grad()
def train(
    model: nn.Module,
    train_loader: DataLoader,
    epoch,
    args,
) -> List[float]:
    """
        Train the model.
        Arguments:
            model: The model to train.
            train_loader: The training data loader.
            args: The arguments.
    """

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    device = "cuda:0" if args.cuda else "cpu"
    losses = []

    for batch_idx, batch in enumerate(train_loader):

        optimizer.zero_grad()
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()

        optimizer.step()
        losses.append(loss.item())

        if args.fast_dev_run:
            break

    return losses


@torch.no_grad()
def test(
    model: nn.Module,
    test_loader: DataLoader,
    args,
) -> float:
    """
        Test the model.
        Arguments:
            model: The model to test.
            test_loader: The test data loader.
            args: The arguments.
        Returns:
            Average loss.
    """

    model.eval()
    device = "cuda:0" if args.cuda else "cpu"
    loss_fn = nn.MSELoss()
    losses = []

    for _, batch in enumerate(test_loader):

        x, y = batch
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        losses.append(loss.item())

        if args.fast_dev_run:
            break

    return sum(losses) / len(losses)
