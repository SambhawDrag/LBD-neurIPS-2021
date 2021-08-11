from typing import Tuple
import numpy as np
from Traj2Dataset import TrajDataset, DatasetTransform
from torch.utils.data import DataLoader, SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import argparse
from model import LitMLP


def create_dataloaders(
    args: argparse.Namespace,
    shuffle=True,
    **kwargs
) -> Tuple[DataLoader, DataLoader, Tuple[int, int]]:
    """
        Returns a tuple with first two as train and validation dataloader
        and the third element as a tuple of state and action dimensions
    """
    train_dataset = TrajDataset(args.system, args.root_dir, train=True)

    state_dim = len(train_dataset.states)
    action_dim = len(train_dataset.actions)

    mean = [x['mean'] for x in train_dataset.states]  # mean
    std = [x['std'] for x in train_dataset.states]   # std_dev
    transform = DatasetTransform(mean, std)
    target_mean = [x['mean'] for x in train_dataset.actions]  # mean
    target_std = [x['std'] for x in train_dataset.actions]   # std_dev
    target_transform = DatasetTransform(target_mean, target_std)

    train_dataset.tranform = transform
    train_dataset.target_transform = target_transform

    test_dataset = TrajDataset(args.system, args.root_dir, train=False,
                               transform=transform,
                               target_transform=target_transform)
    indices = np.arange(len(train_dataset))

    if shuffle is True:
        np.random.shuffle(indices)

    split = int(args.val_split * len(train_dataset))
    train_indices = indices[split:]
    valid_indices = indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size, num_workers=args.num_workers,
                                  sampler=train_sampler)
    valid_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size, num_workers=args.num_workers,
                                  sampler=valid_sampler)
    return train_dataloader, valid_dataloader, (state_dim, action_dim)


def main(args: argparse.Namespace):
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.logdir}/model", monitor='val_loss',
        filename=f"{args.system}" + "-{epoch:02d}",
        period=1,
        save_last=True)

    callbacks = [checkpoint_callback]
    tb_logger = loggers.TensorBoardLogger(args.logdir)

    train_loader, val_loader, dims = create_dataloaders(args)

    model = LitMLP(*dims, args.lr)

    trainer = pl.Trainer(
        gpus=args.gpus, logger=tb_logger,
        callbacks=callbacks,
        progress_bar_refresh_rate=30,
        max_epochs=args.epochs,)

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trajectory Inference with MLP')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--root-dir', type=str, default='./dataset')
    parser.add_argument('--system', type=str, default='great-wine-beetle')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--val-split', type=str, default=0.1)
    parser.add_argument('--gpus', type=int, default=1)

    args = parser.parse_args()
    main(args)
