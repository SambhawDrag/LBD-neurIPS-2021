import random
import numpy as np
import torch
from typing import Tuple
from torch.utils.data import DataLoader, SubsetRandomSampler
from .Traj2Dataset import TrajDataset, DatasetTransform, test_dataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloaders(
    system: str,
    root_dir: str,
    shuffle: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
    *args,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, Tuple[int, int]]:
    """
    Creates dataloaders for training, validation, and test set.
    Args:

        system: Name of the system.
        root_dir: path to dataset on local machine or cluster
        shuffle: whether to shuffle the dataset
        batch_size: batch size
        num_workers: number of workers for each dataloader
        val_split: fraction of the dataset to use for validation

    Returns:
        train_loader: training dataloader
        val_loader: validation dataloader
        (inp_dim, out_dim): input and output dimensions
    """

    state_norms, action_norms = TrajDataset.norms(system)

    inp_norms = np.concatenate((state_norms, action_norms), axis=0)
    target_norms = state_norms

    inp_dim, target_dim = inp_norms.shape[0], target_norms.shape[0]

    mean = inp_norms[:, 0]
    std = inp_norms[:, 1]
    transform = DatasetTransform(mean, std)

    target_mean = target_norms[:, 0]
    target_std = target_norms[:, 1]
    target_transform = DatasetTransform(target_mean, target_std)

    train_dataset = TrajDataset(
        system,
        root_dir,
        transform=transform,
        target_transform=target_transform,
        train=True,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )

    valid_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
    )

    return train_loader, valid_loader, (inp_dim, target_dim)


def create_test_loader(
    system: str,
    root_dir: str,
    shuffle: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:

    state_norms, action_norms = TrajDataset.norms(system)

    inp_norms = np.concatenate((state_norms, action_norms), axis=0)
    target_norms = state_norms

    mean = inp_norms[:, 0]
    std = inp_norms[:, 1]
    transform = DatasetTransform(mean, std)

    target_mean = target_norms[:, 0]
    target_std = target_norms[:, 1]
    target_transform = DatasetTransform(target_mean, target_std)

    test_dataset = TrajDataset(
        system,
        root_dir,
        transform=transform,
        target_transform=target_transform,
        train=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return test_loader
