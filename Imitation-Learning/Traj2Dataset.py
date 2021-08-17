from typing import Any, Tuple, List
import torch.utils.data as data
import numpy as np
import json
import os


class TrajDataset(data.Dataset):

    curr_dir = os.path.dirname(__file__)
    metadata = json.load(
        open(os.path.join(curr_dir, 'dataset', 'metadata.json')))
    """
        Custom Dataset class for loading trajectory state-action for Behavior Cloning
    """

    def __init__(
        self,
        system: str,
        root_dir: str,
        train: bool = True,
        transform=None,
        target_transform=None
    ):
        """

        Args:
            system : Name of the system (e.g. "great-bipedal-bumblee")
            root_dir : Root directory of the dataset (json files)
            transform : Input transforms. Defaults to None.
            target_transform : Target transforms. Defaults to None.
        """

        self.system: str = system
        self.root_dir: str = root_dir
        self._transform = transform
        self._target_transform = target_transform
        self.train = train  # train or test set

        if self.train:
            self.data_file = os.path.join(
                self.root_dir, "train", f"{system}.json")
        else:
            self.data_file = os.path.join(
                self.root_dir, "test", f"{system}.json")

        with open(self.data_file) as f:
            self.data = json.load(f)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:

        states = np.array(self.data[idx+1]['state'], dtype=np.float32)
        actions = np.array(self.data[idx+1]['action'], dtype=np.float32)

        if self.transform:
            states = self.transform(states)

        if self.target_transform:
            actions = self.target_transform(actions)

        return states, actions

    def __len__(self) -> int:
        return self.data[0]['Length']

    @property
    def states(self) -> List[str]:
        return self.data[0]['States']

    @property
    def actions(self) -> List[str]:
        return self.data[0]['Actions']

    @property
    def transform(self) -> None:
        return self._transform

    @property
    def target_transform(self) -> None:
        return self._target_transform

    @transform.setter
    def transform(self, transform: Any) -> None:
        self._transform = transform

    @target_transform.setter
    def target_transform(self, transform: Any) -> None:
        self._target_transform = transform

    @staticmethod
    def norms(system: str) -> Tuple[np.ndarray, np.ndarray]:
        state_norm = np.array(
            TrajDataset.metadata[system]['state_norms'], dtype=np.float32)
        action_norm = np.array(
            TrajDataset.metadata[system]['action_norms'], dtype=np.float32)

        return state_norm, action_norm


class DatasetTransform():
    def __init__(self, mean, std_dev) -> None:
        if not isinstance(mean, np.ndarray):
            mean = np.array(mean, dtype=np.float32)
        if not isinstance(std_dev, np.ndarray):
            std_dev = np.array(std_dev, dtype=np.float32)

        self.mean = mean
        self.std_dev = std_dev

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std_dev + 1e-9)  # for 0.00


def test_dataset() -> None:

    root_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    system = 'great-piquant-bumblebee'
    train_dataset = TrajDataset(system, root_dir, train=True)

    state_norms, action_norms = TrajDataset.norms(system)

    mean = state_norms[:, 0]
    std_dev = state_norms[:, 1]
    transform = DatasetTransform(mean, std_dev)

    target_mean = action_norms[:, 0]
    target_std_dev = action_norms[:, 1]
    target_transform = DatasetTransform(target_mean, target_std_dev)

    train_dataset.tranform = transform
    train_dataset.target_transform = target_transform

    test_dataset = TrajDataset(system, root_dir, train=False,
                               transform=transform,
                               target_transform=target_transform)

    print('\nTraining Dataset Length: {}'.format(len(train_dataset)))
    from torch.utils.data import dataloader
    train_dataloader = dataloader.DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=0)

    states, actions = next(iter(train_dataloader))
    print(f'sample state: {states}')
    print(f'sample action: {actions}')

    print('\nTest Dataset Length: {}'.format(len(test_dataset)))
    test_dataloader = dataloader.DataLoader(
        test_dataset, batch_size=8, shuffle=True, num_workers=0)

    states, actions = next(iter(test_dataloader))
    print(f'sample state: {states}')
    print(f'sample action: {actions}')


if __name__ == '__main__':
    test_dataset()
