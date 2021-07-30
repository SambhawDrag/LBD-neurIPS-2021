from typing import Any,Tuple
import torch.utils.data as data
import numpy as np
import json
import os


class TrajDataset(data.Dataset):

    def __init__(
        self,
        system: str,
        root_dir: str,
        transform=None,
        target_transform=None
    ):
        self.system: str = system
        self.root_dir: str = root_dir
        self.transform = transform
        self.target_transform = target_transform

        with open(os.path.join(root_dir, f'{system}.json')) as f:
            self.data = json.load(f)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:

        states = np.array(self.data[idx+1]['state'])
        actions = np.array(self.data[idx+1]['action'])

        if self.transform:
            states = self.transform(states)

        if self.target_transform:
            actions = self.target_transform(actions)

        return states, actions

    def __len__(self) -> int:
        return self.data[0]['Length']


def test_dataset() -> None:

    root_dir = 'dataset'
    system = 'great-piquant-bumblebee'
    dataset = TrajDataset(system, root_dir)

    states, actions = dataset[0]
    print('Dataset Length: {}'.format(len(dataset)))

    from torch.utils.data import dataloader
    dataloader = dataloader.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=0)

    states, actions = next(iter(dataloader))
    print(f'Random Sample: State: {states}, Action: {actions}')


if __name__ == '__main__':
    test_dataset()
