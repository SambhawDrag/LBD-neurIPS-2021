import pandas as pd
import glob
import regex as re
import json
import os
from tqdm import tqdm
from typing import List
import numpy as np


class Traj2JSON():
    """
        Converts Trajectory data to JSON format
        Output format:
            List[MetaInfo[ID, System, States, Actions],Dicts[state,action]]
        First object is the Meta information dict followed by 200*20  dictionaries {state,action}'
    """

    def __init__(self, system, root_dir) -> None:

        self.system: str = system
        self.root_dir = root_dir
        self.traj_files: List[str] = glob.glob(
            os.path.join(root_dir, f'*{system}*.csv'))
        # use last trajectory-file for testing
        self.test_traj_files = [self.traj_files[-1]]
        self.traj_files = self.traj_files[:-1]

        self.states = None
        self.state_norms = None

        self.actions = None
        self.action_norms = None

        self.targets = ['X', 'Y']
        self.traj: list = []
        self.test_traj: list = []
        self.metainfo = None
        self.test_metainfo = None

    def read(self):
        # Read all training trajectory files
        for i, traj_file in enumerate(self.traj_files):
            df = pd.read_csv(traj_file, delimiter=',')

            if not self.states:
                self.states = [x for x in df.columns if self.isState(x)]
                self.state_norms = np.zeros(
                    (len(self.states), 2))

            if not self.actions:
                self.actions = [x for x in df.columns if self.isAction(x)]
                self.action_norms = np.zeros(
                    (len(self.actions), 2))

            if not self.metainfo:
                self.metainfo = {'System': self.system,
                                 'States': self.states + ['Xt', 'Yt'],
                                 'Actions': self.actions.copy()}
                self.traj.append(self.metainfo)

            for x in range(len(df)-1):
                state = df.iloc[x][self.states].to_list()
                action = df.iloc[x][self.actions].to_list()
                state.extend(df.iloc[x+1][self.targets].to_list())

                self.traj.append({
                    "state": state,
                    "action": action
                })

            N = len(df) - 1
            # iterative mean and std_dev calculation to prevent over/underflow
            self.state_norms[:,
                             0] += (np.sum(df[self.states].to_numpy()[:-1] / N, axis=0) - self.state_norms[:, 0]) / (i+1)
            self.state_norms[:,
                             1] += (np.sum(np.square(df[self.states].to_numpy()[:-1]) / N, axis=0) - self.state_norms[:, 1]) / (i+1)

            self.action_norms[:,
                              0] += (np.sum(df[self.actions].to_numpy()[:-1] / N, axis=0) - self.action_norms[:, 0]) / (i+1)
            self.action_norms[:,
                              1] += (np.sum(np.square(df[self.actions].to_numpy()[:-1]) / N, axis=0) - self.action_norms[:, 1]) / (i+1)

        self.state_norms[:, 1] = np.sqrt(self.state_norms[:, 1] -
                                         np.square(self.state_norms[:, 0]))
        self.action_norms[:, 1] = np.sqrt(self.action_norms[:, 1] -
                                          np.square(self.action_norms[:, 0]))
        # append length in meta info
        self.traj[0]['Length'] = len(self.traj)-1  # -1 for meta info

        # Add mean and std_dev to state-meta info
        for i, state in enumerate(self.traj[0]['States']):
            if state in ['Xt', 'Yt']:
                # same mean and std_dev as X,Y (may change later as Xt,Yt are 1 timestep ahead)
                index = next(x for x in range(len(self.states))
                             if self.states[x] == state.strip('t'))
                self.traj[0]['States'][i] = {
                    'Name': state,
                    'mean': self.state_norms[index, 0],
                    'std': self.state_norms[index, 1]
                }
            else:
                self.traj[0]['States'][i] = {
                    'Name': state,
                    'mean': self.state_norms[i, 0],
                    'std': self.state_norms[i, 1]
                }
        # Add mean and std_dev to action-meta info
        for i, action in enumerate(self.traj[0]['Actions']):
            self.traj[0]['Actions'][i] = {
                'Name': action,
                'mean': self.action_norms[i, 0],
                'std': self.action_norms[i, 1]
            }

        # Read test_file and store in test_trajectory
        for traj_file in self.test_traj_files:

            df = pd.read_csv(traj_file, delimiter=',')

            if not self.test_metainfo:
                self.test_metainfo = {'System': self.system,
                                      'States': self.states + ['Xt', 'Yt'],
                                      'Actions': self.actions.copy()}
                self.test_traj.append(self.test_metainfo)

            for i in range(len(df)-1):
                state = df.iloc[i][self.states].to_list()
                state.extend(df.iloc[i+1][self.targets].to_list())
                action = df.iloc[i][self.actions].to_list()

                self.test_traj.append({
                    "state": state,
                    "action": action
                })

        self.test_traj[0]['Length'] = len(self.test_traj)-1  # -1 for meta info

    def isState(self, x):
        return re.match(r'd?X\d?|d?Y\d?', x)

    def isAction(self, x):
        return re.match(r'U\d', x)

    def write(self, filename, test_filename):
        with open(filename, 'w') as f:
            json.dump(self.traj, f)

        with open(test_filename, 'w') as f:
            json.dump(self.test_traj, f)


def main(systems):
    for system in tqdm(systems, desc='Converting to JSON ...'):
        traj = Traj2JSON(system, '../traj_data')
        traj.read()
        traj.write(f'dataset/train/{system}.json',
                   f'dataset/test/{system}.json')


if __name__ == "__main__":
    system = [
        "great-piquant-bumblebee",
        "great-bipedal-bumblebee",
        "great-impartial-bumblebee",
        "great-proficient-bumblebee",
        "lush-piquant-bumblebee",
        "lush-bipedal-bumblebee",
        "lush-impartial-bumblebee",
        "lush-proficient-bumblebee",
        "great-devious-beetle",
        "great-vivacious-beetle",
        "great-mauve-beetle",
        "great-wine-beetle",
        "rebel-devious-beetle",
        "rebel-vivacious-beetle",
        "rebel-mauve-beetle",
        "rebel-wine-beetle",
        "talented-ruddy-butterfly",
        "talented-steel-butterfly",
        "talented-zippy-butterfly",
        "talented-antique-butterfly",
        "thoughtful-ruddy-butterfly",
        "thoughtful-steel-butterfly",
        "thoughtful-zippy-butterfly",
        "thoughtful-antique-butterfly"
    ]
    main(system)
