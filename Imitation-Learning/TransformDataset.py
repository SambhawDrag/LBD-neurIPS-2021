import pandas as pd
import glob
import regex as re
import json
import os
from typing import List


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

        self.states = None
        self.actions = None
        self.targets = ['X', 'Y']
        self.traj: list = []
        self.metainfo = None

        ...

    def read(self):
        for traj_file in self.traj_files:
            df = pd.read_csv(traj_file, delimiter=',')

            if not self.states:
                self.states = [x for x in df.columns if self.isState(x)]
            if not self.actions:
                self.actions = [x for x in df.columns if self.isAction(x)]

            if not self.metainfo:
                self.metainfo = {'System': self.system,
                                 'States': self.states + ['Xt', 'Yt'],
                                 'Actions': self.actions}
                self.traj.append(self.metainfo)

            for i in range(len(df)-1):
                state = df.iloc[i][self.states].to_list()
                action = df.iloc[i][self.actions].to_list()
                state.extend(df.iloc[i+1][self.targets].to_list())

                self.traj.append({
                    "state": state,
                    "action": action
                })

        # append length in meta info
        self.traj[0]['Length'] = len(self.traj)-1  # -1 for meta info

    def isState(self, x):
        return re.match(r'd?X\d?|d?Y\d?', x)

    def isAction(self, x):
        return re.match(r'U\d', x)

    def write(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.traj, f)


def main(systems):
    for system in systems:
        traj = Traj2JSON(system, '../traj_data')
        traj.read()
        traj.write(f'dataset/{system}.json')


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
