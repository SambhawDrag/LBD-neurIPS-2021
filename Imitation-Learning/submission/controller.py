"""
Controller template.
"""

import numpy as np
from model import LitMLP as Net
import torch
import json
import os


class controller(object):

    root_dir = os.path.dirname(__file__)  # root_dir for loading metadata
    metadata = json.load(
        open(os.path.join(root_dir, 'metadata.json')))

    def __init__(self, system, d_control):
        """
        Entry point, called once when starting the controller for a newly
        initialized system.

        Input:
            system - holds the identifying system name;
                     all evaluation systems have corresponding training data;
                     you may use this name to instantiate system specific
                     controllers

            d_control  - indicates the control input dimension
        """
        self.system = system
        self.d_control = d_control
        self.__metadata = controller.metadata[system]

        self.state_norms = (self.__metadata['state_norms'])
        self.action_norms = (self.__metadata['action_norms'])

        self.state_mean = np.array(
            [x[0] for x in self.state_norms], dtype=np.float32)
        self.state_std = np.array(
            [x[1] for x in self.state_norms], dtype=np.float32)

        self.action_mean = np.array(
            [x[0] for x in self.action_norms], dtype=np.float32)
        self.action_std = np.array(
            [x[1] for x in self.action_norms], dtype=np.float32)

        # self.model = Net(*self.__metadata['dim'])
        path = os.path.join(self.root_dir, "models", self.__metadata['path'])

        # self.model.load_state_dict(torch.load(path))
        self.model = Net.load_from_checkpoint(
            path, in_dims=self.__metadata['dim'][0], out_dims=self.__metadata['dim'][1])
        self.model.eval()

    def get_input(self, state, position, target):
        """
        This function is called at each time step and expects the next
        control input to apply to the system as return value.

        Input: (all column vectors, if default wrapcontroller.py is used)
            state - vector representing the current state of the system;
                    by convention the first two entries always correspond
                    to the end effectors X and Y coordinate;
                    the state variables are in the same order as in the
                    corresponding training data for the current system
                    with name self.system
            position - vector of length two representing the X and Y
                       coordinates of the current position
            target - vector of length two representing the X and Y
                     coordinates of the next steps target position
        """
        # placeholder that just returns a next control input of correct shape
        inp = np.concatenate((state, target), axis=0).astype(
            np.float32).flatten()
        inp = (inp - self.state_mean) / (self.state_std + 1e-6)

        out = self.model(torch.from_numpy(inp)).detach().numpy()
        out = (out * (self.action_std + 1e-6)) + self.action_mean
        return out


def test():
    ctrl = controller(system='great-piquant-bumblebee', d_control=2)
    ## input provided as a vector of shape (X,1)
    print(ctrl.get_input(
        np.random.randn(8,1),
        np.random.randn(2,1),
        np.random.randn(2,1)
    ))


if __name__ == "__main__":
    test()
