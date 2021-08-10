"""
Controller template.
"""

import numpy as np
from metadata import model_dict
from model import Net
import torch


class controller(object):

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
        self.model = Net(model_dict[self.system]['dim'][0],model_dict[self.system]['dim'][1])
        self.model_path = 'models/' + model_dict[self.system]['path']
        self.model.load_state_dict(torch.load(self.model_path))

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
        inp = np.concatenate((state, target), dtype=np.float32)
        return self.model(torch.Tensor(inp)).numpy()
 