"""
This program can be launched directly.
"""

import math
import os
import sys
from pathlib import Path
from typing import List, Type, Tuple

import arcade
import numpy as np
import matplotlib.pyplot as plt

# Insert the parent directory of the current file's directory into sys.path.
# This allows Python to locate modules that are one level above the current
# script, in this case spg_overlay.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spg_overlay.utils.path import Path
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.utils import clamp
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.utils.utils import normalize_angle
from spg_overlay.utils.misc_data import MiscData


class MyDronePid(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.counter = 0
        self.x = []
        self.d2_x = []
        self.command_history = []
        self.max_counter = 40

    def define_message_for_all(self):
        pass

    def control(self):
        self.x.append(self.true_position()[0])

        forward = 1
        command = {"forward": forward,
                   "rotation": 0}

        if self.counter == self.max_counter:
            self.plot_results()

        if self.counter >= 3:
            d_x = self.x[-1]-self.x[-2]
            d_x_prec = self.x[-2]-self.x[-3]
        
            d2_x = d_x - d_x_prec

            self.d2_x.append(d2_x)
            self.command_history.append(forward)

        self.counter += 1

        return command
    
    def plot_results(self):
            import matplotlib.pyplot as plt

            time_steps = np.arange(len(self.d2_x))
            position_data = np.array(self.d2_x)
            command_history = np.array(self.command_history)

            plt.plot(time_steps, position_data, label="d2x drone")

            plt.legend()
            plt.show()

class MyMap(MapAbstract):
    def __init__(self):
        super().__init__()

        # PARAMETERS MAP
        self._size_area = (800, 800)

        # POSITIONS OF THE DRONES
        self._number_drones = 1
        self._drones_pos = []
        for i in range(self._number_drones):
            pos = ((-100, -100), 0)
            self._drones_pos.append(pos)

        self._drones: List[DroneAbstract] = []

    def construct_playground(self, drone_type: Type[DroneAbstract]):
        playground = ClosedPlayground(size=self._size_area)

        # POSITIONS OF THE DRONES
        misc_data = MiscData(size_area=self._size_area,
                             number_drones=self._number_drones,
                             max_timestep_limit=self._max_timestep_limit,
                             max_walltime_limit=self._max_walltime_limit)
        for i in range(self._number_drones):
            drone = drone_type(identifier=i, misc_data=misc_data)
            self._drones.append(drone)
            playground.add(drone, self._drones_pos[i])

        return playground


def main():
    my_map = MyMap()
    my_playground = my_map.construct_playground(drone_type=MyDronePid)

    gui = GuiSR(playground=my_playground,
                the_map=my_map,
                use_keyboard=False,
                use_mouse_measure=True,
                enable_visu_noises=False,
                )

    gui.run()


if __name__ == '__main__':
    main()
