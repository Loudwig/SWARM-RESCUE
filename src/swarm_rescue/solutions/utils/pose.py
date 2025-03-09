from typing import Tuple
import numpy as np
from spg_overlay.utils.utils import normalize_angle


class Position:
    def __init__(self):
        self.data = np.zeros((2, ))

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        return 'Position({})'.format(self.data)

    def set(self, x, y):
        self.data[0] = x
        self.data[1] = y

    @property
    def x(self):
        return self.data[0]

    @property
    def y(self):
        return self.data[1]


# class Pose:
#     def __init__(self, position=Position(), orientation=0.0):
#         if not isinstance(position, Position):
#             print("type position=", type(position))
#             raise TypeError("position must be an instance of Position")
#         self.position: Position = position
#         self.orientation: float = orientation


class Pose:
    def __init__(self, position=np.zeros(2, ), orientation=0.0,odometer=[0.0,0.0,0.0],previous_position=np.zeros(2, ),previous_orientation=0.0,taille :Tuple[float, float] = (0.0,0.0)):

        if not isinstance(position, np.ndarray):
            print("type position=", type(position))
            raise TypeError("position must be an instance of np.ndarray")
        
        if position is None or orientation is None:
            self.gps = False
        
            # Initialize filtered values if not already set
            if not hasattr(self, 'filtered_distance'):
                self.filtered_distance = odometer[0]
                self.filtered_alpha = odometer[1]
                self.filtered_theta = odometer[2]
                
            # Smoothing factor: lower values make it respond faster (less smoothing)
            alpha = 0.9  # Adjust this between 0 and 1 as needed

            # Update filtered values in real time
            self.filtered_distance = alpha * self.filtered_distance + (1 - alpha) * odometer[0]
            self.filtered_alpha    = alpha * self.filtered_alpha    + (1 - alpha) * odometer[1]
            self.filtered_theta    = alpha * self.filtered_theta    + (1 - alpha) * odometer[2]

            # Use these denoised values to update your state:
            denoised_odometer = np.array([self.filtered_distance,
                                        self.filtered_alpha,
                                        self.filtered_theta])

            # Update orientation:
            self.orientation = normalize_angle(previous_orientation + denoised_odometer[2])

            # Update position:
            self.position = previous_position.copy()
            self.position[0] += np.cos(normalize_angle(denoised_odometer[1] + previous_orientation)) * denoised_odometer[0]
            self.position[1] += np.sin(normalize_angle(denoised_odometer[1] + previous_orientation)) * denoised_odometer[0]

            # Constrain position within bounds:
            xmax = int(taille[0] / 2)
            xmin = -xmax
            ymax = int(taille[1] / 2)
            ymin = -ymax
            self.position[0] = min(max(self.position[0], xmin), xmax)
            self.position[1] = min(max(self.position[1], ymin), ymax)


        else : 
            #print("GPS available")
            self.filtered_distance = odometer[0]
            self.filtered_alpha = odometer[1]
            self.filtered_theta = odometer[2]
            
            self.gps = True
            self.position: np.array = position
            self.orientation: float = orientation
