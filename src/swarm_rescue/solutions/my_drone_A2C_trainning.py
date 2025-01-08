from enum import Enum
from collections import deque
import math
from typing import Optional
import cv2
import numpy as np
import arcade
import torch
import sys
from pathlib import Path
import torch.nn as nn
import torch.optim as optim


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maps.map_intermediate_01 import MyMapIntermediate01 as M1
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.entities.rescue_center import RescueCenter
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.utils.utils import circular_mean, normalize_angle
from solutions.utils.pose import Pose
from solutions.utils.grid import OccupancyGrid
from solutions.utils.astar import *
from solutions.utils.NetworkPolicy import NetworkPolicy
from solutions.utils.NetworkValue import NetworkValue


class MyDroneHulk(DroneAbstract):
    class State(Enum):
        """
        All the states of the drone as a state machine
        """
        EXPLORING = 1
        FINISHED_EXPLORING = 2

    def __init__(self,identifier: Optional[int] = None,misc_data: Optional[MiscData] = None,**kwargs):
        super().__init__(identifier=identifier,misc_data=misc_data,**kwargs)
        
        # Par défaut on load un model. Si on veut l'entrainer il faut redefinir policy net et value net        
        # Si model enregistré
            
        # MAPING
        self.estimated_pose = Pose() # Fonctionne commant sans le GPS ?  erreur ou qu'est ce que cela retourne ? 
        resolution = 10 # pourquoi ?  Ok bon compromis entre précision et temps de calcul
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())
    

        self.display_map = True # Display the probability map during the simulation

        # POSTION 
        # deque remplit de 0 

        self.previous_position = deque(maxlen=1) 
        self.previous_position.append((0,0))  
        self.previous_orientation = deque(maxlen=1) 
        self.previous_orientation.append(0) 

        # Initialisation du state
        self.state  = self.State.EXPLORING
        self.previous_state = self.State.EXPLORING # Utile pour vérfier que c'est la première fois que l'on rentre dans un état

        # paramètres logs
        self.record_log = False
        self.log_file = "logs/log.txt"
        self.log_initialized = False
        self.flush_interval = 50  # Number of timesteps before flushing buffer
        self.timestep_count = 0  # Counter to track timesteps   
         
    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def control(self):
        pass

    def process_semantic_sensor(self):
        semantic_values = self.semantic_values()
        found_wounded = False
        for data in semantic_values:
            if (data.entity_type ==
                    DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped):
                found_wounded = True
                
        return found_wounded
    
    def process_lidar_sensor(self,self_lidar):
        
        lidar_values = self_lidar.get_sensor_values()

        if lidar_values is None:
            return 0
        
        ray_angles = self_lidar.ray_angles
        size = self_lidar.resolution

        angle_nearest_obstacle = 0
        if size != 0:
            min_dist = min(lidar_values)

        return min_dist
    
    def compute_reward(self,is_collision, found_wounded, time_penalty,action):
        forward = action["forward"]
        lateral = action["lateral"]
        rotation = action["rotation"]

        reward = 0

        # Penalize collisions heavily

        if is_collision:
            reward -= 50

        if found_wounded:
            reward += 100

        reward += self.grid.exploration_score

        # Penalize idling or lack of movement
        reward -= time_penalty

        # give penalty when action are initialy not between -0.9 and 0.9 
        if (abs(forward) < 0.5 and abs(lateral) < 0.5 and abs(rotation) < 0.5):
            print("actions not saturated")
            
        return reward

    def state_update(self,found_wounded):
        
        self.previous_state = self.state
        #print(f"Previous state : {self.previous_state}")
        if self.state is self.State.EXPLORING and (found_wounded):
            self.state = self.State.FINISHED_EXPLORING
        
        #print(f"State : {self.state}")
    
    def update_map_pose_speed(self):
        
        if self.timestep_count == 1: # first iteration
            # print("Starting control")
            start_x, start_y = self.measured_gps_position() # never none ? 
            # print(f"Initial position: {start_x}, {start_y}")
            self.grid.set_initial_cell(start_x, start_y)
        

        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                   self.measured_compass_angle(),self.odometer_values(),self.previous_position[-1],self.previous_orientation[-1],self.size_area)
        
        self.previous_position.append(self.estimated_pose.position)
        self.previous_orientation.append(self.estimated_pose.orientation)
        
        self.grid.update_grid(pose=self.estimated_pose)
        
    def showMaps(self,display_zoomed_position_grid = False,display_zoomed_grid = False):
        if  (self.timestep_count % 5 == 0):
            
            if display_zoomed_position_grid:
                self.grid.display(self.grid.zoomed_position_grid,
                                self.estimated_pose,
                                title="zoomed position grid")
            
            if display_zoomed_grid:
                self.grid.display(self.grid.zoomed_grid ,
                                    self.estimated_pose,
                                    title="zoomed occupancy grid")

    # Use this function only at one place in the control method. Not handled othewise.
    # params : variables_to_log : dict of variables to log with keys as variable names and values as variable values.
    def logging_variables(self, variables_to_log):
        """
        Buffers and logs variables to the log file when the buffer reaches the flush interval.

        :param variables_to_log: dict of variables to log with keys as variable names 
                                and values as variable values.
        """
        if not self.record_log:
            return

        # Initialize the log buffer if not already done
        if not hasattr(self, "log_buffer"):
            self.log_buffer = []

        # Append the current variables to the buffer
        log_entry = {"Timestep": self.timestep_count, **variables_to_log}
        self.log_buffer.append(log_entry)

        # Write the buffer to file when it reaches the flush interval
        if len(self.log_buffer) >= self.flush_interval:
            mode = "w" if not self.log_initialized else "a"
            with open(self.log_file, mode) as log_file:
                # Write the header if not initialized
                if not self.log_initialized:
                    headers = ",".join(log_entry.keys())
                    log_file.write(headers + "\n")
                    self.log_initialized = True

                # Write buffered entries
                for entry in self.log_buffer:
                    line = ",".join(map(str, entry.values()))
                    log_file.write(line + "\n")

            # Clear the buffer
            self.log_buffer.clear()
        
    def draw_top_layer(self):
        pass

    def process_actions(self,actions):
        return {
            "forward": actions[0],
            "lateral": actions[1],
            "rotation": actions[2],
            "grasper": 0  # 0 or 1 for grasping
        }

    def process_state_before_network(self,positionX,positionY,orientation,vitesse_X,vitesse_Y,vitesse_angulaire):
        Px,Py = self.grid._conv_world_to_grid(positionX,positionY)
        #print(f"Px,Py : {Px,Py}")
        return torch.tensor([Px,Py,orientation,vitesse_X,vitesse_Y,vitesse_angulaire], dtype=torch.float32, device=device).unsqueeze(0)


