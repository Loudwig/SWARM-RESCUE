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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        

        # MAPING
        self.estimated_pose = Pose() # Fonctionne commant sans le GPS ?  erreur ou qu'est ce que cela retourne ? 
        resolution = 8 # pourquoi ?  Ok bon compromis entre précision et temps de calcul
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())
    


        self.frame_stack = 4
        self.frame_buffer = deque(maxlen=self.frame_stack)  
        # l'initialisé avec des images de la map
        for _ in range(self.frame_stack):
            self.frame_buffer.append(torch.zeros((1,2,self.grid.grid.shape[0], self.grid.grid.shape[1]), dtype=torch.float32, device=device))
        
        # Par défaut on load un model. Si on veut l'entrainer il faut redefinir policy net et value net        
        # Si model enregistré
            
        

        try : 
                
             self.policy_model_path = "solutions/utils/trained_models/policy_net.pth"
             self.value_model_path = "solutions/utils/trained_models/value_net.pth"
             self.policy_net = NetworkPolicy(h=self.grid.grid.shape[0],w=self.grid.grid.shape[1],frame_stack=self.frame_stack)
             self.value_net = NetworkValue(h=self.grid.grid.shape[0],w=self.grid.grid.shape[1],frame_stack=self.frame_stack)
             self.policy_net.load_state_dict(torch.load(self.policy_model_path))
             self.value_net.load_state_dict(torch.load(self.value_model_path))
             # self.policy_net.to(device)
             # self.value_net.to(device)
             print("Model loaded successfully")
    
        except :
            print("No model found, using default policy and value networks")
            self.policy_net = NetworkPolicy(h=self.grid.grid.shape[0],w=self.grid.grid.shape[1],frame_stack=self.frame_stack)
            self.value_net = NetworkValue(h=self.grid.grid.shape[0],w=self.grid.grid.shape[1],frame_stack=self.frame_stack)

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
        self.timestep_count +=1 
        self.update_map_pose_speed()
        # print( f"EXPLORATION SCORE : {self.grid.exploration_score}")
        maps = torch.tensor([self.grid.grid, self.grid.position_grid], dtype=torch.float32, device=device).unsqueeze(0)
        global_state = torch.tensor([self.estimated_pose.position[0], self.estimated_pose.position[1], self.estimated_pose.orientation, self.estimated_pose.vitesse_X, self.estimated_pose.vitesse_Y, self.estimated_pose.vitesse_angulaire], dtype=torch.float32, device=device).unsqueeze(0)
        self.frame_buffer.append(maps)
        stacked_frames = torch.cat(list(self.frame_buffer), dim=1)
        action,_ = self.select_action(stacked_frames,global_state)
        command = self.process_actions(action)
        return command

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
        if (abs(forward) < 0.95 and abs(lateral) < 0.95 and abs(rotation) < 0.95):
            print("actions in range")
            
            
        
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

    def select_action(self, state_map,state_vector):
        
        state_map = torch.FloatTensor(state_map).to(device)
        state_vector = torch.FloatTensor(state_vector).to(device)
        means, log_stds = self.policy_net(state_map,state_vector)

        # Sample continuous actions
        stds = torch.exp(log_stds)
        sampled_continuous_actions = means + torch.randn_like(means) * stds

        # print(f"actions before tanh: {sampled_continuous_actions}")
        continuous_actions = torch.tanh(sampled_continuous_actions)
        # print(f"actions after tanh: {continuous_actions}")
        
        # Compute log probabilities 
        # log prob with tanh squashing and gaussian prob
        log_probs_continuous = -0.5 * (((sampled_continuous_actions - means) / (stds + 1e-8)) ** 2 + 2 * log_stds + math.log(2 * math.pi))
        log_probs_continuous = log_probs_continuous.sum(dim=1)

        log_probs_continuous = log_probs_continuous - torch.sum(torch.log(1 - continuous_actions ** 2 + 1e-6), dim=1)

        # Combine actions
        action = continuous_actions[0]
        #print(f"action: {action}")
        
        if torch.isnan(action).any():
            print("NaN detected in action")
            return [0, 0, 0], None
        
        log_prob = log_probs_continuous 
        return action.detach().cpu(), log_prob


