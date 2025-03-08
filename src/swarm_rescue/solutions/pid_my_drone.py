"""
The drone explores the map following frontiers between explored an unexplored areas.
"""

from enum import Enum, auto
from collections import deque
import math
from typing import List, Optional
import numpy as np
import arcade
import csv
import os

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.utils import circular_mean, normalize_angle
from solutions.utils.pose import Pose
from solutions.utils.astar import *
from solutions.utils.messages import *
from solutions.utils.grids import *
from solutions.utils.dataclasses_config import *
from solutions.utils.exploration_tracker import *

# Paths
import path_creator.path1

class MyDronePID(DroneAbstract):
    class State(Enum):
        """
        All the states of the drone as a state machine
        """
        FOLLOWING_PATH = auto()

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         **kwargs)
        
        self.estimated_pose = Pose()

        # POSITION
        self.previous_position = deque(maxlen=1) 
        self.previous_position.append((0,0))  
        self.previous_orientation = deque(maxlen=1) 
        self.previous_orientation.append(0) 

        # PID PARAMS
        self.wall_following_params = WallFollowingParams()
        self.pid_params = PIDParams()
        self.past_ten_errors_angle = [0] * 10
        self.past_ten_errors_distance = [0] * 10
        
        # PATH FOLLOWING
        self.path_params = PathParams()
        self.indice_current_waypoint = 0
        self.inital_point_path = (0,0)
        self.finished_path = False
###########################PATH CHOICE##########################
        self.path = path_creator.path1.path
###########################PATH CHOICE##########################
        self.path_grid = []

        # TIME
        self.timestep_count = 0

        # LOG PARAMS
        self.id_log_file = input("Write a number for the log files")
        self.log_params = LogParams()
        self.serialize_PID_params()
        self.epsilon_angle = 0
        self.epsilon_lateral = 0

        # GRAPHICAL INTERFACE
        self.visualisation_params = VisualisationParams()

    def define_message_for_all(self):
        pass

    def control(self):
        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                self.measured_compass_angle(),self.odometer_values(),self.previous_position[-1],self.previous_orientation[-1],self.size_area)
        self.timestep_count += 1

        self.visualise_actions()

        self.log_pid_values()

        return self.follow_path(self.path)
    
    def log_pid_values(self):
        """Log PID error values to a CSV file at each timestep"""
        log_dir = "logs"
        
        # Define CSV file path
        csv_file = os.path.join(log_dir, f"{self.id_log_file}_pid_values_drone.csv")
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestep', 'epsilon_angle', 'epsilon_lateral'])
        
        # Append values
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.timestep_count, self.epsilon_angle, self.epsilon_lateral])

    def serialize_PID_params(self):
        """Serialize PIDParams to CSV file"""
        log_dir = "logs"
        
        csv_file = os.path.join(log_dir, f"{self.id_log_file}_pid_params_drone.csv")
        
        # Get all attributes from LogParams
        params_dict = self.pid_params.__dict__
        
        # Create/write headers if file doesn't exist
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(list(params_dict.keys()))
        
        # Write values
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(params_dict.values()))

    # Takes the current relative error and with a PID controller, returns the command
    # mode : "rotation" or "lateral" for now could be speed or other if implemented
    def pid_controller(self,command,epsilon,Kp,Kd,Ki,past_ten_errors,mode,command_slow = 0.8):
        
        past_ten_errors.pop(0)
        past_ten_errors.append(epsilon)
        if mode == "rotation":
            epsilon = normalize_angle(epsilon)
            self.epsilon_angle = epsilon
            deriv_epsilon = normalize_angle(self.odometer_values()[2])
        elif mode == "lateral":
            self.epsilon_lateral = epsilon
            deriv_epsilon = -np.sin(self.odometer_values()[1])*self.odometer_values()[0] # vitesse latérale
        elif mode == "forward" : 
            deriv_epsilon = self.odometer_values()[0]*np.cos(self.odometer_values()[1]) # vitesse longitudinale
        else : 
            raise ValueError("Mode not found")
        
        correction_proportionnelle = Kp * epsilon
        correction_derivee = Kd * deriv_epsilon
        correction_integrale = 0
        #correction_integrale = Ki * sum(past_ten_errors)
        correction = correction_proportionnelle + correction_derivee + correction_integrale
        command[mode] = correction
        command[mode] = min( max(-1,correction) , 1 )


        if mode == "rotation" : 
            if correction > command_slow :
                command["forward"] = self.wall_following_params.speed_turning

        return command
    
    def is_near_waypoint(self,waypoint):
        distance_to_waypoint = np.linalg.norm(waypoint - self.estimated_pose.position)
        if distance_to_waypoint < self.path_params.distance_close_waypoint:
            return True
        return False

    def follow_path(self,path):
        if path is not None:
            if self.is_near_waypoint(path[self.indice_current_waypoint]):
                self.indice_current_waypoint += 1
                if self.indice_current_waypoint >= len(path):
                    self.finished_path = True
                    self.indice_current_waypoint = 0
                    self.path = []
                    self.path_grid = []
                    return
        
        return self.go_to_waypoint(path[self.indice_current_waypoint][0],path[self.indice_current_waypoint][1])

    def go_to_waypoint(self,x,y):
        
        # ASSERVISSEMENT EN ANGLE
        dx = x - self.estimated_pose.position[0]
        dy = y - self.estimated_pose.position[1]
        epsilon = math.atan2(dy,dx) - self.estimated_pose.orientation
        epsilon = normalize_angle(epsilon)
        command_path = self.pid_controller({"forward": 1,"lateral": 0.0,"rotation": 0.0,"grasper": 0},epsilon,self.pid_params.Kp_angle,self.pid_params.Kd_angle,self.pid_params.Ki_angle,self.past_ten_errors_angle,"rotation",0.5)

        # ASSERVISSEMENT LATERAL
        if self.indice_current_waypoint == 0:
            x_previous_waypoint,y_previous_waypoint = self.inital_point_path
        else : 
            x_previous_waypoint,y_previous_waypoint = self.path[self.indice_current_waypoint-1][0],self.path[self.indice_current_waypoint-1][1]

        epsilon_distance = compute_relative_distance_to_droite(x_previous_waypoint,y_previous_waypoint,x,y,self.estimated_pose.position[0],self.estimated_pose.position[1])
        # epsilon distance needs to be signed (positive if the angle relative to the theoritical path is positive)
        command_path = self.pid_controller(command_path,epsilon_distance,self.pid_params.Kp_distance,self.pid_params.Kd_distance,self.pid_params.Ki_distance,self.past_ten_errors_distance,"lateral",0.5)

        return command_path

    

    def draw_path(self, path):
        length = len(path)
        pt2 = None
        for ind_pt in range(length):
            pose = path[ind_pt]
            pt1 = pose + self._half_size_array
            if ind_pt > 0:
                arcade.draw_line(float(pt2[0]),
                                 float(pt2[1]),
                                 float(pt1[0]),
                                 float(pt1[1]), [125,125,125])
            pt2 = pt1

    def draw_top_layer(self):
        if self.visualisation_params.draw_path:
            self.draw_path(self.path)

    def visualise_actions(self):
        """
        It's mandatory to use draw_top_layer to draw anything on the interface
        """
        self.draw_top_layer()