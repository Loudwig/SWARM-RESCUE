"""
The drone explores the map following frontiers between explored an unexplored areas.
"""

from enum import Enum, auto
from collections import deque
import math
from typing import List, Optional
import numpy as np
import arcade

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
##########################################################
        self.path = self.path = [
    (-332, 263),# Starting point
    (-78, 238),
    (211, 243),
    (348, 166),
    (359, -157),
    (191, -273),
    (-144, -254),
    (-306, -162),
    (-336, 41),
    (-216, 134),
    (18, 166),
    (171, 147),
    (279, 31),
    (277, -118),
    (-52, -165),
    (-150, -67),
    (-151, 24),
    (16, 70),
    (137, 25),
    (96, -48),
    (-25, -47),
]
##########################################################
        self.path_grid = []

        # TIME
        self.timestep_count = 0

        # LOG PARAMS
        self.log_params = LogParams()   

        # GRAPHICAL INTERFACE
        self.visualisation_params = VisualisationParams()

    def define_message_for_all(self):
        pass

    def control(self):
        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                self.measured_compass_angle(),self.odometer_values(),self.previous_position[-1],self.previous_orientation[-1],self.size_area)
        self.timestep_count += 1

        self.visualise_actions()

        return self.follow_path(self.path)

    # Takes the current relative error and with a PID controller, returns the command
    # mode : "rotation" or "lateral" for now could be speed or other if implemented
    def pid_controller(self,command,epsilon,Kp,Kd,Ki,past_ten_errors,mode,command_slow = 0.8):
        
        past_ten_errors.pop(0)
        past_ten_errors.append(epsilon)
        if mode == "rotation":
            epsilon = normalize_angle(epsilon)
            deriv_epsilon = normalize_angle(self.odometer_values()[2])
        elif mode == "lateral":
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
        command_path = self.pid_controller({"forward": 1,"lateral": 0.0,"rotation": 0.0,"grasper": 0},epsilon,self.pid_params.Kp_angle_1,self.pid_params.Kd_angle_1,self.pid_params.Ki_angle,self.past_ten_errors_angle,"rotation",0.5)

        # ASSERVISSEMENT LATERAL
        if self.indice_current_waypoint == 0:
            x_previous_waypoint,y_previous_waypoint = self.inital_point_path
        else : 
            x_previous_waypoint,y_previous_waypoint = self.path[self.indice_current_waypoint-1][0],self.path[self.indice_current_waypoint-1][1]

        epsilon_distance = compute_relative_distance_to_droite(x_previous_waypoint,y_previous_waypoint,x,y,self.estimated_pose.position[0],self.estimated_pose.position[1])
        # epsilon distance needs to be signed (positive if the angle relative to the theoritical path is positive)
        command_path = self.pid_controller(command_path,epsilon_distance,self.pid_params.Kp_distance_1,self.pid_params.Kd_distance_1,self.pid_params.Ki_distance_1,self.past_ten_errors_distance,"lateral",0.5)
        
        # ASSERVISSENT EN DISTANCE
        epsilon_distance_to_waypoint = np.linalg.norm(np.array([x,y]) - self.estimated_pose.position)
        # command_path = self.pid_controller(command_path,epsilon_distance_to_waypoint,self.pid_params.Kp_distance_2,self.pid_params.Kp_distance_2,self.pid_params.Ki_distance_1,self.past_ten_errors_distance,"forward",1)

        return command_path

    # Use this function only at one place in the control method. Not handled othewise.
    # params : variables_to_log : dict of variables to log with keys as variable names and values as variable values.
    def logging_variables(self, variables_to_log):
        """
        Buffers and logs variables to the log file when the buffer reaches the flush interval.

        :param variables_to_log: dict of variables to log with keys as variable names 
                                and values as variable values.
        """
        if not self.log_params.record_log:
            return

        # Initialize the log buffer if not already done
        if not hasattr(self, "log_buffer"):
            self.log_buffer = []

        # Append the current variables to the buffer
        log_entry = {"Timestep": self.timestep_count, **variables_to_log}
        self.log_buffer.append(log_entry)

        # Write the buffer to file when it reaches the flush interval
        if len(self.log_buffer) >= self.log_params.flush_interval:
            mode = "w" if not self.log_initialized else "a"
            with open(self.log_params.log_file, mode) as log_file:
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