"""
Le drone suit un path après avoir trouver le wounded person.
"""

from enum import Enum
from collections import deque
import math
from typing import Optional
import cv2
import numpy as np
import arcade

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


class MyDroneHulk(DroneAbstract):
    class State(Enum):
        """
        All the states of the drone as a state machine
        """
        EXPLORING = 1
        FINISHED_EXPLORING = 2

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         **kwargs)
        
        # MAPING
        self.estimated_pose = Pose() # Fonctionne commant sans le GPS ?  erreur ou qu'est ce que cela retourne ? 
        resolution = 8 # pourquoi ?  Ok bon compromis entre précision et temps de calcul
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
        
        # increment the iteration counter
        self.timestep_count += 1
        
        # MAPPING is updating the pose also so donc forget to do it at the beginnig
        self.update_map_pose_speed()
        self.showMaps(True,True)
        
        # RECUPÈRATION INFORMATIONS SENSORS (LIDAR, SEMANTIC)
        min_dist = self.process_lidar_sensor(self.lidar())
        found_wounded = self.process_semantic_sensor()

        # paramètres responsables des transitions

        # TRANSITIONS OF THE STATE 
        self.state_update(found_wounded)
        print(f"EXPLORATION REWARD : {self.grid.exploration_score}")
        print(f"Vitesse (X,Y,0) : {self.estimated_pose.vitesse_X}{self.estimated_pose.vitesse_Y}{self.estimated_pose.vitesse_angulaire}")
        

        ##########
        # COMMANDS FOR EACH STATE
        ##########
        command_nothing = {"forward": 0.0,"lateral": 0.0,"rotation": 0.0,"grasper": 0}
        command_tout_droit = {"forward": 0.3,"lateral": 0.0,"rotation": 0.0,"grasper": 0}

        if  self.state is self.State.EXPLORING:
            return command_tout_droit
        
        else :
            return command_nothing

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
    
    def state_update(self,found_wounded):
        
        self.previous_state = self.state
        #print(f"Previous state : {self.previous_state}")
        if self.state is self.State.EXPLORING and (found_wounded):
            self.state = self.State.FINISHED_EXPLORING
        
        #print(f"State : {self.state}")
    
    def update_map_pose_speed(self):
        
        if self.timestep_count == 1: # first iteration
            print("Starting control")
            start_x, start_y = self.measured_gps_position() # never none ? 
            print(f"Initial position: {start_x}, {start_y}")
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