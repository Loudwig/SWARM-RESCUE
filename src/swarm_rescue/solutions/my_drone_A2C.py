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
        WAITING = 1
        EXPLORING = 2
        FINISHED_EXPLORING = 3

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
        self.previous_state = self.State.WAITING # Utile pour vérfier que c'est la première fois que l'on rentre dans un état
        
        # WAITING STATE
        self.step_waiting = 50 # step waiting without mooving when loosing the sight of wounded
        self.step_waiting_count = 0

        # GRASPING 
        self.grasping_speed = 0.3

        # Paramètre following walls ---------------------
        self.dmax = 60 # distance max pour suivre un mur
        self.dist_to_stay = 40 # distance à laquelle on veut rester du mur
        self.speed_following_wall = 0.3
        self.speed_turning = 0.05

        self.Kp_angle = 4/math.pi # correction proportionnelle # theoriquement j'aurais du mettre 2
        self.Kp_angle_1 = 9/math.pi
        self.Kd_angle_1 = self.Kp_angle_1/10
        self.Kd_angle =self.Kp_angle/10 # correction dérivée
        self.Ki_angle = (1/10)*(1/20)*2/math.pi#4 # (1/10) * 1/20 # correction intégrale
        self.past_ten_errors_angle = [0]*10
        

        self.Kp_distance_1 = 2/(abs(10))
        self.Ki_distance_1 = 1/abs(10) *1/20 *1/10
        self.Kd_distance_1 = 2*self.Kp_distance_1


        self.Kp_distance = 2/(abs(self.dmax-self.dist_to_stay))
        self.Ki_distance = 1/abs(self.dist_to_stay-self.dmax) *1/20 *1/10
        self.Kd_distance = 2*self.Kp_distance
        self.past_ten_errors_distance = [0]*10
        # -----------------------------------------
        
        # following path
        self.indice_current_waypoint = 0
        self.inital_point_path = (0,0)
        self.finished_path = False
        self.path = []
        self.path_grid = []
        self.distance_close_waypoint = 20

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
        
        # MAPPING
        self.mapping(display = self.display_map)
        
        # RECUPÈRATION INFORMATIONS SENSORS (LIDAR, SEMANTIC)
        found_wall,epsilon_wall_angle, min_dist = self.process_lidar_sensor(self.lidar())
        found_wounded, found_rescue_center,epsilon_wounded,epsilon_rescue_center,is_near_rescue_center = self.process_semantic_sensor()

        # paramètres responsables des transitions

        paramètres_transitions = { "found_wall": found_wall, "found_wounded": found_wounded, "found_rescue_center": found_rescue_center,"grasped_entities" : bool(self.base.grasper.grasped_entities), "\nstep_waiting_count": self.step_waiting_count} 
        #print(paramètres_transitions)
        # TRANSITIONS OF THE STATE 
        self.state_update(found_wall,found_wounded,found_rescue_center)

        ##########
        # COMMANDS FOR EACH STATE
        ##########
        command_nothing = {"forward": 0.0,"lateral": 0.0,"rotation": 0.0,"grasper": 0}
        command_following_walls = {"forward": self.speed_following_wall,"lateral": 0.0,"rotation": 0.0,"grasper": 0}
        command_grasping_wounded = {"forward": self.grasping_speed,"lateral": 0.0,"rotation": 0.0,"grasper": 1}
        command_tout_droit = {"forward": 0.5,"lateral": 0.0,"rotation": 0.0,"grasper": 0}
        command_searching_rescue_center = {"forward": self.speed_following_wall,"lateral": 0.0,"rotation": 0.0,"grasper": 1}
        command_going_rescue_center = {"forward": 3*self.grasping_speed,"lateral": 0.0,"rotation": 0.0,"grasper": 1}
        command_following_path_with_wounded = {"forward": 0.5,"lateral": 0.0,"rotation": 0.0,"grasper": 1}

        # WAITING STATE
        if self.state is self.State.WAITING:
            self.step_waiting_count += 1
            return command_nothing

        elif  self.state is self.State.EXPLORING:
            return command_tout_droit
         
        elif  self.state is self.State.FOLLOWING_WALL:

            epsilon_wall_angle = normalize_angle(epsilon_wall_angle) 
            epsilon_wall_distance =  min_dist - self.dist_to_stay 
            
            self.logging_variables({"epsilon_wall_angle": epsilon_wall_angle, "epsilon_wall_distance": epsilon_wall_distance})
            
            command_following_walls = self.pid_controller(command_following_walls,epsilon_wall_angle,self.Kp_angle,self.Kd_angle,self.Ki_angle,self.past_ten_errors_angle,"rotation")
            command_following_walls = self.pid_controller(command_following_walls,epsilon_wall_distance,self.Kp_distance,self.Kd_distance,self.Ki_distance,self.past_ten_errors_distance,"lateral")
        
            return command_following_walls

        elif self.state is self.State.GRASPING_WOUNDED:
            
            epsilon_wounded_angle = normalize_angle(epsilon_wounded) 
            command_grasping_wounded = self.pid_controller(command_grasping_wounded,epsilon_wounded_angle,self.Kp_angle,self.Kd_angle,self.Ki_angle,self.past_ten_errors_angle,"rotation")
            
            return command_grasping_wounded

        elif self.state is self.State.SEARCHING_RESCUE_CENTER:
            # Calculate path at the beginning of the state
            if self.previous_state is not self.State.SEARCHING_RESCUE_CENTER:
                
                # CREATION DU PATH : 
                
                # enregistre la position lorsque l'on rentre dans LE STATE. -> use to make l'asservissement lateral
                self.inital_point_path = self.estimated_pose.position[0],self.estimated_pose.position[1]
                MAP = self.grid.to_binary_map() # Convertit la MAP de proba en MAP binaire
                grid_initial_point_path = self.grid._conv_world_to_grid(self.inital_point_path[0],self.inital_point_path[1]) # initial point in grid coordinates

                # On élargie les mur au plus large possible pour trouver un chemin qui passe le plus loin des murs/obstacles possibles.
                Max_inflation =  7
                for x in range(Max_inflation+1):
                    #print("inflation : ",Max_inflation - x)
                    MAP_inflated = inflate_obstacles(MAP,Max_inflation-x)
                    # redefinir le start comme le point libre le plus proche de la position actuelle à distance max d'inflation pour pas que le point soit inacessible.
                    # SUREMENT UNE MEILLEUR MANIERE DE FAIRE.
                    start_point_x, start_point_y = next_point_free(MAP_inflated,grid_initial_point_path[0],grid_initial_point_path[1],Max_inflation-x + 3)
                    end_point_x, end_point_y = next_point_free(MAP_inflated,self.grid.initial_cell[0],self.grid.initial_cell[1],Max_inflation-x+ 3) # initial cell already in grid coordinates.
                    path = a_star_search(MAP_inflated,(start_point_x,start_point_y),(end_point_x,end_point_y))
                    
                    if len(path) > 0:
                        #print( f"inflation : {Max_inflation-(x)}")
                        break
                
                # Remove colinear points
                path_simplified = simplify_collinear_points(path)

                # Simplification par ligne de vue
                path_line_of_sight = simplify_by_line_of_sight(path_simplified, MAP_inflated)
               
                # Simplification par Ramer-Douglas-Peucker avec epsilon = 0.5 par exemple
                path_rdp = ramer_douglas_peucker(path_line_of_sight, 0.5)
                self.path_grid = path_rdp
                self.path = [self.grid._conv_grid_to_world(x,y) for x,y in self.path_grid]
                self.indice_current_waypoint = 0
                #print("Path calculated")
                
            command = self.follow_path(self.path)
            return command
        
        elif self.state is self.State.GOING_RESCUE_CENTER:
            epsilon_rescue_center = normalize_angle(epsilon_rescue_center) 
            command_going_rescue_center = self.pid_controller(command_going_rescue_center,epsilon_rescue_center,self.Kp_angle,self.Kd_angle,self.Ki_angle,self.past_ten_errors_angle,"rotation")
            
            if is_near_rescue_center:
                command_going_rescue_center["forward"] = 0.0
                command_going_rescue_center["rotation"] = 1.0
            
            return command_going_rescue_center
        
        # STATE NOT FOUND raise error
        raise ValueError("State not found")
    
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
    
    def state_update(self,found_wall,found_wounded,found_rescue_center):
        
        self.previous_state = self.state
        #print(f"Previous state : {self.previous_state}")
        if self.state is self.State.EXPLORING and (found_wounded):
            self.state = self.State.FINISHED_EXPLORING
        
        elif (self.state is self.State.WAITING and self.step_waiting_count >= self.step_waiting):
            self.state = self.State.EXPLORING
            self.step_waiting_count = 0
        
        #print(f"State : {self.state}")
    
    def mapping(self, display = False):
        
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
        
        if display and (self.timestep_count % 5 == 0):
             self.grid.display(self.grid.grid,
                               self.estimated_pose,
                               title="occupancy grid")
             self.grid.display(self.grid.zoomed_grid,
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