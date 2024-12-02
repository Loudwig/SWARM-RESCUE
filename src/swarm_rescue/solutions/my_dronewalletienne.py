"""
Le drone suit les murs
"""

from enum import Enum
import math
from typing import Optional

import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData

from spg_overlay.utils.utils import normalize_angle


class MyDroneWall(DroneAbstract):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        SEARCHING_WALL = 1
        FOLLOWING_WALL = 2

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         **kwargs)
        # Initialisation de l'activité
        self.state  = self.Activity.SEARCHING_WALL
        
        
        # Paramètre following walls
        self.dmax = 50 # distance max pour suivre un mur
        self.dist_to_stay = 30 # distance à laquelle on veut rester du mur
        
        self.Kp = 4/math.pi # correction proportionnelle # theoriquement j'aurais du mettre 2
        self.Kd = 1 # correction dérivée
        self.Kdist = 1 # correction proportionnelle à la distance
        self.speed_following_wall = 0.2
        self.Ki = (1/10) * 1/20 # correction intégrale
        self.past_ten_errors_angle = [0]*10
        
        self.Kp_distance = 1/(abs(self.dmax-self.dist_to_stay))
        self.Ki_distance = 1/abs(self.dist_to_stay-self.dmax) * 1/10 * 1/20
        self.past_ten_errors_distance = [0]*10

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def control(self):
        found_wall,epsilon_wall_angle, min_dist = self.process_lidar_sensor(self.lidar())

        #############
        # TRANSITIONS OF THE STATE MACHINE
        #############

        if self.state is self.Activity.SEARCHING_WALL and found_wall:
            self.state = self.Activity.FOLLOWING_WALL
        
        elif self.state is self.Activity.FOLLOWING_WALL and not found_wall:
            self.state = self.Activity.SEARCHING_WALL

        print(f"State : {self.state}")
        
        ##########
        # COMMANDS FOR EACH STATE
        ##########
        
        command = {"forward": self.speed_following_wall,"lateral": 0.0,"rotation": 0.0,"grasper": 0}

        if found_wall :

            epsilon_wall_angle = normalize_angle(epsilon_wall_angle) 
            
            self.past_ten_errors_angle.pop(0)
            self.past_ten_errors_angle.append(epsilon_wall_angle)
            
            deriv_epsilon_wall_angle = normalize_angle(self.odometer_values()[2]) # vitesse angulaire

            correction_proportionnelle = self.Kp * epsilon_wall_angle
            correction_derivee = self.Kd * deriv_epsilon_wall_angle
            #correction_sharp_turns = self.Kdist*1/(1+np.exp(-0.1*(min_dist-self.dmax)))
            correction_integrale = self.Ki * sum(self.past_ten_errors_angle)

            rotation = correction_proportionnelle + correction_derivee + correction_integrale 
            rotation = min( max(-1,rotation) , 1 ) 

            # rotation ne commence que si l'angle est supérieur à activation
            activation = 0.05
            if abs(rotation) < activation: # en soit pas obligatoire mais pas besoin de solliciter les moteurs pour un rien, il vont décéder après.
                rotation = 0.0

            command["rotation"] = rotation

            # lateral control to stay at a distance from the wall
            epsilon_wall_distance = + min_dist - self.dist_to_stay 
            deriv_epsilon_wall_distance = - np.sin(self.odometer_values()[1])*self.odometer_values()[0] # vitesse latérale
            
            print(f"deriv_epsilon_wall_distance : {deriv_epsilon_wall_distance}")
            self.past_ten_errors_distance.pop(0)
            self.past_ten_errors_distance.append(epsilon_wall_distance)

            correction_proportionnelle_distance = self.Kp_distance * epsilon_wall_distance
            correction_derivee_distance = self.Kd * deriv_epsilon_wall_distance
            correction_integrale_distance = self.Ki_distance * sum(self.past_ten_errors_distance)

            lateral = correction_proportionnelle_distance + correction_derivee_distance + correction_integrale_distance
            lateral = min( max(-1,lateral) , 1 )

            activation_distance = 0.1
            if abs(lateral) < activation_distance:
                lateral = 0.0

            command["lateral"] = lateral
            print(f"rotation_command : {rotation} | lateral_command : {lateral} |  epsilon_wall_angle : {epsilon_wall_angle} | epsilon_wall_distance : {epsilon_wall_distance} | min_dist : {min_dist}")

        else:
            pass

        return command
    
    def process_lidar_sensor(self,self_lidar):
        """
        -> ( bool near_obstacle , float epsilon_wall_angle )
        where epsilon_wall_angle is the (counter-clockwise convention) angle made
        between the drone and the nearest wall
        """
        lidar_values = self_lidar.get_sensor_values()

        if lidar_values is None:
            return (False,0)
        
        ray_angles = self_lidar.ray_angles
        size = self_lidar.resolution

        angle_nearest_obstacle = 0
        if size != 0:
            min_dist = min(lidar_values)
            angle_nearest_obstacle = ray_angles[np.argmin(lidar_values)]

        near_obstacle = False
        if min_dist < self.dmax: # pourcentage de la vitesse je pense
            near_obstacle = True

        epsilon_wall_angle = angle_nearest_obstacle - np.pi/2
        #print(epsilon_wall_angle,near_obstacle)

        return (near_obstacle,epsilon_wall_angle,min_dist)