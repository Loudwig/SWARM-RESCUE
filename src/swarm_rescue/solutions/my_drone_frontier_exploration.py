"""
The drone explores the map following frontiers between explored an unexplored areas.
"""

from enum import Enum, auto
from collections import deque
import math
from typing import Optional
import cv2
import numpy as np
import arcade
from gym.envs.toy_text.blackjack import score

from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.entities.rescue_center import RescueCenter
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.utils.utils import circular_mean, normalize_angle
from solutions.utils.pose import Pose
from spg_overlay.utils.grid import Grid
from solutions.utils.astar import *
from solutions.utils.messages import DroneMessage
from solutions.utils.grids import *
from solutions.utils.dataclasses_config import *

from scipy.optimize import linear_sum_assignment


class MyDroneFrontex(DroneAbstract):
    class State(Enum):
        """
        All the states of the drone as a state machine
        """
        WAITING = auto()    # Assigns 1

        SEARCHING_WALL = auto()     # Assigns 2 etc ... This allows to easily add new states
        FOLLOWING_WALL = auto()

        EXPLORING_FRONTIERS = auto()

        GRASPING_WOUNDED = auto()
        SEARCHING_RESCUE_CENTER = auto()
        GOING_RESCUE_CENTER = auto()

        SEARCHING_RETURN_AREA = auto()
        GOING_RETURN_AREA = auto()

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         **kwargs)
        
        # MAPPING
        self.mapping_params = MappingParams()
        self.estimated_pose = Pose() 
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=self.mapping_params.resolution,
                                  lidar=self.lidar(),semantic=self.semantic())

        # POSITION
        self.previous_position = deque(maxlen=1) 
        self.previous_position.append((0,0))  
        self.previous_orientation = deque(maxlen=1) 
        self.previous_orientation.append(0) 

        # STATE INITIALISATION
        self.state  = self.State.WAITING
        self.previous_state = self.State.WAITING # Utile pour vérfier que c'est la première fois que l'on rentre dans un état
        
        # PARAMS FOR DIFFERENT STATES 

            # WAITING STATE
        self.waiting_params = WaitingStateParams()
        self.step_waiting_count = 0

            # GRASPING 
        self.grasping_params = GraspingParams()

            # WALL FOLLOWING
        self.wall_following_params = WallFollowingParams()

            # FRONTIER EXPLORATION
        self.explored_all_frontiers = False
        self.next_frontier = None
        self.next_frontier_centroid = None

        # PID PARAMS
        self.pid_params = PIDParams()
        self.past_ten_errors_angle = [0] * 10
        self.past_ten_errors_distance = [0] * 10
        
        # PATH FOLLOWING
        self.path_params = PathParams()
        self.indice_current_waypoint = 0
        self.inital_point_path = (0,0)
        self.finished_path = True
        self.path = []
        self.path_grid = []

        # LOG PARAMS
        self.log_params = LogParams()   
        self.timestep_count = 0

        # GRAPHICAL INTERFACE
        self.visualisation_params = VisualisationParams()

        self.wounded_locked = []
        self.other_drones_pos = []
    
    def reset_exploration_path_params(self):
        """
        Resets the parameters related to the exploration path.
        """
        self.next_frontier = None
        self.next_frontier_centroid = None
        self.finished_path = True
        self.path = []

    def define_message_for_all(self):
        inKillZone =self.lidar().get_sensor_values() is None 
        message = []
        if self.timestep_count<=1 or inKillZone:
            return None

        if self.timestep_count % CommunicationParams().TIME_INTERVAL == 0:
            confiance = self.compute_confidence(self.estimated_pose.gps)
            message.append(DroneMessage(subject=DroneMessage.Subject.MAPPING, arg={"map": self.grid.grid, "confiance": confiance}))
        #message = self.grid.to_update(pose=self.estimated_pose)
        else :
            message.append(DroneMessage(subject=DroneMessage.Subject.PASS, arg=None))

        if self.state == self.State.GRASPING_WOUNDED or self.state == self.State.SEARCHING_RESCUE_CENTER or self.state == self.State.GOING_RESCUE_CENTER:
            broadcast_msg = DroneMessage(
                subject=DroneMessage.Subject.LOCK_WOUNDED,
                arg=(self.identifier, self.estimated_pose.position.tolist())
            )
            message.append(broadcast_msg)
        loc_msg = DroneMessage(
            subject=DroneMessage.Subject.FRONTIER_PRIO,
            arg=(self.identifier, self.estimated_pose.position.tolist()))
        message.append(loc_msg)
        return message

    def communication_management(self):
        self.wounded_locked = []
        self.other_drones_pos = []
        if self.communicator:
            received_messages = self.communicator.received_messages
            for msg in received_messages:
                for drone_msg in msg[1]:
                    if not isinstance(drone_msg, DroneMessage):
                        raise ValueError("Invalid message type. Expected a DroneMessage instance.")
                    if drone_msg.subject == DroneMessage.Subject.MAPPING :
                        self.grid.merge_maps(drone_msg.arg["map"],drone_msg.arg["confiance"])
                    if drone_msg.subject == DroneMessage.Subject.LOCK_WOUNDED:
                        drone_id, position = drone_msg.arg
                        self.wounded_locked.append((drone_id, position))
                    if drone_msg.subject == DroneMessage.Subject.FRONTIER_PRIO:
                        drone_id, position = drone_msg.arg
                        self.other_drones_pos.append((drone_id,position))

    def compute_confidence(self, gps):
        if gps is None: # Si en zone non gps
            return 0.1
        else :
            return 0.5
    def control(self):
        inKillZone =self.lidar().get_sensor_values() is None 

        if not inKillZone : 

            self.timestep_count += 1
            
            self.mapping(display=self.mapping_params.display_map)
            self.communication_management()

            # Retrieve Sensor Data
            found_wall, epsilon_wall_angle, min_dist = self.process_lidar_sensor(self.lidar())
            found_wounded, found_rescue_center, score_wounded, epsilon_wounded, epsilon_rescue_center, is_near_rescue_center,min_dist_wnd = self.process_semantic_sensor()

            is_near_rescuing_drone = self.check_near_rescuing_drone(threshold=30.0)
            if is_near_rescuing_drone:
                print("Hampering a rescue, waiting...")

            # TRANSITIONS OF THE STATE
            self.state_update(found_wall, found_wounded, found_rescue_center, is_near_rescuing_drone)

            # Execute Corresponding Command
            state_handlers = {
                self.State.WAITING: self.handle_waiting,
                self.State.SEARCHING_WALL: self.handle_searching_wall,
                self.State.FOLLOWING_WALL: lambda: self.handle_following_wall(epsilon_wall_angle, min_dist),
                self.State.GRASPING_WOUNDED: lambda: self.handle_grasping_wounded(min_dist_wnd, epsilon_wounded),
                self.State.SEARCHING_RESCUE_CENTER: self.handle_searching_rescue_center,
                self.State.GOING_RESCUE_CENTER: lambda: self.handle_going_rescue_center(epsilon_rescue_center, is_near_rescue_center),
                self.State.EXPLORING_FRONTIERS: self.handle_exploring_frontiers,
            }

            print(self.identifier, self.state)

            self.visualise_actions()

            return state_handlers.get(self.state, self.handle_unknown_state)()
        
        else : 
            # Drone in KillZone. Or at least no lidar available
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

    def handle_waiting(self):
        self.reset_exploration_path_params()
        self.step_waiting_count += 1
        return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

    def handle_searching_wall(self):
        return {"forward": 0.5, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

    def handle_following_wall(self, epsilon_wall_angle, min_dist):
        epsilon_wall_angle = normalize_angle(epsilon_wall_angle)
        epsilon_wall_distance = min_dist - self.wall_following_params.dist_to_stay

        self.logging_variables({"epsilon_wall_angle": epsilon_wall_angle, "epsilon_wall_distance": epsilon_wall_distance})

        command = {"forward": self.wall_following_params.speed_following_wall, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        command = self.pid_controller(command, epsilon_wall_angle, self.pid_params.Kp_angle, self.pid_params.Kd_angle, self.pid_params.Ki_angle, self.past_ten_errors_angle, "rotation")
        command = self.pid_controller(command, epsilon_wall_distance, self.pid_params.Kp_distance, self.pid_params.Kd_distance, self.pid_params.Ki_distance, self.past_ten_errors_distance, "lateral")

        return command

    def handle_grasping_wounded(self, score_wounded, epsilon_wounded):
        epsilon_wounded = normalize_angle(epsilon_wounded)
        print(self.identifier,score_wounded)
        command = {"forward": self.grasping_params.grasping_speed, "lateral": 0.0, "rotation": 0.0, "grasper": 1 if score_wounded<GraspingParams().grasping_dist else 0}
        return self.pid_controller(command, epsilon_wounded, self.pid_params.Kp_angle, self.pid_params.Kd_angle, self.pid_params.Ki_angle, self.past_ten_errors_angle, "rotation")

    def handle_searching_rescue_center(self):
        if self.previous_state is not self.State.SEARCHING_RESCUE_CENTER:
            self.plan_path_to_rescue_center()
        return self.follow_path(self.path, found_and_near_wounded=True)

    def plan_path_to_rescue_center(self):
        start_cell = self.grid._conv_world_to_grid(*self.estimated_pose.position)
        target_cell = self.grid.initial_cell
        max_inflation = self.path_params.max_inflation_obstacle
        self.path = self.grid.compute_safest_path(start_cell, target_cell, max_inflation)
        self.indice_current_waypoint = 0

    def handle_going_rescue_center(self, epsilon_rescue_center, is_near_rescue_center):
        epsilon_rescue_center = normalize_angle(epsilon_rescue_center)
        command = {"forward": 3 * self.grasping_params.grasping_speed, "lateral": 0.0, "rotation": 0.0, "grasper": 1}
        command = self.pid_controller(command, epsilon_rescue_center, self.pid_params.Kp_angle, self.pid_params.Kd_angle, self.pid_params.Ki_angle, self.past_ten_errors_angle, "rotation")

        if is_near_rescue_center:
            command["forward"] = 0.0
            command["rotation"] = 1.0  # Rotate in place to drop off

        return command

    def handle_exploring_frontiers(self):
        if self.finished_path:
            self.plan_path_to_frontier()
            self.finished_path = False

        if self.explored_all_frontiers or self.path is None:
            return self.handle_waiting()
        else:
            return self.follow_path(self.path, found_and_near_wounded=False)

    def assign_frontier_cluster(self):
        """
        Utilise DBSCAN pour regrouper les points frontaliers et assigne
        les clusters aux drones via l'algorithme hongrois.
        Si le nombre de clusters est insuffisant, les drones non affectés
        se voient attribuer le cluster ayant le coût minimal.
        Retourne le cluster assigné à ce drone, ou None si aucun cluster n'est disponible.
        """
        # 1. Clusterisation des points frontaliers
        clusters = self.grid.cluster_frontiers_dbscan(eps=2, min_samples=3)
        if not clusters:
            return None

        # 2. Récupérer les positions de tous les drones (via messages broadcast)
        drone_positions = {}
        # On ajoute la position du drone courant
        drone_positions[self.identifier] = np.array(self.estimated_pose.position)
        for drone_id, pos in self.other_drones_pos:
            drone_positions[drone_id] = np.array(pos)

        # 3. On s'assure d'un ordre cohérent des IDs
        drone_ids = sorted(drone_positions.keys())
        num_drones = len(drone_ids)
        num_clusters = len(clusters)

        # 4. Construction de la matrice de coût : [num_drones x num_clusters]
        cost_matrix = np.zeros((num_drones, num_clusters))
        for i, drone_id in enumerate(drone_ids):
            drone_pos = drone_positions[drone_id]
            for j, cluster in enumerate(clusters):
                centroid = cluster["centroid"]
                # Le coût est la distance euclidienne divisée par (taille du cluster + 1)
                cost_matrix[i, j] = np.linalg.norm(drone_pos - centroid) / (cluster["size"] + 1)

        # 5. Affectation via l'algorithme hongrois
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignments = {drone_ids[r]: clusters[col_ind[r]] for r in range(len(row_ind))}

        # 6. Pour les drones non affectés (si num_clusters < num_drones)
        if self.identifier not in assignments:
            # On choisit le cluster dont le coût est minimal pour ce drone
            drone_index = drone_ids.index(self.identifier)
            min_cluster_index = np.argmin(cost_matrix[drone_index, :])
            assignments[self.identifier] = clusters[min_cluster_index]

        return assignments[self.identifier]

    def plan_path_to_frontier(self):
        if self.grid.closest_largest_frontier(self.estimated_pose) is not None:
            self.next_frontier, self.next_frontier_centroid = self.grid.closest_largest_frontier(self.estimated_pose)
            if self.next_frontier_centroid is not None:
                start_cell = self.grid._conv_world_to_grid(*self.estimated_pose.position)
                target_cell = self.next_frontier_centroid
                max_inflation = self.path_params.max_inflation_obstacle

                self.path = self.grid.compute_safest_path(start_cell, target_cell, max_inflation)
                print(self.path)
                if self.path is None:   # The frontier is unreachable, probably due to artifacts of FREE zones inside boxes set in the mapping process
                    print(self.next_frontier.cells)
                    #self.grid.delete_frontier_artifacts(self.next_frontier)
                else:
                    self.indice_current_waypoint = 0

        else:
            self.explored_all_frontiers = True

    def plan_path_to_frontier(self):
        assigned_cluster = self.assign_frontier_cluster()
        if assigned_cluster is not None:
            self.next_frontier_centroid = assigned_cluster["centroid"]
            start_cell = self.grid._conv_world_to_grid(*self.estimated_pose.position)
            target_cell = self.next_frontier_centroid
            max_inflation = self.path_params.max_inflation_obstacle
            self.path = self.grid.compute_safest_path(start_cell, target_cell, max_inflation)
            if self.path is None:
                print("Assigned frontier unreachable, deleting artifacts.")
                self.grid.delete_frontier_artifacts(self.next_frontier)
                #self.state = self.State.SEARCHING_WALL
            else:
                self.indice_current_waypoint = 0
        else:
            self.explored_all_frontiers = True
    
    def handle_unknown_state(self):
        raise ValueError("State not found")

    def check_near_rescuing_drone(self, threshold, messages=None):
        """
        Checks if any received broadcast message indicates a drone (other than self)
        is grasping a wounded and is closer than the given threshold.
        """

        for _,broadcast_loc in self.wounded_locked :
            distance = np.linalg.norm(np.array(self.estimated_pose.position) - np.array(broadcast_loc))
            if distance < threshold:
                print("Near a rescuing drone")
                return True
        return False


    def process_semantic_sensor(self):
        semantic_values = self.semantic_values()
        
        best_angle_wounded = 0
        best_angle_rescue_center = 0
        mindist = 1000
        found_wounded = False
        found_rescue_center = False
        is_near_rescue_center = False
        angles_list = []

        scores = []
        for data in semantic_values:
            if (data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER):
                found_rescue_center = True
                angles_list.append(data.angle)
                is_near_rescue_center = (data.distance < 30)
                best_angle_rescue_center = circular_mean(np.array(angles_list))
            
            # If the wounded person detected is held by nobody
            elif (data.entity_type ==
                    DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped):
                found_wounded = True
                v = (data.angle * data.angle) + \
                    (data.distance * data.distance / 10 ** 5)
                scores.append((v, data.angle, data.distance))


        filtered_scores = []
        for score in scores :
            conflict = False
            for wnd_locked in self.wounded_locked :
                dx = score[2] * math.cos(score[1] + self.estimated_pose.orientation)
                dy = score[2] * math.sin(score[1] + self.estimated_pose.orientation)
                detection_position = np.array(self.estimated_pose.position) + np.array([dx, dy])
                conflict = False
                if np.linalg.norm(detection_position - np.array(wnd_locked[1])) < 20.0 : # adjust threshold as needed
                    conflict = True
                    print("Conflict of wounded")
                    break
            if not conflict :
                filtered_scores.append(score)
        best_score = 10000
        for score in filtered_scores:
            if score[0] < best_score:
                best_score = score[0]
                best_angle_wounded = score[1]
                mindist = score[2]

        return found_wounded,found_rescue_center,best_score,best_angle_wounded,best_angle_rescue_center,is_near_rescue_center,mindist
    
    def process_lidar_sensor(self,self_lidar):
        """
        -> ( bool near_obstacle , float epsilon_wall_angle )
        where epsilon_wall_angle is the (counter-clockwise convention) angle made
        between the drone and the nearest wall - pi/2
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
        if min_dist < self.wall_following_params.dmax: # pourcentage de la vitesse je pense
            near_obstacle = True

        epsilon_wall_angle = angle_nearest_obstacle - np.pi/2

        return (near_obstacle,epsilon_wall_angle,min_dist)

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
            #print(f"WAYPOINT {self.indice_current_waypoint} REACH")
            return True
        return False

    def follow_path(self,path,found_and_near_wounded):
        if self.is_near_waypoint(path[self.indice_current_waypoint]):
            self.indice_current_waypoint += 1 # next point in path
            #print(f"Waypoint reached {self.indice_current_waypoint}")
            if self.indice_current_waypoint >= len(path):
                self.finished_path = True # NOT USE YET
                self.indice_current_waypoint = 0
                self.path = []
                self.path_grid = []
                return
        
        return self.go_to_waypoint(path[self.indice_current_waypoint][0],path[self.indice_current_waypoint][1],found_and_near_wounded)

    def go_to_waypoint(self,x,y,found_and_near_wounded):
        
        # ASSERVISSEMENT EN ANGLE
        dx = x - self.estimated_pose.position[0]
        dy = y - self.estimated_pose.position[1]
        epsilon = math.atan2(dy,dx) - self.estimated_pose.orientation
        epsilon = normalize_angle(epsilon)
        command_path = self.pid_controller({"forward": 1,"lateral": 0.0,"rotation": 0.0,"grasper": 1 if found_and_near_wounded else 0},epsilon,self.pid_params.Kp_angle_1,self.pid_params.Kd_angle_1,self.pid_params.Ki_angle,self.past_ten_errors_angle,"rotation",0.5)

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

    def state_update(self, found_wall, found_wounded, found_rescue_center, is_near_rescuing_drone):
        """
        A visualisation of the state machine is available at doc/Drone states
        """
        self.previous_state = self.state
        
        conditions = {
            "found_wall": found_wall,
            "lost_wall": not found_wall,
            "found_wounded": found_wounded,
            "holding_wounded": bool(self.base.grasper.grasped_entities),
            "lost_wounded": not found_wounded and not self.base.grasper.grasped_entities,
            "found_rescue_center": found_rescue_center,
            "lost_rescue_center": not self.base.grasper.grasped_entities,
            "no_frontiers_left": len(self.grid.frontiers) == 0,
            "waiting_time_over": self.step_waiting_count >= self.waiting_params.step_waiting,
            "is_near_rescuing_drone": is_near_rescuing_drone
        }

        STATE_TRANSITIONS = {
            self.State.WAITING: {
                "found_wounded": self.State.GRASPING_WOUNDED,
                "waiting_time_over": self.State.EXPLORING_FRONTIERS
            },
            self.State.GRASPING_WOUNDED: {
                "lost_wounded": self.State.WAITING,
                "holding_wounded": self.State.SEARCHING_RESCUE_CENTER
            },
            self.State.SEARCHING_RESCUE_CENTER: {
                "lost_rescue_center": self.State.WAITING,
                "found_rescue_center": self.State.GOING_RESCUE_CENTER
            },
            self.State.GOING_RESCUE_CENTER: {
                "lost_rescue_center": self.State.WAITING
            },
            self.State.EXPLORING_FRONTIERS: {
                "found_wounded": self.State.GRASPING_WOUNDED,
                "no_frontiers_left": self.State.FOLLOWING_WALL,
                "is_near_rescuing_drone": self.State.WAITING
            },
            self.State.SEARCHING_WALL: {
                "found_wounded": self.State.GRASPING_WOUNDED,
                "found_wall": self.State.FOLLOWING_WALL,
                "is_near_rescuing_drone": self.State.WAITING
            },
            self.State.FOLLOWING_WALL: {
                "found_wounded": self.State.GRASPING_WOUNDED,
                "lost_wall": self.State.SEARCHING_WALL,
                "is_near_rescuing_drone": self.State.WAITING
            }
        }

        for condition, next_state in STATE_TRANSITIONS.get(self.state, {}).items():
            if conditions[condition]:
                self.state = next_state
                break

        if self.state != self.previous_state and self.state == self.State.WAITING:
            self.step_waiting_count = 0

    
    def mapping(self, display = False):
        
        if self.timestep_count == 1: # first iterations
            print("Starting control")
            start_x, start_y = self.measured_gps_position() # never none ? 
            print(f"Initial position: {start_x}, {start_y}")
            self.grid.set_initial_cell(start_x, start_y)
        

        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                   self.measured_compass_angle(),self.odometer_values(),self.previous_position[-1],self.previous_orientation[-1],self.size_area)
        
        self.previous_position.append(self.estimated_pose.position)
        self.previous_orientation.append(self.estimated_pose.orientation)
        

        self.grid.update(pose=self.estimated_pose)
        
        if display and (self.timestep_count % 5 == 0):
             self.grid.display(self.grid.zoomed_grid,
                               self.estimated_pose,
                               title=f"Drone {self.identifier} zoomed occupancy grid")


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

    def draw_point(self,point, color=arcade.color.GO_GREEN):
        arcade.draw_circle_filled(point[0], point[1], 5, color)

    def draw_path(self, path):
        length = len(path)
        pt2 = None
        for ind_pt in range(length):
            pose = path[ind_pt]
            pt1 = pose + self._half_size_array
            # print(ind_pt, pt1, pt2)
            if ind_pt > 0:
                arcade.draw_line(float(pt2[0]),
                                 float(pt2[1]),
                                 float(pt1[0]),
                                 float(pt1[1]), [125,125,125])
            pt2 = pt1

    def draw_top_layer(self):
        if self.visualisation_params.draw_path:
            self.draw_path(self.path)

        if self.state == self.State.EXPLORING_FRONTIERS:
            if self.visualisation_params.draw_frontier_centroid and self.next_frontier_centroid is not None:
                self.draw_point(self.grid._conv_grid_to_world(*self.next_frontier_centroid) + self._half_size_array)     # frame of reference change
            
            if self.visualisation_params.draw_frontier_points and self.next_frontier is not None:
                for point in self.next_frontier.cells:
                    self.draw_point(self.grid._conv_grid_to_world(*point) + self._half_size_array, color=arcade.color.AIR_FORCE_BLUE)     # frame of reference change

    def visualise_actions(self):
        """
        It's mandatory to use draw_top_layer to draw anything on the interface
        """
        self.draw_top_layer()