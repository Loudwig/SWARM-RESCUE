"""
The drone explores the map following frontiers between explored an unexplored areas.
"""

from enum import Enum, auto
from collections import deque
import math
from typing import Optional
import numpy as np
import arcade
from scipy.optimize import linear_sum_assignment

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.utils import circular_mean, normalize_angle
from solutions.utils.pose import Pose
from solutions.utils.astar import *
from solutions.utils.messages import DroneMessage
from solutions.utils.grids import *
from solutions.utils.dataclasses_config import *

from solutions.utils.astar import can_go_straight
from solutions.utils.dataclasses_config import WallFollowingParams, WaitingStateParams


class MyDroneFrontex(DroneAbstract):
    class State(Enum):
        """
        All the states of the drone as a state machine
        """
        WAITING = auto()        # Assigns 1
        DEADLOCK = auto()       # Assigns 2 etc ... This allows to easily add new states
        WAITING_DEPARTURE = auto()

        SEARCHING_WALL = auto()
        FOLLOWING_WALL = auto()

        EXPLORING_FRONTIERS = auto()

        GRASPING_WOUNDED = auto()
        SEARCHING_RESCUE_CENTER = auto()
        GOING_RESCUE_CENTER = auto()
        GRASPING_FOLLOWING_WALL = auto()

        SEARCHING_RETURN_AREA = auto()
        GOING_RETURN_AREA = auto()

        STOP = auto()

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         **kwargs)
        
        # MAPPING
        self.estimated_pose = Pose()
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=MappingParams.resolution,
                                  lidar=self.lidar(),semantic=self.semantic())

        # POSITION
        self.previous_position = deque(maxlen=1) 
        self.previous_position.append((0,0))  
        self.previous_orientation = deque(maxlen=1) 
        self.previous_orientation.append(0)
        self.prev_diff_position = 0
        self.history_forward_epsilon = deque(maxlen=50)

        # STATE INITIALISATION
        self.state  = self.State.WAITING_DEPARTURE
        self.previous_state = self.State.WAITING_DEPARTURE
        self.timestep_count = 0
        
        # PARAMS FOR DIFFERENT STATES 

            # WAITING STATE
        self.step_waiting_count = 0
        self.step_deadlock_count = 0

            # WAITING DEPARTURE STATE
        self.all_drones_departure_score = []
        self.departure = False
        self.own_departure_timestep = 0
        self.interval_departure = int(WaitingDepartureStateParams.interval_departure*self._misc_data.max_timestep_limit/self._misc_data.number_drones)

            # FRONTIER EXPLORATION
        self.explored_all_frontiers = False
        self.next_frontier = None
        self.next_frontier_centroid = None
        self.did_not_find_path = False

            # GOING RESCUE CENTER
        self.group_barycenter = None

        self.state_lock_counter = 0

        # PATH FOLLOWING
        self.indice_current_waypoint = 0
        self.inital_point_path = (0,0)
        self.finished_path = True
        self.path = []
        self.path_grid = []

        self.wounded_locked = []
        self.other_drones_pos = []

        self.history_health = deque(maxlen=50)
        self.history_path_searching = deque([False]*10, maxlen=10)

        self.counter_static = 0
        self.last_position = None
        self.unexplored_point = None

    def reset_exploration_path_params(self):
        """
        Resets the parameters related to the exploration path.
        """
        self.next_frontier = None
        self.next_frontier_centroid = None
        self.finished_path = True
        self.path = []
        self.did_not_find_path = False

    def define_message_for_all(self):
        inKillZone =self.lidar().get_sensor_values() is None or self._drone_health<=0 or self.state == self.State.STOP
        message = []
        if self.timestep_count<=1 or inKillZone:
            return None

        if self.timestep_count % CommunicationParams.TIME_INTERVAL == 0:
            confidence = self.compute_confidence(self.estimated_pose.gps)
            message.append(DroneMessage(subject=DroneMessage.Subject.MAPPING, arg={"map": self.grid.grid, "confidence": confidence}))
        else :
            message.append(DroneMessage(subject=DroneMessage.Subject.PASS, arg=None))

        if self.state == self.State.GRASPING_WOUNDED or self.state == self.State.SEARCHING_RESCUE_CENTER or self.state == self.State.GOING_RESCUE_CENTER:
            broadcast_msg = DroneMessage(
                subject=DroneMessage.Subject.LOCK_WOUNDED,
                arg=(self.identifier, self.estimated_pose.position.tolist())
            )
            message.append(broadcast_msg)

        if self.state == self.State.WAITING_DEPARTURE:
            broadcast_msg = DroneMessage(
                subject=DroneMessage.Subject.DEPARTURE,
                arg=(self.identifier, self.compute_departure_score())
            )
            message.append(broadcast_msg)

        loc_msg = DroneMessage(
            subject=DroneMessage.Subject.FRONTIER_PRIO,
            arg=(self.identifier, self.estimated_pose.position.tolist())
        )
        message.append(loc_msg)

        return message

    def communication_management(self):
        self.wounded_locked = []
        self.other_drones_pos = []
        self.all_drones_departure_score = []
        if self.communicator:
            received_messages = self.communicator.received_messages
            for msg in received_messages:
                for drone_msg in msg[1]:
                    if not isinstance(drone_msg, DroneMessage):
                        raise ValueError("Invalid message type. Expected a DroneMessage instance.")
                    if drone_msg.subject == DroneMessage.Subject.MAPPING :
                        self.grid.merge_maps(drone_msg.arg["map"],drone_msg.arg["confidence"])
                    if drone_msg.subject == DroneMessage.Subject.LOCK_WOUNDED:
                        drone_id, position = drone_msg.arg
                        self.wounded_locked.append((drone_id, position))
                    if drone_msg.subject == DroneMessage.Subject.FRONTIER_PRIO:
                        drone_id, position = drone_msg.arg
                        self.other_drones_pos.append((drone_id,position))
                    if drone_msg.subject == DroneMessage.Subject.DEPARTURE:
                        drone_id, score = drone_msg.arg
                        self.all_drones_departure_score.append(score)

    def compute_confidence(self, gps):
        if gps is None:
            return 0.1
        else :
            return 0.5
    
    def control(self):
        self.timestep_count += 1
        inKillZone =self.lidar().get_sensor_values() is None or self._drone_health<=0

        if not inKillZone :
            self.history_health.append(self.drone_health)
            self.history_forward_epsilon.append(self.odometer_values()[0] * np.cos(self.odometer_values()[1]))
            self.history_path_searching.append(False)
            self.mapping(display=MappingParams.display_map)
            self.communication_management()

            # Retrieve Sensor Data
            found_wall, epsilon_wall_angle, min_dist = self.process_lidar_sensor(self.lidar())
            found_wounded, found_rescue_center, score_wounded, epsilon_wounded, epsilon_rescue_center, is_near_rescue_center,is_very_near_rescue_center, min_dist_wnd = self.process_semantic_sensor()

            is_near_rescuing_drone = self.check_near_rescuing_drone(threshold=GraspingParams.hampering_dist)
            if is_near_rescuing_drone:
                pass

            must_return = self.must_return_area()

            # TRANSITIONS OF THE STATE
            if self.state_lock_counter <= 0:
                self.state_update(found_wall, found_wounded, found_rescue_center, is_near_rescuing_drone, must_return)

            # Execute Corresponding Command
            state_handlers = {
                self.State.WAITING: self.handle_waiting,
                self.State.DEADLOCK: self.handle_deadlock,
                self.State.WAITING_DEPARTURE: self.handle_waiting_departure,
                self.State.SEARCHING_WALL: self.handle_searching_wall,
                self.State.FOLLOWING_WALL: lambda: self.handle_following_wall(epsilon_wall_angle, min_dist),
                self.State.FOLLOWING_WALL: lambda: self.handle_following_wall(epsilon_wall_angle, min_dist),
                self.State.GRASPING_WOUNDED: lambda: self.handle_grasping_wounded(min_dist_wnd, epsilon_wounded),
                self.State.SEARCHING_RESCUE_CENTER: self.handle_searching_rescue_center,
                self.State.GOING_RESCUE_CENTER: lambda: self.handle_going_rescue_center(epsilon_rescue_center, is_very_near_rescue_center),
                self.State.GRASPING_FOLLOWING_WALL: lambda: self.handle_grasping_following_wall(epsilon_wall_angle, min_dist),
                self.State.EXPLORING_FRONTIERS: lambda: self.handle_exploring_frontiers(is_near_rescue_center),
                self.State.STOP: lambda: {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0},
            }

            self.visualise_actions()

            if self.state == self.State.STOP:
                pass

            self.last_position = self.grid._conv_world_to_grid(*self.estimated_pose.position)

            return state_handlers.get(self.state, self.handle_unknown_state)()
        
        else : 
            # Drone in KillZone. Or at least no lidar available
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

    def handle_waiting(self):
        self.reset_exploration_path_params()
        self.step_waiting_count += 1
        return {"forward": 0.0, "lateral": 0.0, "rotation": np.random.random(), "grasper": 0}

    def handle_deadlock(self):
        self.step_deadlock_count += 1
        print("DEADLOCK")
        return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

    def compute_departure_score(self):
        """
        Drones that have the highest departure scores can start exploring first.
        """
        other_drone_pos = self.other_drones_pos
        if len(other_drone_pos) == 0:
            return 0

        score = np.sum(self.lidar().get_sensor_values())
        return int(score)

    def barycenter(self, all_drone_positions):
        positions = [np.array(pos) for _, pos in all_drone_positions]
        positions_array = np.array(positions)
        return self.grid._conv_world_to_grid(*np.mean(positions_array, axis=0))

    def handle_waiting_departure(self):
        self.all_drones_departure_score.append(self.compute_departure_score())
        if self.timestep_count % self.interval_departure == 10:
            size_drone_group = WaitingDepartureStateParams.size_drone_group
            if self.compute_departure_score() in sorted(self.all_drones_departure_score, reverse=True)[:size_drone_group]:
                self.departure = True
                self.own_departure_timestep = self.elapsed_timestep

        if self.group_barycenter is None:
            if len(self.other_drones_pos) != 0:
                self.group_barycenter = self.barycenter(self.other_drones_pos + [(self.identifier,self.estimated_pose.position)])

        return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

    def handle_searching_wall(self):
        return {"forward": 0.5, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

    def handle_following_wall(self, epsilon_wall_angle, min_dist):
        self.reset_exploration_path_params()
        epsilon_wall_angle = normalize_angle(epsilon_wall_angle)
        epsilon_wall_distance = min_dist - WallFollowingParams.dist_to_stay

        command = {"forward": WallFollowingParams.speed_following_wall, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        command = self.pid_controller(command, epsilon_wall_angle, PIDParams.Kp_angle, PIDParams.Kd_angle, "rotation")
        command = self.pid_controller(command, epsilon_wall_distance, PIDParams.Kp_distance, PIDParams.Kd_distance, "lateral")

        return command

    def handle_grasping_wounded(self, score_wounded, epsilon_wounded):
        self.reset_exploration_path_params()
        epsilon_wounded = normalize_angle(epsilon_wounded)
        command = {"forward": GraspingParams.grasping_speed, "lateral": 0.0, "rotation": 0.0, "grasper": 1 if score_wounded<GraspingParams.grasping_dist else 0}
        return self.pid_controller(command, epsilon_wounded, PIDParams.Kp_angle, PIDParams.Kd_angle,"rotation")

    def handle_searching_rescue_center(self):
        if self.previous_state is not self.State.SEARCHING_RESCUE_CENTER:
            self.plan_path_to_rescue_center()

        command = self.follow_path(self.path, found_and_near_wounded=True)
        return command

    def plan_path_to_rescue_center(self):
        start_cell = self.grid._conv_world_to_grid(*self.estimated_pose.position)
        target_cell = self.grid.initial_cell
        max_inflation = PathParams.max_inflation_grasping if bool(self.base.grasper.grasped_entities) else PathParams.max_inflation_obstacle
        if bool(self.estimated_pose.gps):
            self.path = self.grid.compute_safest_path(start_cell, target_cell, max_inflation)
            self.indice_current_waypoint = 0
            self.history_path_searching.pop()
            self.history_path_searching.append(True)
        else:
            self.path = None

    def handle_grasping_following_wall(self, epsilon_wall_angle, min_dist):
        command = self.handle_following_wall(epsilon_wall_angle, min_dist)
        command["grasper"] = 1
        return command

    def handle_going_rescue_center(self, epsilon_rescue_center, is_very_near):
        epsilon_rescue_center = normalize_angle(epsilon_rescue_center)
        command = {"forward":  1.0, "lateral": 0.0, "rotation": 0.0, "grasper": 1}
        command = self.pid_controller(command, epsilon_rescue_center, PIDParams.Kp_angle, PIDParams.Kd_angle,"rotation")

        if is_very_near:
            command["forward"] = 0.0
            movement = math.dist(self.grid._conv_world_to_grid(*self.estimated_pose.position), self.last_position)
            if movement < 0.5:
                self.counter_static += 1
            else:
                self.counter_static = 0

            if self.counter_static > 40 and self.state_lock_counter==0:
                self.state_lock_counter = 20
            else:
                command["forward"] = -0.3
                command["grasper"] = 0
                self.state_lock_counter -= 1
        else:
            self.state_lock_counter = 0

        return command

    def handle_exploring_frontiers(self,is_near_rescue_center):
        if self.finished_path:
            self.plan_path_to_frontier(is_near_rescue_center)
            self.finished_path = False

        if self.explored_all_frontiers or self.path is None:
            return self.handle_waiting()

        else:
            return self.follow_path(self.path, found_and_near_wounded=False)

    def assign_frontier_cluster(self, is_near_rescue_center):
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


        if is_near_rescue_center :
            if self.elapsed_timestep/self._misc_data.max_timestep_limit > 0.5:

                # Définir le threshold pour considérer une cellule non explorée
                threshold = -1.6
                ternary_map = self.grid.to_ternary_map()
                grid_shape = ternary_map.shape

                # Éviter les bords pour la sélection des points aléatoires
                # Garantir qu'on peut toujours prendre un cube 3x3 autour
                padding = 1  # Pour le voisinage 3x3

                # Trouver un point non exploré
                max_attempts = 20
                self.unexplored_point = None

                for attempt in range(max_attempts):
                    # Sélectionner un point aléatoire avec des marges sécurisées
                    random_x = np.random.randint(padding, grid_shape[0]-padding)
                    random_y = np.random.randint(padding, grid_shape[1]-padding)

                    # Vérifier le voisinage 3x3
                    neighborhood = ternary_map[random_x-1:random_x+2, random_y-1:random_y+2]
                    avg_abs_value = np.mean(neighborhood)
                    # Si la valeur moyenne est sous le seuil, considérer comme non exploré
                    if avg_abs_value < threshold:
                        unexplored_point = (random_x, random_y)
                        self.unexplored_point = unexplored_point
                        break





        for i, drone_id in enumerate(drone_ids):
            drone_pos = drone_positions[drone_id]
            for j, cluster in enumerate(clusters):
                cell_centroid = cluster.point_closest_to_centroid()
                if self.timestep_count - self.own_departure_timestep < self.interval_departure:
                    cost_matrix[i,j] = 1/math.dist(self.grid.initial_cell, cell_centroid)
                else :
                    if is_near_rescue_center:
                        if self.unexplored_point is not None:
                            cost_matrix[i, j] = math.dist(cell_centroid, self.unexplored_point)
                        else :
                            cost_matrix[i, j] = math.dist(self.grid._conv_world_to_grid(*drone_pos), cell_centroid)
                    else :
                        own_cell = self.grid._conv_world_to_grid(*drone_pos)
                        target_cell = self.grid._conv_world_to_grid(*cell_centroid)
                        if can_go_straight(*own_cell, *target_cell, self.grid.to_ternary_map()):
                            cost_matrix[i, j] = math.dist(self.grid._conv_world_to_grid(*drone_pos), cell_centroid)
                        else:
                            cost_matrix[i, j] = math.dist(self.grid._conv_world_to_grid(*drone_pos), cell_centroid)*10**3

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

    def plan_path_to_frontier(self, is_near_rescue_center):
        assigned_cluster = self.assign_frontier_cluster(is_near_rescue_center)
        if assigned_cluster is not None:
            self.next_frontier = assigned_cluster
            self.next_frontier_centroid = assigned_cluster.point_closest_to_centroid()
            start_cell = self.grid._conv_world_to_grid(*self.estimated_pose.position)
            target_cell = self.next_frontier_centroid
            max_inflation = PathParams.max_inflation_obstacle
            self.path = self.grid.compute_safest_path(start_cell, target_cell, max_inflation)
            self.history_path_searching.pop()
            self.history_path_searching.append(True)
            if self.path is None:
                self.did_not_find_path = True
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
            own_cell = self.grid._conv_world_to_grid(*self.estimated_pose.position)
            other_cell = self.grid._conv_world_to_grid(*broadcast_loc)
            distance = math.dist(own_cell,other_cell)
            if distance < threshold and can_go_straight(*own_cell,*other_cell,self.grid.to_ternary_map()):
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
        is_very_near_rescue_center = False
        angles_list = []

        scores = []
        for data in semantic_values:
            if (data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER):
                found_rescue_center = True
                angles_list.append(data.angle)
                if data.distance < 45.0:
                    is_very_near_rescue_center = True
                if data.distance < 80.0:
                    is_near_rescue_center = True
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
                if np.linalg.norm(detection_position - np.array(wnd_locked[1])) < GraspingParams.hampering_dist :
                    conflict = True
                    break
            if not conflict :
                filtered_scores.append(score)
        best_score = 10000
        for score in filtered_scores:
            if score[0] < best_score:
                best_score = score[0]
                best_angle_wounded = score[1]
                mindist = score[2]

        return found_wounded,found_rescue_center,best_score,best_angle_wounded,best_angle_rescue_center,is_near_rescue_center,is_very_near_rescue_center,mindist
    
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
        if min_dist < WallFollowingParams.dmax:
            near_obstacle = True

        epsilon_wall_angle = angle_nearest_obstacle - np.pi/2

        return (near_obstacle,epsilon_wall_angle,min_dist)

    def pid_controller(self,command,epsilon,Kp,Kd,mode,command_slow = 0.8,grasping=False):
        if mode == "rotation":
            epsilon = normalize_angle(epsilon)
            deriv_epsilon = normalize_angle(self.odometer_values()[2])
        elif mode == "lateral":
            deriv_epsilon = -np.sin(self.odometer_values()[1])*self.odometer_values()[0] # vitesse latérale
        elif mode == "forward" : 
            deriv_epsilon = epsilon - self.prev_diff_position
            self.prev_diff_position = epsilon
            
        else : 
            raise ValueError("Mode not found")
        
        correction_proportionnelle = Kp * epsilon
        correction_derivee = Kd * deriv_epsilon
        correction = correction_proportionnelle + correction_derivee

        if grasping and mode == "forward":
            correction = 3*correction

        command[mode] = correction
        command[mode] = min( max(-1,correction) , 1 )


        if mode == "rotation" : 
            if correction > command_slow :
                command["forward"] = WallFollowingParams.speed_turning

        return command
    
    def is_near_waypoint(self,waypoint):
        distance_to_waypoint = np.linalg.norm(waypoint - self.estimated_pose.position)
        if distance_to_waypoint < PathParams.distance_close_waypoint:
            return True
        return False

    def follow_path(self,path,found_and_near_wounded):
        if path is None :
            self.finished_path = True
            self.indice_current_waypoint = 0
            self.path = []
            self.path_grid = []
            return 
        else : 
            if self.is_near_waypoint(path[self.indice_current_waypoint]):
                self.indice_current_waypoint += 1
                if self.indice_current_waypoint >= len(path):
                    self.finished_path = True
                    self.indice_current_waypoint = 0
                    self.path = []
                    self.path_grid = []
                    return
            
            return self.go_to_waypoint(path[self.indice_current_waypoint][0],path[self.indice_current_waypoint][1],found_and_near_wounded)

    def go_to_waypoint(self,x,y,found_and_near_wounded):
        dx = x - self.estimated_pose.position[0]
        dy = y - self.estimated_pose.position[1]
        epsilon = math.atan2(dy,dx) - self.estimated_pose.orientation
        epsilon = normalize_angle(epsilon)
        command_path = self.pid_controller({"forward": 0,"lateral": 0.0,"rotation": 0.0,"grasper": 1 if found_and_near_wounded else 0},epsilon,PIDParams.Kp_angle_1,PIDParams.Kd_angle_1,"rotation",0.5)

        if self.indice_current_waypoint == 0:
            x_previous_waypoint,y_previous_waypoint = self.inital_point_path
        else : 
            x_previous_waypoint,y_previous_waypoint = self.path[self.indice_current_waypoint-1][0],self.path[self.indice_current_waypoint-1][1]

        epsilon_distance = compute_relative_distance_to_droite(x_previous_waypoint,y_previous_waypoint,x,y,self.estimated_pose.position[0],self.estimated_pose.position[1])
        # epsilon distance needs to be signed (positive if the angle relative to the theoritical path is positive)
        command_path = self.pid_controller(command_path,epsilon_distance,PIDParams.Kp_distance_1,PIDParams.Kd_distance_1,"lateral",0.5)

        diff_position = math.dist(np.array([x,y]), self.estimated_pose.position)

        command_path = self.pid_controller(command_path,diff_position,PIDParams.Kp_distance_2,PIDParams.Kd_distance_2,"forward",1,found_and_near_wounded)

        return command_path

    def must_return_area(self):
        if not(bool(self.estimated_pose.gps)):
            return False
        return self.history_health[-1] < HealthParams.THRESHOLD_HEALTH or self.elapsed_timestep / self._misc_data.max_timestep_limit > HealthParams.THRESHOLD_TIMESTEP

    def state_update(self, found_wall, found_wounded, found_rescue_center, is_near_rescuing_drone, must_return):
        """
        A visualisation of the state machine is available at doc/Drone states
        """
        self.previous_state = self.state
        
        conditions = {
            "departure": self.departure,
            "must_return": must_return,
            "is_and_must_inside_return": self.is_inside_return_area and must_return and (not bool(self.base.grasper.grasped_entities)),
            "no_longer_inside_return": not self.is_inside_return_area and must_return and (not bool(self.base.grasper.grasped_entities)),
            "try_searching_rescue_center" : self.timestep_count%200==0 and bool(self.estimated_pose.gps),
            "gps": bool(self.estimated_pose.gps),
            "no_gps" : not(bool(self.estimated_pose.gps)),
            "found_wall": found_wall,
            "lost_wall": not found_wall,
            "found_wounded": found_wounded,
            "must_return_grasping_available": must_return and (not bool(self.base.grasper.grasped_entities)) and found_wounded,
            "holding_wounded": bool(self.base.grasper.grasped_entities),
            "lost_wounded": not found_wounded and not self.base.grasper.grasped_entities,
            "found_rescue_center": found_rescue_center and not (must_return and not bool(self.base.grasper.grasped_entities)),
            "lost_rescue_center": not self.base.grasper.grasped_entities and not must_return,
            "no_frontiers_left": len(self.grid.frontiers) == 0,
            "waiting_time_over": self.step_waiting_count >= WaitingStateParams.step_waiting,
            "deadlock_time_over": self.step_deadlock_count >= WaitingStateParams.step_deadlock,
            "is_near_rescuing_drone": is_near_rescuing_drone,
            "health_decreasing" : np.sum(np.diff(np.array(self.history_health)) < 0)>1,
            "did_not_find_path": self.did_not_find_path,
            "no_path_to_rescue_center": self.path is None or len(self.path) == 0,
            "too_much_path_searching": sum(self.history_path_searching) >= 3,
            "immobile_no_gps": np.mean(np.array(self.history_forward_epsilon)) < 0.5
        }

        STATE_TRANSITIONS = {
            self.State.WAITING_DEPARTURE:{
                "departure": self.State.WAITING
            },
            self.State.STOP: {
                "no_longer_inside_return": self.State.SEARCHING_RESCUE_CENTER
            },
            self.State.WAITING: {
                "found_wounded": self.State.GRASPING_WOUNDED,
                "is_and_must_inside_return": self.State.STOP,
                "waiting_time_over": self.State.EXPLORING_FRONTIERS
            },
            self.State.DEADLOCK: {
                "deadlock_time_over": self.State.WAITING
            },
            self.State.GRASPING_WOUNDED: {
                "lost_wounded": self.State.WAITING,
                "holding_wounded": self.State.SEARCHING_RESCUE_CENTER
            },
            self.State.SEARCHING_RESCUE_CENTER: {
                "too_much_path_searching": self.State.DEADLOCK,
                "no_path_to_rescue_center": self.State.GRASPING_FOLLOWING_WALL,
                "must_return_grasping_available": self.State.GRASPING_WOUNDED,
                "is_and_must_inside_return": self.State.STOP,
                "lost_rescue_center": self.State.WAITING,
                "found_rescue_center": self.State.GOING_RESCUE_CENTER
            },
            self.State.GOING_RESCUE_CENTER: {
                "is_and_must_inside_return": self.State.STOP,
                "no_longer_inside_return": self.State.SEARCHING_RESCUE_CENTER,
                "lost_rescue_center": self.State.WAITING
            },
            self.State.GRASPING_FOLLOWING_WALL:{
                "try_searching_rescue_center": self.State.SEARCHING_RESCUE_CENTER,
                "lost_wounded": self.State.FOLLOWING_WALL,
                "immobile_no_gps": self.State.WAITING
            },
            self.State.EXPLORING_FRONTIERS: {
                "found_wounded": self.State.GRASPING_WOUNDED,
                "is_and_must_inside_return": self.State.STOP,
                "must_return": self.State.SEARCHING_RESCUE_CENTER,
                "health_decreasing": self.State.WAITING,
                "no_gps" : self.State.FOLLOWING_WALL,
                "did_not_find_path": self.State.FOLLOWING_WALL,
                "no_frontiers_left": self.State.SEARCHING_RESCUE_CENTER,
                "is_near_rescuing_drone": self.State.WAITING
            },
            self.State.SEARCHING_WALL: {
                "is_and_must_inside_return": self.State.STOP,
                "must_return": self.State.SEARCHING_RESCUE_CENTER,
                "found_wounded": self.State.GRASPING_WOUNDED,
                "health_decreasing": self.State.WAITING,
                "found_wall": self.State.FOLLOWING_WALL,
                "is_near_rescuing_drone": self.State.WAITING
            },
            self.State.FOLLOWING_WALL: {
                "is_and_must_inside_return": self.State.STOP,
                "must_return": self.State.SEARCHING_RESCUE_CENTER,
                "found_wounded": self.State.GRASPING_WOUNDED,
                "health_decreasing": self.State.WAITING,
                "gps": self.State.WAITING,
                "lost_wall": self.State.SEARCHING_WALL,
                "is_near_rescuing_drone": self.State.WAITING
            }
        }

        for condition, next_state in STATE_TRANSITIONS.get(self.state, {}).items():
            if conditions[condition]:
                self.state = next_state
                break

        if self.state != self.previous_state:
                if self.state == self.State.WAITING:
                    self.step_waiting_count = 0
                if self.state == self.State.DEADLOCK:
                    self.step_deadlock_count = 0
    
    def mapping(self, display = False):
        
        if self.timestep_count == 1: # first iterations
            start_x, start_y = self.measured_gps_position() # never none ?
            self.grid.set_initial_cell(start_x, start_y)
            self.last_position = self.grid.initial_cell

        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                   self.measured_compass_angle(),self.odometer_values(),self.previous_position[-1],self.previous_orientation[-1],self.size_area)
        
        self.previous_position.append(self.estimated_pose.position)
        self.previous_orientation.append(self.estimated_pose.orientation)
        
        if self.estimated_pose.gps : 
            self.grid.update(pose=self.estimated_pose)
        
        if display and (self.timestep_count % 5 == 0):
             self.grid.display(self.grid.to_ternary_map(),
                               self.estimated_pose,
                               title=f"Drone {self.identifier} ternary map grid")



    def draw_point(self,point, color=arcade.color.GO_GREEN):
        arcade.draw_circle_filled(point[0], point[1], 5, color)

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
        if VisualisationParams.draw_path:
            self.draw_path(self.path)



        if self.state == self.State.EXPLORING_FRONTIERS:
            
            if VisualisationParams.draw_frontier_points and self.next_frontier is not None:
                colors = [arcade.color.RED, arcade.color.BLUE, arcade.color.GREEN, arcade.color.YELLOW, arcade.color.ORANGE, arcade.color.PURPLE]
                for i,f in enumerate(self.grid.frontiers):
                    for cell in f.cells :
                        point = self.grid._conv_grid_to_world(*cell) + self._half_size_array
                        self.draw_point(point, color=colors[i % len(colors)])

            if VisualisationParams.draw_frontier_centroid and self.next_frontier_centroid is not None:
                self.draw_point(self.grid._conv_grid_to_world(*self.next_frontier_centroid) + self._half_size_array)     # frame of reference change

            if self.unexplored_point is not None:
                self.draw_point(self.grid._conv_grid_to_world(*self.unexplored_point) + self._half_size_array, color=arcade.color.YELLOW)

    def visualise_actions(self):
        """
        It's mandatory to use draw_top_layer to draw anything on the interface
        """
        self.draw_top_layer()



    
