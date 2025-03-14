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

class MyDroneFrontex(DroneAbstract):
    """
    A class of drone that can explore the environment,communicate and rescue wounded people.
    The drone is always in one of its predefinded states. The transistions are managed by a state machine.
    Drone positions in this class are in the form of numpy arrays and always in the world frame.
    """
    class State(Enum):
        """
        All the states of the drone as a state machine
        """
        WAITING_DEPARTURE = auto()      # Assigns 1
        WAITING = auto()        # Assigns 2 etc ... This allows to easily add new states

        EXPLORING_FRONTIERS = auto()

        SEARCHING_WALL = auto()
        FOLLOWING_WALL = auto()
        GRASPING_FOLLOWING_WALL = auto()

        GRASPING_WOUNDED = auto()
        SEARCHING_RESCUE_CENTER = auto()
        GOING_RESCUE_CENTER = auto()

        SEARCHING_RETURN_AREA = auto()
        GOING_RETURN_AREA = auto()

        STOP = auto()
        DEADLOCK = auto()

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         **kwargs)
        
        # MAPPING
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=MappingParams.resolution,
                                  lidar=self.lidar(),semantic=self.semantic())

        # POSITION
        self.estimated_pose = Pose()
        self.initial_pos = np.array([0.0,0.0])
        self.previous_position = np.array([0.0,0.0])
        self.previous_orientation = 0.0
        self.prev_diff_position = 0
        self.history_forward_epsilon = deque(maxlen=50)

        # STATE INITIALISATION
        self.state  = self.State.WAITING_DEPARTURE
        self.previous_state = self.State.WAITING_DEPARTURE
        self.state_lock_counter = 0
        
        # PARAMS FOR DIFFERENT STATES 

            # WAITING STATE
        self.step_waiting_count = 0
        self.step_deadlock_count = 0

            # WAITING DEPARTURE STATE
        self.all_drones_departure_score = []
        self.departure = False
        self.interval_departure = int( WaitingDepartureStateParams.departure_time_rate * 
                                      self._misc_data.max_timestep_limit/self._misc_data.number_drones )
        self.own_departure_timestep = 0

            # FRONTIER EXPLORATION
        self.explored_all_frontiers = False
        self.next_frontier = None
        self.next_frontier_target_pos = None
        self.unexplored_point_incentive = None

            # GOING RESCUE CENTER
        self.initial_group_barycenter = None

            # WALL FOLLOWING
        self.found_obstacle = False
        self.distance_nearest_obstacle = 0.0
        self.angle_nearest_obstacle = 0.0

            # GRASPING WOUNDED
        self.found_wounded = False
        self.best_score = 0.0
        self.best_angle_wounded = 0.0
        self.best_angle_rescue_center = 0.0
        self.distance_nearest_wounded = 0.0
        self.found_rescue_center = False
        self.is_near_rescue_center = False

        # PATH FOLLOWING
        self.index_current_waypoint = 0
        self.previous_waypoint_position = np.array([0.0,0.0])
        self.next_waypoint_position = np.array([0.0,0.0])

        self.epsilon_angle = 0.0
        self.epsilon_lateral = 0.0
        self.epsilon_forward = 0.0
        self.history_epsilon_angle = deque([0.0] * 10, maxlen=10)
        self.history_epsilon_lateral = deque([0.0] * 10, maxlen=10)
        self.history_epsilon_forward = deque([0.0] * 10, maxlen=10)

        self.finished_path = True
        self.path = []
        self.history_path_searching = deque([False]*10, maxlen=10)

        # COMMUNICATION
        self.wounded_locked = []
        self.other_drones_pos = []
        self.is_near_rescuing_drone = False

        # MISCELLANEOUS
        self.history_health = deque(maxlen=50)

    def reset_exploration_path_infos(self):
        """
        After the drone has finished exploring a path or has unexpectedly changed state.
        """
        self.next_frontier = None
        self.next_frontier_target_pos = None
        self.finished_path = True
        self.path = []

    def set_new_path(self, start_pos, target_pos, max_inflation):
        self.path = self.grid.compute_safest_path(start_pos, target_pos, max_inflation)

        if self.path is not None:
            self.index_current_waypoint = 0
            self.previous_waypoint_position = self.path[0]
            if len(self.path) > 0:
                self.next_waypoint_position = self.path[1]
            else:
                self.next_waypoint_position = self.path[0]
            
            self.finished_path = False
            
        self.history_path_searching[-1] = True
    
    def reset_received_infos(self):
        """
        At each time step.
        """
        self.wounded_locked = []
        self.other_drones_pos = []
        self.all_drones_departure_score = []
    
    def reset_semantic_infos(self):
        self.best_angle_wounded = 0.0
        self.best_angle_rescue_center = 0.0
        self.found_wounded = False
        self.best_score = 0.0
        self.found_rescue_center = False
        self.is_near_rescue_center = False

    def reset_pid_history(self):
        self.history_epsilon_angle = deque([0.0] * 10, maxlen=10)
        self.history_epsilon_lateral = deque([0.0] * 10, maxlen=10)
        self.history_epsilon_forward = deque([0.0] * 10, maxlen=10)
    
    def is_killed(self):
        return self.lidar().get_sensor_values() is None or self._drone_health<=0

    def define_message_for_all(self):
        message = []

        if self.elapsed_timestep<=1 or self.is_killed():
            return None

        if self.elapsed_timestep % CommunicationParams.GRID_SHARE_TIME_INTERVAL == 0:
            confidence = self.compute_confidence()
            grid_msg = DroneMessage(
                subject=DroneMessage.Subject.MAPPING,
                arg={"map": self.grid.grid, "confidence": confidence})
            message.append(grid_msg)

        if self.state == self.State.GRASPING_WOUNDED or self.state == self.State.SEARCHING_RESCUE_CENTER or self.state == self.State.GOING_RESCUE_CENTER:
            wounded_msg = DroneMessage(
                subject=DroneMessage.Subject.LOCK_WOUNDED,
                arg=(self.identifier, self.estimated_pose.position.tolist())
            )
            message.append(wounded_msg)

        if self.state == self.State.WAITING_DEPARTURE:
            departure_msg = DroneMessage(
                subject=DroneMessage.Subject.DEPARTURE,
                arg=(self.identifier, self.compute_departure_score())
            )
            message.append(departure_msg)

        loc_msg = DroneMessage(
            subject=DroneMessage.Subject.FRONTIER_PRIO,
            arg=(self.identifier, self.estimated_pose.position.tolist())
        )

        message.append(loc_msg)

        return message

    def communication_management(self):
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

    def compute_confidence(self):
        if self.estimated_pose.gps is None:
            return 0.1
        else :
            return 0.5
        
    def clip_command(self, command):
        command["forward"] = np.clip(command["forward"], -1.0, 1.0)
        command["lateral"] = np.clip(command["lateral"], -1.0, 1.0)
        command["rotation"] = np.clip(command["rotation"], -1.0, 1.0)
        return command
    
    def control(self):
        if not self.is_killed() :
            self.history_health.append(self.drone_health)
            self.history_forward_epsilon.append(self.odometer_values()[0] * np.cos(self.odometer_values()[1]))
            self.history_path_searching.append(False)

            self.mapping(display=MappingParams.display_map)

            # Retrieve other drones infos
            self.reset_received_infos()
            self.communication_management()

            # Retrieve Sensor Data
            self.process_lidar_sensor()
            self.process_semantic_sensor()

            self.is_near_rescuing_drone = self.check_near_rescuing_drone(threshold=GraspingParams.hampering_dist)
            if self.is_near_rescuing_drone:
                pass

            # Transitions of the state
            if self.state_lock_counter <= 0:
                self.state_update()

            # Execute Corresponding Command
            state_handlers = {
                self.State.WAITING: self.handle_waiting,
                self.State.DEADLOCK: self.handle_deadlock,
                self.State.WAITING_DEPARTURE: self.handle_waiting_departure,
                self.State.SEARCHING_WALL: self.handle_searching_wall,
                self.State.FOLLOWING_WALL: self.handle_following_wall,
                self.State.GRASPING_WOUNDED: self.handle_grasping_wounded,
                self.State.SEARCHING_RESCUE_CENTER: self.handle_searching_rescue_center,
                self.State.GOING_RESCUE_CENTER: self.handle_going_rescue_center,
                self.State.GRASPING_FOLLOWING_WALL: self.handle_grasping_following_wall,
                self.State.EXPLORING_FRONTIERS: self.handle_exploring_frontiers,
                self.State.STOP: lambda: {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0},
            }

            self.visualise_actions()

            if self.state == self.State.STOP:
                pass

            command = state_handlers.get(self.state, self.handle_unknown_state)()
            command["grasper"] = int(self.need_to_grasp())
            return command
        
        else : 
            # Drone in KillZone. Or at least no lidar available
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
    
    def need_to_grasp(self):
        if self.state in {self.State.GRASPING_WOUNDED,
                          self.State.GRASPING_FOLLOWING_WALL,
                          self.State.SEARCHING_RESCUE_CENTER,
                          self.State.GOING_RESCUE_CENTER}:
            return True
        
        else:
            return False

    def handle_waiting(self):
        self.reset_exploration_path_infos()
        self.step_waiting_count += 1
        return {"forward": 0.0, "lateral": 0.0, "rotation": np.random.random(), "grasper": 0}

    def handle_deadlock(self):
        self.reset_exploration_path_infos()
        self.step_deadlock_count += 1
        return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

    def compute_departure_score(self):
        """
        Drones that have the highest departure scores can start exploring first.
        """
        score = np.sum(self.lidar().get_sensor_values())
        return int(score)

    def barycenter(self, all_drone_positions):
        positions = np.array([pos for _, pos in all_drone_positions])
        return np.mean(positions, axis=0)

    def handle_waiting_departure(self):
        self.all_drones_departure_score.append(self.compute_departure_score())
        if self.elapsed_timestep % self.interval_departure == WaitingDepartureStateParams.departure_time_offset:
            size_drone_group = WaitingDepartureStateParams.size_drone_group
            if self.compute_departure_score() in sorted(self.all_drones_departure_score, reverse=True)[:size_drone_group]:
                self.departure = True
                self.own_departure_timestep = self.elapsed_timestep

        return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

    def handle_searching_wall(self):
        return {"forward": 0.5, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

    def handle_following_wall(self):
        self.epsilon_angle = normalize_angle(self.angle_nearest_obstacle - np.pi/2)     # Parallel to the wall
        self.epsilon_lateral = self.distance_nearest_obstacle - WallFollowingParams.dist_to_stay

        return self.pid_controller( imposed_command_forward = WallFollowingParams.speed_following_wall )

    def handle_grasping_wounded(self):
        self.epsilon_angle = normalize_angle(self.best_angle_wounded)

        return self.pid_controller( imposed_command_forward = GraspingParams.grasping_speed )

    def handle_searching_rescue_center(self):
        if self.previous_state is not self.State.SEARCHING_RESCUE_CENTER:
            self.plan_path_to_rescue_center()

        command = self.follow_path()
        return command

    def plan_path_to_rescue_center(self):
        start_pos = self.estimated_pose.position
        target_pos = self.initial_pos
        max_inflation = PathParams.max_inflation_grasping if bool(self.base.grasper.grasped_entities) \
                                                            else PathParams.max_inflation_obstacle
        if bool(self.estimated_pose.gps):
            self.set_new_path(start_pos, target_pos, max_inflation)
        else:
            self.path = None

    def handle_grasping_following_wall(self):
        command = self.handle_following_wall()
        command["grasper"] = 1
        return command

    def handle_going_rescue_center(self):
        self.epsilon_angle = normalize_angle(self.best_angle_rescue_center)

        return self.pid_controller()

    def handle_exploring_frontiers(self):
        if self.finished_path:
            self.plan_path_to_frontier()

        if self.explored_all_frontiers or self.path is None:
            return self.handle_waiting()

        else:
            return self.follow_path()

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


        if self.is_near_rescue_center :
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
                self.unexplored_point_incentive = None

                for attempt in range(max_attempts):
                    # Sélectionner un point aléatoire avec des marges sécurisées
                    random_x = np.random.randint(padding, grid_shape[0]-padding)
                    random_y = np.random.randint(padding, grid_shape[1]-padding)

                    # Vérifier le voisinage 3x3
                    neighborhood = ternary_map[random_x-1:random_x+2, random_y-1:random_y+2]
                    avg_abs_value = np.mean(neighborhood)
                    # Si la valeur moyenne est sous le seuil, considérer comme non exploré
                    if avg_abs_value < threshold:
                        unexplored_point_incentive = (random_x, random_y)
                        self.unexplored_point_incentive = unexplored_point_incentive
                        break





        for i, drone_id in enumerate(drone_ids):
            drone_pos = drone_positions[drone_id]
            for j, cluster in enumerate(clusters):
                cell_centroid = cluster.cell_closest_to_centroid()
                if self.elapsed_timestep - self.own_departure_timestep < self.interval_departure:
                    cost_matrix[i,j] = 1/math.dist(self.grid._conv_world_to_grid(*self.initial_pos), cell_centroid)
                else :
                    if self.is_near_rescue_center:
                        if self.unexplored_point_incentive is not None:
                            cost_matrix[i, j] = math.dist(cell_centroid, self.unexplored_point_incentive)
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

    def plan_path_to_frontier(self):
        assigned_cluster = self.assign_frontier_cluster()
        if assigned_cluster is not None:
            self.next_frontier = assigned_cluster
            self.next_frontier_target_pos = self.grid.pos_closest_to_centroid(assigned_cluster)

            start_pos = self.estimated_pose.position
            target_pos = self.next_frontier_target_pos
            max_inflation = PathParams.max_inflation_obstacle

            self.set_new_path(start_pos, target_pos, max_inflation)

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

        self.reset_semantic_infos()

        scores = []
        rescue_center_fov_angles = []
        for data in semantic_values:
            if (data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER):
                self.found_rescue_center = True
                rescue_center_fov_angles.append(data.angle)
                if data.distance < 80.0:
                    self.is_near_rescue_center = True
                self.best_angle_rescue_center = circular_mean(np.array(rescue_center_fov_angles))
            
            # If the wounded person detected is held by nobody
            elif (data.entity_type ==
                    DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped):
                self.found_wounded = True
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
        self.best_score = 10000
        for score in filtered_scores:
            if score[0] < self.best_score:
                self.best_score = score[0]
                self.best_angle_wounded = score[1]
                self.distance_nearest_wounded = score[2]
    
    def process_lidar_sensor(self):
        lidar_values = self.lidar().get_sensor_values()

        if lidar_values is None:
            return (False,0)
        
        ray_angles = self.lidar().ray_angles
        size = self.lidar().resolution

        if size != 0:
            self.angle_nearest_obstacle = ray_angles[np.argmin(lidar_values)]

        self.distance_nearest_obstacle = np.min(lidar_values)
        self.found_obstacle = self.distance_nearest_obstacle <= WallFollowingParams.dmax
    
    def pid(self, epsilon, deriv_epsilon, Kp, Kd):
        return Kp*epsilon + Kd*deriv_epsilon

    def pid_controller(self, imposed_command_angle=None, imposed_command_forward=None, imposed_command_lateral=None):
        """
        Uses class attributes (self.epsilon_*) to compute the PID control for the drone.
        """
        command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        # Rotation
        deriv_epsilon = normalize_angle(self.odometer_values()[2])
        self.history_epsilon_angle.append(self.epsilon_angle)
        Kp = PIDParams.Kp_angle
        Kd = PIDParams.Kd_angle
        command["rotation"] = self.pid(self.epsilon_angle, deriv_epsilon, Kp, Kd)

        # Lateral
        deriv_epsilon = -np.sin(self.odometer_values()[1])*self.odometer_values()[0] # vitesse latérale
        self.history_epsilon_lateral.append(self.epsilon_lateral)
        Kp = PIDParams.Kp_lateral
        Kd = PIDParams.Kd_lateral
        command["lateral"] = self.pid(self.epsilon_lateral, deriv_epsilon, Kp, Kd)

        # Forward
        deriv_epsilon = self.epsilon_forward - self.history_epsilon_forward[-1]
        self.history_epsilon_forward.append(self.epsilon_forward)
        Kp = PIDParams.Kp_forward
        Kd = PIDParams.Kd_forward
        command["forward"] = self.pid(self.epsilon_forward, deriv_epsilon, Kp, Kd)

        # Forward and lateral control are efficient only if the angle error is small
        if abs(self.epsilon_angle) >= 0.2:
            command["forward"] = 0.0
            command["lateral"] = 0.0

        if imposed_command_angle != None:
            command["angle"] = imposed_command_angle
        if imposed_command_forward != None:
            command["forward"] = imposed_command_forward
        if imposed_command_lateral != None:
            command["lateral"] = imposed_command_lateral

        self.clip_command(command)

        return command
    
    def is_near_waypoint(self,waypoint):
        distance_to_waypoint = np.linalg.norm(waypoint - self.estimated_pose.position)
        if distance_to_waypoint <= PathParams.distance_close_waypoint:
            return True
        return False

    def is_stabilized(self):
        return np.mean([abs(error) for error in self.history_epsilon_forward]) <= 30.0

    def follow_path(self):
        # Progression in the path
        if self.path is not None:
            if self.is_near_waypoint(self.next_waypoint_position) and self.is_stabilized:
                self.index_current_waypoint += 1
                if self.index_current_waypoint >= len(self.path):
                    self.finished_path = True
                    return self.handle_waiting()
                else:
                    self.previous_waypoint_position = self.next_waypoint_position
                    self.next_waypoint_position = self.path[self.index_current_waypoint]
        
        return self.go_to_next_waypoint()

    def go_to_next_waypoint(self):
        current_segment = self.next_waypoint_position - self.previous_waypoint_position
        error_vector = self.next_waypoint_position - self.estimated_pose.position

        # ANGLE CONTROL
        epsilon_angle = np.arctan2(current_segment[1], current_segment[0]) - self.estimated_pose.orientation
        self.epsilon_angle = normalize_angle(epsilon_angle)

        # LATERAL CONTROL
        if np.linalg.norm(current_segment) != 0.0:
            epsilon_lateral = np.cross(current_segment, error_vector) / np.linalg.norm(current_segment)
            self.epsilon_lateral = epsilon_lateral
        else:
            self.epsilon_lateral = 0.0

        # FORWARD CONTROL
        if np.linalg.norm(current_segment) != 0.0:
            epsilon_forward = np.dot(current_segment, error_vector) / np.linalg.norm(current_segment)
            self.epsilon_forward = epsilon_forward
        else:
            self.epsilon_forward = 0.0

        return self.pid_controller()

    def must_return_area(self):
        if not(bool(self.estimated_pose.gps)):
            return False
        return self.history_health[-1] < HealthParams.THRESHOLD_HEALTH or self.elapsed_timestep / self._misc_data.max_timestep_limit > HealthParams.THRESHOLD_TIMESTEP

    def state_update(self):
        """
        A visualisation of the state machine is available at doc/Drone states
        """
        self.previous_state = self.state
        must_return = self.must_return_area()
        
        conditions = {
            "departure": self.departure,
            "must_return": must_return,
            "is_and_must_inside_return": self.is_inside_return_area and must_return and (not bool(self.base.grasper.grasped_entities)),
            "no_longer_inside_return": not self.is_inside_return_area and must_return and (not bool(self.base.grasper.grasped_entities)),
            "try_searching_rescue_center" : self.elapsed_timestep%200==0 and bool(self.estimated_pose.gps),
            "gps": bool(self.estimated_pose.gps),
            "no_gps" : not(bool(self.estimated_pose.gps)),
            "found_obstacle": self.found_obstacle,
            "lost_wall": not self.found_obstacle,
            "found_wounded": self.found_wounded,
            "must_return_grasping_available": must_return and (not bool(self.base.grasper.grasped_entities)) and self.found_wounded,
            "holding_wounded": bool(self.base.grasper.grasped_entities),
            "lost_wounded": not self.found_wounded and not self.base.grasper.grasped_entities,
            "found_rescue_center": self.found_rescue_center and not (must_return and not bool(self.base.grasper.grasped_entities)),
            "lost_rescue_center": not self.base.grasper.grasped_entities and not must_return,
            "no_frontiers_left": len(self.grid.frontiers) == 0,
            "waiting_time_over": self.step_waiting_count >= WaitingStateParams.step_waiting,
            "deadlock_time_over": self.step_deadlock_count >= WaitingStateParams.step_deadlock,
            "is_near_rescuing_drone": self.is_near_rescuing_drone,
            "health_decreasing" : np.sum(np.diff(np.array(self.history_health)) < 0)>1,
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
                "no_frontiers_left": self.State.SEARCHING_RESCUE_CENTER,
                "is_near_rescuing_drone": self.State.WAITING
            },
            self.State.SEARCHING_WALL: {
                "is_and_must_inside_return": self.State.STOP,
                "must_return": self.State.SEARCHING_RESCUE_CENTER,
                "found_wounded": self.State.GRASPING_WOUNDED,
                "health_decreasing": self.State.WAITING,
                "found_obstacle": self.State.FOLLOWING_WALL,
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

                self.reset_pid_history()
    
    def mapping(self, display = False):
        
        if self.elapsed_timestep == 1: # first iterations
            self.initial_pos = self.measured_gps_position()

        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                   self.measured_compass_angle(),self.odometer_values(),self.previous_position,self.previous_orientation,self.size_area)
        
        self.previous_position = self.estimated_pose.position
        self.previous_orientation = self.estimated_pose.orientation
        
        if self.estimated_pose.gps : 
            self.grid.update(pose=self.estimated_pose)
        
        if display and (self.elapsed_timestep % 5 == 0):
             self.grid.display(self.grid.to_ternary_map(),
                               self.estimated_pose,
                               title=f"Drone {self.identifier} ternary map grid")



    def draw_point(self,point, color=arcade.color.GO_GREEN):
        arcade.draw_circle_filled(point[0], point[1], 5, color)

    def draw_path(self, path):
        if path is None:
            return
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

            if VisualisationParams.draw_frontier_centroid and self.next_frontier_target_pos is not None:
                self.draw_point(self.grid._conv_grid_to_world(*self.next_frontier_target_pos) + self._half_size_array)     # frame of reference change

            if self.unexplored_point_incentive is not None:
                self.draw_point(self.grid._conv_grid_to_world(*self.unexplored_point_incentive) + self._half_size_array, color=arcade.color.YELLOW)

    def visualise_actions(self):
        """
        It's mandatory to use draw_top_layer to draw anything on the interface
        """
        self.draw_top_layer()



    
