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

from swarm_rescue.solutions.utils.dataclasses_config import HealthParams


# from swarm_rescue.solutions.utils.astar import can_go_straight


class MyDroneFrontex(DroneAbstract):
    class State(Enum):
        """
        All the states of the drone as a state machine
        """
        WAITING = auto()  # Assigns 1

        SEARCHING_WALL = auto()  # Assigns 2 etc ... This allows to easily add new states
        FOLLOWING_WALL = auto()

        EXPLORING_FRONTIERS = auto()

        GRASPING_WOUNDED = auto()
        SEARCHING_RESCUE_CENTER = auto()
        GOING_RESCUE_CENTER = auto()

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
        self.mapping_params = MappingParams()
        self.estimated_pose = Pose()
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=self.mapping_params.resolution,
                                  lidar=self.lidar(), semantic=self.semantic())

        # POSITION
        self.previous_position = deque(maxlen=1)
        self.previous_position.append((0, 0))
        self.previous_orientation = deque(maxlen=1)
        self.previous_orientation.append(0)

        # STATE INITIALISATION
        self.state = self.State.WAITING
        self.previous_state = self.State.WAITING  # Utile pour vérfier que c'est la première fois que l'on rentre dans un état

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
        self.inital_point_path = (0, 0)
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
        self.last_position = None
        self.counter_static = 0

        self.history_health = deque(maxlen=10)

    def reset_exploration_path_params(self):
        """
        Resets the parameters related to the exploration path.
        """
        self.next_frontier = None
        self.next_frontier_centroid = None
        self.finished_path = True
        self.path = []

    def define_message_for_all(self):
        inKillZone = self.lidar().get_sensor_values() is None
        message = []
        if self.timestep_count <= 1 or inKillZone:
            return None

        if self.timestep_count % CommunicationParams().TIME_INTERVAL == 0:
            confiance = self.compute_confidence(self.estimated_pose.gps)
            message.append(
                DroneMessage(subject=DroneMessage.Subject.MAPPING, arg={"map": self.grid.grid, "confiance": confiance}))
        # message = self.grid.to_update(pose=self.estimated_pose)
        else:
            message.append(DroneMessage(subject=DroneMessage.Subject.PASS, arg=None))

        if self.state == self.State.GRASPING_WOUNDED or self.state == self.State.SEARCHING_RESCUE_CENTER or self.state == self.State.GOING_RESCUE_CENTER:
            broadcast_msg = DroneMessage(
                subject=DroneMessage.Subject.LOCK_WOUNDED,
                arg=(self.identifier, self.estimated_pose.position.tolist())
            )
            message.append(broadcast_msg)
        # if not (self.health_crit() and math.dist(self.grid._conv_world_to_grid(*self.estimated_pose.position),self.grid._conv_world_to_grid(*self.grid.initial_cell)) < 1):
        loc_msg = DroneMessage(
            subject=DroneMessage.Subject.FRONTIER_PRIO,
            arg=(self.identifier, self.estimated_pose.position.tolist(), self.estimated_pose.orientation,
                 self.odometer_values()))
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
                    if drone_msg.subject == DroneMessage.Subject.MAPPING:
                        self.grid.merge_maps(drone_msg.arg["map"], drone_msg.arg["confiance"])
                    if drone_msg.subject == DroneMessage.Subject.LOCK_WOUNDED:
                        drone_id, position = drone_msg.arg
                        self.wounded_locked.append((drone_id, position))
                    if drone_msg.subject == DroneMessage.Subject.FRONTIER_PRIO:
                        drone_id, position, orientation, odometer = drone_msg.arg
                        self.other_drones_pos.append((drone_id, position, orientation, odometer))

    def compute_confidence(self, gps):
        if gps is None:  # Si en zone non gps
            return 0.1
        else:
            return 0.5

    def control(self):
        inKillZone = self.lidar().get_sensor_values() is None or self._drone_health <= 0 or self.State == self.State.STOP

        if not inKillZone:

            self.timestep_count += 1
            print(self.is_inside_return_area)
            self.history_health.append(self._drone_health)
            if abs(self.history_health[-1] - self.history_health[0]) > 2:
                print("health drown")

            # if self.state not in [self.State.SEARCHING_RESCUE_CENTER,self.State.GOING_RESCUE_CENTER]:
            self.mapping(display=self.mapping_params.display_map)

            # print(math.dist(self.grid._conv_world_to_grid(*self.estimated_pose.position),
            # self.grid._conv_world_to_grid(*self.grid.initial_cell)))

            self.communication_management()

            # Retrieve Sensor Data
            found_wall, epsilon_wall_angle, min_dist = self.process_lidar_sensor(self.lidar())
            found_wounded, found_rescue_center, score_wounded, epsilon_wounded, epsilon_rescue_center, is_near_rescue_center, min_dist_wnd = self.process_semantic_sensor()

            is_near_rescuing_drone = self.check_near_rescuing_drone(threshold=GraspingParams.hampering_dist)
            # is_near_rescuing_drone = False
            # if is_near_rescuing_drone:
            #    print("Hampering a rescue, waiting...")

            health_crit = self.health_crit()

            # TRANSITIONS OF THE STATE
            self.state_update(found_wall, found_wounded, found_rescue_center, is_near_rescuing_drone, health_crit)

            # Execute Corresponding Command
            state_handlers = {
                self.State.WAITING: self.handle_waiting,
                self.State.SEARCHING_WALL: self.handle_searching_wall,
                self.State.FOLLOWING_WALL: lambda: self.handle_following_wall(epsilon_wall_angle, min_dist),
                self.State.GRASPING_WOUNDED: lambda: self.handle_grasping_wounded(min_dist_wnd, epsilon_wounded),
                self.State.SEARCHING_RESCUE_CENTER: lambda: self.handle_searching_rescue_center(health_crit),
                self.State.GOING_RESCUE_CENTER: lambda: self.handle_going_rescue_center(epsilon_rescue_center,
                                                                                        is_near_rescue_center,
                                                                                        health_crit),
                self.State.EXPLORING_FRONTIERS: lambda: self.handle_exploring_frontiers(is_near_rescue_center),
                self.State.STOP: lambda: {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
            }

            print(self.identifier, self.state)

            self.visualise_actions()

            self.last_position = self.estimated_pose.position

            return state_handlers.get(self.state, self.handle_unknown_state)()

        else:
            # Drone in KillZone. Or at least no lidar available
            self.last_position = self.estimated_pose.position
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

    def handle_waiting(self):
        self.reset_exploration_path_params()
        self.step_waiting_count += 1
        return {"forward": 0.0, "lateral": 0.0, "rotation": np.random.random(), "grasper": 0}

    def handle_searching_wall(self):
        return {"forward": 0.5, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

    def handle_following_wall(self, epsilon_wall_angle, min_dist):
        epsilon_wall_angle = normalize_angle(epsilon_wall_angle)
        epsilon_wall_distance = min_dist - self.wall_following_params.dist_to_stay

        self.logging_variables(
            {"epsilon_wall_angle": epsilon_wall_angle, "epsilon_wall_distance": epsilon_wall_distance})

        command = {"forward": self.wall_following_params.speed_following_wall, "lateral": 0.0, "rotation": 0.0,
                   "grasper": 0}
        command = self.pid_controller(command, epsilon_wall_angle, self.pid_params.Kp_angle, self.pid_params.Kd_angle,
                                      self.pid_params.Ki_angle, self.past_ten_errors_angle, "rotation")
        command = self.pid_controller(command, epsilon_wall_distance, self.pid_params.Kp_distance,
                                      self.pid_params.Kd_distance, self.pid_params.Ki_distance,
                                      self.past_ten_errors_distance, "lateral")

        return command

    def handle_grasping_wounded(self, score_wounded, epsilon_wounded):
        epsilon_wounded = normalize_angle(epsilon_wounded)
        # print(self.identifier,score_wounded)
        command = {"forward": self.grasping_params.grasping_speed, "lateral": 0.0, "rotation": 0.0,
                   "grasper": 1 if score_wounded < GraspingParams().grasping_dist else 0}
        return self.pid_controller(command, epsilon_wounded, self.pid_params.Kp_angle, self.pid_params.Kd_angle,
                                   self.pid_params.Ki_angle, self.past_ten_errors_angle, "rotation")

    def handle_searching_rescue_center(self, health_crit):
        if self.previous_state is not self.State.SEARCHING_RESCUE_CENTER:
            self.plan_path_to_rescue_center()

        command = self.follow_path(self.path, found_and_near_wounded=True)

        movement = math.dist(self.estimated_pose.position, self.last_position)
        if movement < 0.5:
            self.counter_static += 1
        else:
            self.counter_static = 0
        if self.counter_static > 300:
            print("Grasper drone too static, exiting")
            command["grasper"] = 0
            self.counter_static = 0

        if health_crit:
            command["grasper"] = 0
        return command

    def plan_path_to_rescue_center(self):
        start_cell = self.grid._conv_world_to_grid(*self.estimated_pose.position)
        target_cell = self.grid.initial_cell
        max_inflation = self.path_params.max_inflation_obstacle
        self.path = self.grid.compute_safest_path(start_cell, target_cell, max_inflation)
        self.indice_current_waypoint = 0

    def handle_going_rescue_center(self, epsilon_rescue_center, is_near_rescue_center, health_crit):
        epsilon_rescue_center = normalize_angle(epsilon_rescue_center)
        command = {"forward": 2 * self.grasping_params.grasping_speed, "lateral": 0.0, "rotation": 0.0, "grasper": 1}
        command = self.pid_controller(command, epsilon_rescue_center, self.pid_params.Kp_angle,
                                      self.pid_params.Kd_angle, self.pid_params.Ki_angle, self.past_ten_errors_angle,
                                      "rotation")

        if is_near_rescue_center:
            command["forward"] = -0.1
            command["rotation"] = np.random.rand()  # Rotate in place to drop off

        if health_crit:
            command["grasper"] = 0

        return command

    def handle_exploring_frontiers(self, is_near_rescue_center):
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
        for drone_id, pos, _, _ in self.other_drones_pos:
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
                cell_centroid = cluster.point_closest_to_centroid()
                # Le coût est la distance à parcourir si jamais il prend le path du path qu'il va devoir prendre divisée par (taille du cluster + 1)
                # cost_matrix[i, j] = self.path_distance(self.grid.compute_safest_path(
                #     self.grid._conv_world_to_grid(*drone_pos),
                #     cell_centroid,
                #     self.path_params.max_inflation_obstacle
                # )) /( (cluster.size()) + 1)
                if is_near_rescue_center:
                    cost_matrix[i, j] = math.dist(self.grid._conv_world_to_grid(*drone_pos), cell_centroid)
                else:
                    own_cell = self.grid._conv_world_to_grid(*drone_pos)
                    target_cell = self.grid._conv_world_to_grid(*cell_centroid)
                    if can_go_straight(*own_cell, *target_cell, self.grid.to_ternary_map()):
                        cost_matrix[i, j] = math.dist(self.grid._conv_world_to_grid(*drone_pos), cell_centroid)
                    else:
                        cost_matrix[i, j] = math.dist(self.grid._conv_world_to_grid(*drone_pos),
                                                      cell_centroid) * 10 ** 3

                # cost_matrix[i,j] = self.path_distance(self.grid.compute_safest_path(
                #     self.grid._conv_world_to_grid(*drone_pos),
                #     cell_centroid,
                #     0
                # ))
                # #print(cost_matrix[i, j])
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

    # def plan_path_to_frontier(self):
    #     if self.grid.closest_largest_frontier(self.estimated_pose) is not None:
    #         self.next_frontier, self.next_frontier_centroid = self.grid.closest_largest_frontier(self.estimated_pose)
    #         if self.next_frontier_centroid is not None:
    #             start_cell = self.grid._conv_world_to_grid(*self.estimated_pose.position)
    #             target_cell = self.next_frontier_centroid
    #             max_inflation = self.path_params.max_inflation_obstacle

    #             self.path = self.grid.compute_safest_path(start_cell, target_cell, max_inflation)
    #             print(self.path)
    #             if self.path is None:   # The frontier is unreachable, probably due to artifacts of FREE zones inside boxes set in the mapping process
    #                 print(self.next_frontier.cells)
    #                 #self.grid.delete_frontier_artifacts(self.next_frontier)
    #             else:
    #                 self.indice_current_waypoint = 0

    #     else:
    #         self.explored_all_frontiers = True

    def plan_path_to_frontier(self, is_near_rescue_center):
        assigned_cluster = self.assign_frontier_cluster(is_near_rescue_center)
        # print(f"Assigned cluster: {assigned_cluster}")
        if assigned_cluster is not None:
            self.next_frontier = assigned_cluster
            self.next_frontier_centroid = assigned_cluster.point_closest_to_centroid()
            start_cell = self.grid._conv_world_to_grid(*self.estimated_pose.position)
            target_cell = self.next_frontier_centroid
            max_inflation = self.path_params.max_inflation_obstacle
            self.path = self.grid.compute_safest_path(start_cell, target_cell, max_inflation)
            if self.path is None:
                print("Assigned frontier unreachable, deleting artifacts.")
                # self.path = self.grid.compute_safest_path(start_cell,self.grid.initial_cell,max_inflation)
                # self.indice_current_waypoint = 0
                # self.grid.delete_frontier_artifacts(self.next_frontier)
                # self.state = self.State.SEARCHING_WALL
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

        for _, broadcast_loc in self.wounded_locked:
            # start_cell = self.grid._conv_world_to_grid(*self.estimated_pose.position)
            # target_cell = self.grid._conv_world_to_grid(*broadcast_loc)
            # max_inflation = self.path_params.max_inflation_obstacle
            # path_to_drone = self.grid.compute_safest_path(start_cell, target_cell, max_inflation)
            # if path_to_drone is not None:
            # distance = self.grid.path_distance(path_to_drone)
            # print("distance a*",distance)
            # else:
            # distance = math.dist(self.grid._conv_world_to_grid(*self.estimated_pose.position),self.grid._conv_world_to_grid(*broadcast_loc))
            own_cell = self.grid._conv_world_to_grid(*self.estimated_pose.position)
            other_cell = self.grid._conv_world_to_grid(*broadcast_loc)
            distance = math.dist(own_cell, other_cell)
            # distance = np.linalg.norm(np.array(self.estimated_pose.position) - np.array(broadcast_loc))
            if distance < threshold and can_go_straight(*own_cell, *other_cell, self.grid.to_ternary_map()):
                # print("Near a rescuing drone")
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
                if data.distance < 30.0:
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
        for score in scores:
            conflict = False
            for wnd_locked in self.wounded_locked:
                dx = score[2] * math.cos(score[1] + self.estimated_pose.orientation)
                dy = score[2] * math.sin(score[1] + self.estimated_pose.orientation)
                detection_position = np.array(self.estimated_pose.position) + np.array([dx, dy])
                conflict = False
                if np.linalg.norm(detection_position - np.array(
                        wnd_locked[1])) < GraspingParams.hampering_dist:  # adjust threshold as needed
                    conflict = True
                    print("Conflict of wounded")
                    break
            if not conflict:
                filtered_scores.append(score)
        best_score = 10000
        for score in filtered_scores:
            if score[0] < best_score:
                best_score = score[0]
                best_angle_wounded = score[1]
                mindist = score[2]

        return found_wounded, found_rescue_center, best_score, best_angle_wounded, best_angle_rescue_center, is_near_rescue_center, mindist

    def process_lidar_sensor(self, self_lidar):
        """
        -> ( bool near_obstacle , float epsilon_wall_angle )
        where epsilon_wall_angle is the (counter-clockwise convention) angle made
        between the drone and the nearest wall - pi/2
        """
        lidar_values = self_lidar.get_sensor_values()

        if lidar_values is None:
            return (False, 0)

        ray_angles = self_lidar.ray_angles
        size = self_lidar.resolution

        angle_nearest_obstacle = 0
        if size != 0:
            min_dist = min(lidar_values)
            angle_nearest_obstacle = ray_angles[np.argmin(lidar_values)]

        near_obstacle = False
        if min_dist < self.wall_following_params.dmax:  # pourcentage de la vitesse je pense
            near_obstacle = True

        epsilon_wall_angle = angle_nearest_obstacle - np.pi / 2

        return (near_obstacle, epsilon_wall_angle, min_dist)

    # Takes the current relative error and with a PID controller, returns the command
    # mode : "rotation" or "lateral" for now could be speed or other if implemented
    def pid_controller(self, command, epsilon, Kp, Kd, Ki, past_ten_errors, mode, command_slow=0.8):

        past_ten_errors.pop(0)
        past_ten_errors.append(epsilon)
        if mode == "rotation":
            epsilon = normalize_angle(epsilon)
            deriv_epsilon = normalize_angle(self.odometer_values()[2])
        elif mode == "lateral":
            deriv_epsilon = -np.sin(self.odometer_values()[1]) * self.odometer_values()[0]  # vitesse latérale
        elif mode == "forward":
            deriv_epsilon = self.odometer_values()[0] * np.cos(self.odometer_values()[1])  # vitesse longitudinale
        else:
            raise ValueError("Mode not found")

        correction_proportionnelle = Kp * epsilon
        correction_derivee = Kd * deriv_epsilon
        correction_integrale = 0
        # correction_integrale = Ki * sum(past_ten_errors)
        correction = correction_proportionnelle + correction_derivee + correction_integrale
        command[mode] = correction
        command[mode] = min(max(-1, correction), 1)

        if mode == "rotation":
            if correction > command_slow:
                command["forward"] = self.wall_following_params.speed_turning

        return command

    def is_near_waypoint(self, waypoint):
        distance_to_waypoint = np.linalg.norm(waypoint - self.estimated_pose.position)
        if distance_to_waypoint < self.path_params.distance_close_waypoint:
            # print(f"WAYPOINT {self.indice_current_waypoint} REACH")
            return True
        return False

    def avoid_frontal_collision(self, command):
        """
        Enhanced drone collision avoidance using odometry data (distance traveled,
        relative angle change, rotation change) for both the current and other drones.
        """
        if self.state in [self.State.GRASPING_WOUNDED,
                          self.State.SEARCHING_RESCUE_CENTER,
                          self.State.GOING_RESCUE_CENTER]:
            return command

        # Get own odometry readings
        dist_travel, alpha, theta = self.odometer_values()
        own_pos = np.array(self.estimated_pose.position)
        own_orient = normalize_angle(self.estimated_pose.orientation)

        # Parameters
        collision_distance_threshold = 200.0  # Threshold for collision risk
        prediction_dt = 0.5  # Time horizon for predicting motion (seconds)
        front_angle_threshold = math.radians(60.0)  # Drones ahead (±60°)
        max_heading_diff = math.radians(60.0)  # If the heading is close to 180° (head-on)
        avoidance_strength = 1.0  # Adjust lateral/rotation for avoidance

        for other in self.other_drones_pos:
            if len(other) >= 4:
                # Expecting: (drone_id, position, orientation, odometer_values)
                other_id, other_pos, other_orient, (other_dist_travel, other_alpha, other_theta) = other
            else:
                # Default values if missing
                other_id, other_pos, other_orient = other
                other_dist_travel, other_alpha, other_theta = 0.0, 0.0, 0.0

            other_pos = np.array(other_pos)
            other_orient = normalize_angle(other_orient)

            # Compute future position for both drones using odometry data
            my_future_pos = own_pos + dist_travel * np.array([
                math.cos(own_orient + alpha),
                math.sin(own_orient + alpha)
            ])

            other_future_pos = other_pos + other_dist_travel * np.array([
                math.cos(other_orient + other_alpha),
                math.sin(other_orient + other_alpha)
            ])

            # Compute predicted distance between drones
            predicted_distance = np.linalg.norm(my_future_pos - other_future_pos)
            # print(self.identifier,predicted_distance)

            # Skip avoidance if predicted distance is safe
            if predicted_distance > collision_distance_threshold:
                continue

            # Compute relative angle to the other drone
            vec_to_other = other_pos - own_pos
            angle_to_other = math.atan2(vec_to_other[1], vec_to_other[0])
            rel_angle = normalize_angle(angle_to_other - own_orient)

            # Ensure the other drone is **in front** (not behind or too far to the side)
            if abs(rel_angle) > front_angle_threshold:
                continue

            # Check if the drones are **heading toward each other**
            heading_diff = abs(normalize_angle(other_orient - (own_orient + math.pi)))
            is_head_on = heading_diff < max_heading_diff

            # **Avoidance Strategy**
            # print(f"Avoiding drone {other_id}: Predicted distance {predicted_distance:.2f}, previous_command {command}")

            # Slow down if too close
            command["forward"] = max(0.0, command.get("forward", 1.0) - 0.5)

            # Adjust lateral movement based on the relative angle
            lateral_avoid = avoidance_strength * (1.0 - predicted_distance / (collision_distance_threshold))
            if rel_angle > 0:  # Other drone is to the left → move right
                command["lateral"] = min(1.0, command.get("lateral", 0.0) + lateral_avoid)
            else:  # Other drone is to the right → move left
                command["lateral"] = max(-1.0, command.get("lateral", 0.0) - lateral_avoid)

            # If head-on, increase rotation to turn away
            if is_head_on:
                command["rotation"] = min(1.0, command.get("rotation", 0.0) + 0.5 * avoidance_strength)
            print(f"Result computation {other_id}: {command}")

        return command

    def avoid_frontal_collision(self, command):
        if self.state in [self.State.GRASPING_WOUNDED, self.State.SEARCHING_RESCUE_CENTER,
                          self.State.GOING_RESCUE_CENTER]:
            return command
        for _, other_pos, other_orientation, _ in self.other_drones_pos:
            pos_other = self.grid._conv_world_to_grid(*other_pos)
            pos_own = self.grid._conv_world_to_grid(*self.estimated_pose.position)
            own_orientation = self.estimated_pose.orientation
            angle_between = math.atan2(pos_own[1] - pos_other[1], pos_own[0] - pos_other[0])
            if math.dist(pos_other, pos_own) < 40:
                if abs(normalize_angle(own_orientation + other_orientation)) < math.pi / 3 and abs(
                        angle_between) < math.pi / 3:
                    print("avoiding other drone", math.dist(pos_other, pos_own))
                    command["lateral"] = min(1.0, command["lateral"] + 0.2)
                    command["rotation"] = min(1.0, command["rotation"] + 0.2)
        return command

    def avoid_frontal_collision(self, command):
        # Do not modify the command if in a critical state.
        if self.state in [self.State.GRASPING_WOUNDED,
                          self.State.SEARCHING_RESCUE_CENTER,
                          self.State.GOING_RESCUE_CENTER]:
            return command

        # Convert own position once.
        pos_own = self.grid._conv_world_to_grid(*self.estimated_pose.position)
        own_orientation = self.estimated_pose.orientation

        # Define thresholds (units depend on your grid/resolution)
        collision_threshold = 15.0  # Within this distance: apply full avoidance
        early_avoidance_threshold = 40.0  # Start avoidance adjustments within this range

        for _, other_pos, other_orientation, _ in self.other_drones_pos:
            pos_other = self.grid._conv_world_to_grid(*other_pos)

            # Check line-of-sight: if blocked, skip avoidance for that drone.
            if not can_go_straight(*pos_own, *pos_other, self.grid.to_ternary_map()):
                continue

            distance = math.dist(pos_other, pos_own)

            # Consider drones only within the early avoidance threshold.
            if distance < early_avoidance_threshold:
                # Compute the angle from our drone to the other drone.
                angle_to_other = math.atan2(pos_other[1] - pos_own[1],
                                            pos_other[0] - pos_own[0])
                # Compute the relative angle between our heading and the other drone.
                rel_angle = normalize_angle(angle_to_other - own_orientation)

                # Only act if the other drone is roughly in front (within ±60°).
                if abs(rel_angle) < math.pi / 3:
                    # Scaling factor: 1.0 when distance <= collision_threshold and 0 when at early_avoidance_threshold.
                    if distance < collision_threshold:
                        scaling = 1.0
                    else:
                        scaling = (early_avoidance_threshold - distance) / (
                                early_avoidance_threshold - collision_threshold)
                    scaling = max(0.0, min(1.0, scaling))

                    print("Avoiding other drone at distance", distance, "with scaling", scaling)

                    # Reduce forward speed proportionally.
                    command["forward"] = max(0.0, command.get("forward", 1.0) - 0.5 * scaling)

                    # Adjust lateral and rotational commands.
                    if rel_angle > 0:
                        # Other drone is to the left: move right and rotate right.
                        command["lateral"] = min(1.0, command.get("lateral", 0.0) + 0.5 * scaling)
                        command["rotation"] = min(1.0, command.get("rotation", 0.0) + 0.5 * scaling)
                    else:
                        # Other drone is to the right: move left and rotate left.
                        command["lateral"] = max(-1.0, command.get("lateral", 0.0) - 0.5 * scaling)
                        command["rotation"] = max(-1.0, command.get("rotation", 0.0) - 0.5 * scaling)
        return command

    def avoid_frontal_collision(self, command):
        # Do not modify the command if in a critical state.
        if self.state in [self.State.GRASPING_WOUNDED,
                          self.State.SEARCHING_RESCUE_CENTER,
                          self.State.GOING_RESCUE_CENTER]:
            return command

        # Current own position and orientation.
        pos_own = np.array(self.grid._conv_world_to_grid(*self.estimated_pose.position))
        own_orientation = self.estimated_pose.orientation

        # Retrieve own odometry: forward travel and small angle change.
        # For example, assume self.odometer_values() returns (dist_travel, alpha, theta)
        dist_travel, alpha, _ = self.odometer_values()
        prediction_dt = 0.5  # Prediction time horizon in seconds.

        # Predict your future position.
        my_future_pos = pos_own + dist_travel * np.array([math.cos(own_orientation + alpha),
                                                          math.sin(own_orientation + alpha)])

        # Define thresholds.
        collision_threshold = 15.0  # Threshold for predicted collision.
        early_avoidance_threshold = 30.0  # Start avoidance when within this range.

        for drone_data in self.other_drones_pos:
            # Unpack other drone data. Expecting (id, pos, orientation, odometry?)
            if len(drone_data) >= 4:
                other_id, other_pos, other_orientation, other_odometry = drone_data
                other_dist, other_alpha, _ = other_odometry
            else:
                other_id, other_pos, other_orientation, _ = drone_data
                other_dist, other_alpha = 0.0, 0.0  # Default if no odometry.

            pos_other = np.array(self.grid._conv_world_to_grid(*other_pos))

            # Check if there is a clear line of sight.
            if not can_go_straight(*pos_own, *pos_other, self.grid.to_ternary_map()):
                continue

            # Predict the other drone's future position if possible.
            other_future_pos = pos_other
            if other_dist > 0:
                other_future_pos = pos_other + other_dist * np.array([math.cos(other_orientation + other_alpha),
                                                                      math.sin(other_orientation + other_alpha)])

            # Compute predicted distance between future positions.
            predicted_distance = np.linalg.norm(my_future_pos - other_future_pos)
            # print(predicted_distance)

            # Act if the predicted distance is within the early avoidance threshold.
            if predicted_distance < early_avoidance_threshold:
                # Compute the angle from our current position to the other drone's current position.
                angle_to_other = math.atan2(pos_other[1] - pos_own[1], pos_other[0] - pos_own[0])
                rel_angle = normalize_angle(angle_to_other - own_orientation)

                if abs(rel_angle) < math.pi / 3:  # Only if roughly in front.
                    # Scale adjustments based on how close the predicted distance is.
                    if predicted_distance < collision_threshold:
                        scaling = 1.0
                    else:
                        scaling = (early_avoidance_threshold - predicted_distance) / (
                                early_avoidance_threshold - collision_threshold)
                    scaling = max(0.0, min(1.0, scaling))

                    print(
                        f"Avoiding drone {other_id}: predicted_distance {predicted_distance:.2f}, scaling {scaling:.2f}")

                    # Reduce forward speed.
                    command["forward"] = max(0.0, command.get("forward", 1.0) - 1.0 * scaling)

                    # Adjust lateral and rotational commands.
                    if rel_angle > 0:
                        # Other drone is to the left: steer right.
                        command["lateral"] = min(1.0, command.get("lateral", 0.0) + 1.0 * scaling)
                        command["rotation"] = min(1.0, command.get("rotation", 0.0) + 1.0 * scaling)
                    else:
                        # Other drone is to the right: steer left.
                        command["lateral"] = max(-1.0, command.get("lateral", 0.0) - 1.0 * scaling)
                        command["rotation"] = max(-1.0, command.get("rotation", 0.0) - 1.0 * scaling)
        return command

    def avoid_frontal_collision(self, command):
        # Do not modify the command if in a critical state.
        if self.state in [self.State.GRASPING_WOUNDED,
                          self.State.SEARCHING_RESCUE_CENTER,
                          self.State.GOING_RESCUE_CENTER]:
            return command

        # Current own position and orientation.
        pos_own = np.array(self.grid._conv_world_to_grid(*self.estimated_pose.position))
        own_orientation = self.estimated_pose.orientation

        # Retrieve own odometry: forward travel and small angle change.
        # For example, assume self.odometer_values() returns (dist_travel, alpha, theta)
        dist_travel, alpha, _ = self.odometer_values()
        prediction_dt = 0.5  # Prediction time horizon in seconds.

        # Predict your future position.
        my_future_pos = pos_own + dist_travel * np.array([math.cos(own_orientation + alpha),
                                                          math.sin(own_orientation + alpha)])

        # Define thresholds.
        collision_threshold = 15.0  # Threshold for predicted collision.
        early_avoidance_threshold = 30.0  # Start avoidance when within this range.

        for drone_data in self.other_drones_pos:
            # Unpack other drone data. Expecting (id, pos, orientation, odometry?)
            if len(drone_data) >= 4:
                other_id, other_pos, other_orientation, other_odometry = drone_data
                other_dist, other_alpha, _ = other_odometry
            else:
                other_id, other_pos, other_orientation, _ = drone_data
                other_dist, other_alpha = 0.0, 0.0  # Default if no odometry.

            pos_other = np.array(self.grid._conv_world_to_grid(*other_pos))

            # Check if there is a clear line of sight.
            if not can_go_straight(*pos_own, *pos_other, self.grid.to_ternary_map()):
                continue

            # Predict the other drone's future position if possible.
            other_future_pos = pos_other
            if other_dist > 0:
                other_future_pos = pos_other + other_dist * np.array([math.cos(other_orientation + other_alpha),
                                                                      math.sin(other_orientation + other_alpha)])

            # Compute predicted distance between future positions.
            predicted_distance = np.linalg.norm(my_future_pos - other_future_pos)
            # print(predicted_distance)

            # Act if the predicted distance is within the early avoidance threshold.
            if predicted_distance < early_avoidance_threshold:
                # Compute the angle from our current position to the other drone's current position.
                angle_to_other = math.atan2(pos_other[1] - pos_own[1], pos_other[0] - pos_own[0])
                rel_angle = normalize_angle(angle_to_other - own_orientation)

                if abs(rel_angle) < math.pi / 3:  # Only if roughly in front.
                    # Scale adjustments based on how close the predicted distance is.
                    if predicted_distance < collision_threshold:
                        scaling = 1.0
                    else:
                        scaling = (early_avoidance_threshold - predicted_distance) / (
                                early_avoidance_threshold - collision_threshold)
                    scaling = max(0.0, min(1.0, scaling))

                    print(
                        f"Avoiding drone {other_id}: predicted_distance {predicted_distance:.2f}, scaling {scaling:.2f}")

                    # Reduce forward speed.
                    command["forward"] = max(0.0, command.get("forward", 1.0) - 1.0 * scaling)

                    # Adjust lateral and rotational commands.
                    if rel_angle > 0:
                        # Other drone is to the left: steer right.
                        command["lateral"] = min(1.0, command.get("lateral", 0.0) + 1.0 * scaling)
                        command["rotation"] = min(1.0, command.get("rotation", 0.0) + 1.0 * scaling)
                    else:
                        # Other drone is to the right: steer left.
                        command["lateral"] = max(-1.0, command.get("lateral", 0.0) - 1.0 * scaling)
                        command["rotation"] = max(-1.0, command.get("rotation", 0.0) - 1.0 * scaling)
        return command

    def avoid_frontal_collision(self, command):
        # Do not modify the command if in a critical state.
        if self.state in [self.State.GRASPING_WOUNDED,
                          self.State.SEARCHING_RESCUE_CENTER,
                          self.State.GOING_RESCUE_CENTER]:
            return command

        # Current own position, velocity vector, and orientation
        pos_own = np.array(self.grid._conv_world_to_grid(*self.estimated_pose.position))
        own_orientation = self.estimated_pose.orientation

        # Get current velocity from odometry
        dist_travel, alpha, _ = self.odometer_values()
        # Convert to velocity vector in grid coordinates
        own_vel = dist_travel * np.array([math.cos(own_orientation + alpha),
                                          math.sin(own_orientation + alpha)])
        own_speed = np.linalg.norm(own_vel)

        # Safety radius (combined radius of both drones) - INCREASED
        safety_radius = 5.0
        # Time horizon for velocity obstacles - INCREASED for faster reactions
        time_horizon = 3.0

        # Collect all velocity obstacles
        velocity_obstacles = []
        collision_risk = False
        min_approach_time = float('inf')

        for drone_data in self.other_drones_pos:
            # Unpack other drone data
            if len(drone_data) >= 4:
                other_id, other_pos, other_orientation, other_odometry = drone_data
                other_dist, other_alpha, _ = other_odometry
            else:
                other_id, other_pos, other_orientation = drone_data[:3]
                other_dist, other_alpha = 0.0  # Default if no odometry

            pos_other = np.array(self.grid._conv_world_to_grid(*other_pos))

            # Check if there is a clear line of sight
            if not can_go_straight(*pos_own, *pos_other, self.grid.to_ternary_map()):
                continue

            # Compute relative position
            rel_pos = pos_other - pos_own
            distance = np.linalg.norm(rel_pos)

            # Skip if the drone is too far away to be relevant - INCREASED RANGE
            if distance > 80.0:  # Increased from 50.0
                continue

            # Compute other drone's velocity
            other_vel = other_dist * np.array([math.cos(other_orientation + other_alpha),
                                               math.sin(other_orientation + other_alpha)])
            other_speed = np.linalg.norm(other_vel)

            # Compute relative velocity
            rel_vel = own_vel - other_vel
            rel_speed = np.linalg.norm(rel_vel)

            # If relative speed is very small, they're moving together
            if rel_speed < 0.1:
                continue

            # Check if collision course (using time to closest approach)
            rel_pos_unit = rel_pos / distance if distance > 0 else np.array([0, 0])

            # Calculate dot product to see if approaching
            approaching = np.dot(rel_vel, rel_pos_unit) < 0

            if approaching:
                # Time to closest approach
                time_to_closest = -np.dot(rel_pos, rel_vel) / (rel_speed * rel_speed)
                if time_to_closest > 0:
                    # Distance at closest approach
                    closest_approach_dist = np.linalg.norm(rel_pos + time_to_closest * rel_vel)

                    # Update minimum approach time
                    if time_to_closest < min_approach_time:
                        min_approach_time = time_to_closest

                    # RELAXED CONDITION: Use a larger safety radius for high-speed encounters
                    dynamic_safety_radius = safety_radius * (1.0 + rel_speed / 2.0)

                    if closest_approach_dist < dynamic_safety_radius:
                        collision_risk = True

                        # Calculate the apex of the velocity obstacle cone
                        vo_apex = other_vel

                        # Calculate the center of the constraint circle
                        constraint_center = pos_other - pos_own
                        constraint_center = constraint_center / time_horizon

                        # Radius of the constraint circle - DYNAMIC based on speeds
                        constraint_radius = dynamic_safety_radius / time_horizon

                        # Store the velocity obstacle data
                        velocity_obstacles.append({
                            'id': other_id,
                            'apex': vo_apex,
                            'center': constraint_center,
                            'radius': constraint_radius,
                            'distance': distance,
                            'rel_speed': rel_speed,
                            'time_to_closest': time_to_closest
                        })

        # If no velocity obstacles, return original command
        if not velocity_obstacles:
            return command

        # Sort velocity obstacles by time to closest approach (prioritize imminent collisions)
        velocity_obstacles.sort(key=lambda vo: vo['time_to_closest'])

        # Preferred velocity based on current command
        preferred_vel = np.array([
            command.get("forward", 0.0) * math.cos(own_orientation) - command.get("lateral", 0.0) * math.sin(
                own_orientation),
            command.get("forward", 0.0) * math.sin(own_orientation) + command.get("lateral", 0.0) * math.cos(
                own_orientation)
        ])

        # Sample a set of velocities around preferred velocity
        num_samples = 50  # Increased from 30
        velocity_samples = []
        max_speed = max(1.0, np.linalg.norm(preferred_vel))

        # Include the preferred velocity
        velocity_samples.append(preferred_vel)

        # Generate samples in a circular pattern with bias toward perpendicular directions
        for i in range(num_samples - 1):
            angle = 2 * math.pi * i / (num_samples - 1)
            # Add more samples perpendicular to the current heading
            if min_approach_time < 1.0:  # If collision is imminent
                # Bias sampling toward perpendicular directions
                perp_bias = 0.5 * math.sin(2 * angle)
                biased_angle = angle + perp_bias
                sample_speed = max_speed * (0.5 + 0.5 * abs(math.sin(angle)))  # Vary speed with angle
            else:
                biased_angle = angle
                sample_speed = max_speed

            sample_vel = sample_speed * np.array([math.cos(biased_angle), math.sin(biased_angle)])
            velocity_samples.append(sample_vel)

        # Evaluate each velocity sample
        best_vel = None
        best_score = float('-inf')

        for vel in velocity_samples:
            # Check if velocity is in any velocity obstacle
            in_obstacle = False

            for vo in velocity_obstacles:
                # Vector from apex to velocity
                vel_rel_to_apex = vel - vo['apex']

                # Distance from velocity to constraint center
                dist_to_center = np.linalg.norm(vel_rel_to_apex - vo['center'])

                # If distance is less than radius, velocity is in obstacle
                if dist_to_center < vo['radius']:
                    in_obstacle = True
                    break

            # If velocity is outside all obstacles, evaluate it
            if not in_obstacle:
                # Base score on similarity to preferred velocity
                base_score = -np.linalg.norm(vel - preferred_vel)

                # Additional scoring factors
                # 1. Prefer higher speeds for emergency avoidance
                speed_factor = np.linalg.norm(vel) / max_speed if collision_risk else 1.0

                # 2. Prefer directions that increase separation most rapidly
                # For the closest obstacle
                closest_vo = velocity_obstacles[0]
                separation_dir = pos_own - np.array(self.grid._conv_world_to_grid(*self.other_drones_pos[0][1]))
                separation_dir = separation_dir / np.linalg.norm(separation_dir)
                separation_factor = np.dot(vel / np.linalg.norm(vel), separation_dir) if np.linalg.norm(vel) > 0 else 0

                # Compute final score with emergency weighting if needed
                if min_approach_time < 1.0:  # Imminent collision
                    score = 0.2 * base_score + 0.4 * speed_factor + 0.4 * separation_factor
                else:
                    score = 0.6 * base_score + 0.2 * speed_factor + 0.2 * separation_factor

                if score > best_score:
                    best_score = score
                    best_vel = vel

        # If no valid velocity found, use emergency strategy
        if best_vel is None:
            # Find the velocity obstacle with the closest drone
            closest_vo = velocity_obstacles[0]

            # Calculate escape vector (perpendicular to the line of centers)
            escape_dir = np.array([-closest_vo['center'][1], closest_vo['center'][0]])
            escape_dir = escape_dir / np.linalg.norm(escape_dir)

            # Set best velocity as escape direction with FULL speed for emergencies
            best_vel = max_speed * escape_dir  # Changed from 0.5 * max_speed

        # Convert best velocity back to command format
        # Decompose velocity into forward and lateral components in body frame
        best_forward = best_vel[0] * math.cos(own_orientation) + best_vel[1] * math.sin(own_orientation)
        best_lateral = -best_vel[0] * math.sin(own_orientation) + best_vel[1] * math.cos(own_orientation)

        # Calculate rotation to align with best velocity
        best_heading = math.atan2(best_vel[1], best_vel[0])
        rotation = normalize_angle(best_heading - own_orientation)

        # More aggressive rotation for imminent collisions
        if min_approach_time < 1.0:
            rotation = np.clip(rotation / (math.pi / 4), -1.0, 1.0)  # More aggressive scaling
        else:
            rotation = np.clip(rotation / (math.pi / 2), -1.0, 1.0)  # Regular scaling

        # Update command with more aggressive control for imminent collisions
        if collision_risk and min_approach_time < 1.0:
            # More aggressive scaling for emergency situations
            command["forward"] = np.clip(best_forward * 1.5, -1.0, 1.0)
            command["lateral"] = np.clip(best_lateral * 1.5, -1.0, 1.0)
            command["rotation"] = np.clip(rotation * 1.2, -1.0, 1.0)
        else:
            command["forward"] = np.clip(best_forward, -1.0, 1.0)
            command["lateral"] = np.clip(best_lateral, -1.0, 1.0)
            command["rotation"] = np.clip(rotation, -1.0, 1.0)

        return command

    def avoid_frontal_collision(self, command):
        # Do not modify the command if in a critical state.
        if self.state in [self.State.GRASPING_WOUNDED,
                          self.State.SEARCHING_RESCUE_CENTER,
                          self.State.GOING_RESCUE_CENTER]:
            return command

        # Convert own position once.
        pos_own = self.grid._conv_world_to_grid(*self.estimated_pose.position)
        own_orientation = self.estimated_pose.orientation

        # Define thresholds (units depend on your grid/resolution)
        collision_threshold = 10.0  # Within this distance: apply full avoidance
        early_avoidance_threshold = 30.0  # Start avoidance adjustments within this range

        for _, other_pos, other_orientation, _ in self.other_drones_pos:
            pos_other = self.grid._conv_world_to_grid(*other_pos)

            # Check line-of-sight: if blocked, skip avoidance for that drone.
            if not can_go_straight(*pos_own, *pos_other, self.grid.to_ternary_map()):
                continue

            distance = math.dist(pos_other, pos_own)

            # Consider drones only within the early avoidance threshold.
            if distance < early_avoidance_threshold:
                # Compute the angle from our drone to the other drone.
                angle_to_other = math.atan2(pos_other[1] - pos_own[1],
                                            pos_other[0] - pos_own[0])
                # Compute the relative angle between our heading and the other drone.
                rel_angle = normalize_angle(angle_to_other - own_orientation)

                # Only act if the other drone is roughly in front (within ±60°).
                if abs(rel_angle) < math.pi / 6:
                    # Scaling factor: 1.0 when distance <= collision_threshold and 0 when at early_avoidance_threshold.
                    if distance < collision_threshold:
                        scaling = 1.0
                    else:
                        scaling = (early_avoidance_threshold - distance) / (
                                early_avoidance_threshold - collision_threshold)
                    scaling = max(0.0, min(1.0, scaling))

                    print("Avoiding other drone at distance", distance, "with scaling", scaling)

                    # Reduce forward speed proportionally.
                    command["forward"] = max(0.0, command.get("forward", 1.0) - 0.3 * scaling)

                    # Adjust lateral and rotational commands.
                    if rel_angle > 0:
                        # Other drone is to the left: move right and rotate right.
                        command["lateral"] = min(1.0, command.get("lateral", 0.0) + 0.3 * scaling)
                        command["rotation"] = min(1.0, command.get("rotation", 0.0) + 0.3 * scaling)
                    else:
                        # Other drone is to the right: move left and rotate left.
                        command["lateral"] = max(-1.0, command.get("lateral", 0.0) - 0.3 * scaling)
                        command["rotation"] = max(-1.0, command.get("rotation", 0.0) - 0.3 * scaling)
        return command

    def follow_path(self, path, found_and_near_wounded):
        if path is None:
            self.finished_path = True  # NOT USE YET
            self.indice_current_waypoint = 0
            self.path = []
            self.path_grid = []
            return
        else:
            if self.is_near_waypoint(path[self.indice_current_waypoint]):
                self.indice_current_waypoint += 1  # next point in path
                # print(f"Waypoint reached {self.indice_current_waypoint}")
                if self.indice_current_waypoint >= len(path):
                    self.finished_path = True  # NOT USE YET
                    self.indice_current_waypoint = 0
                    self.path = []
                    self.path_grid = []
                    return

            command = self.go_to_waypoint(path[self.indice_current_waypoint][0], path[self.indice_current_waypoint][1],
                                          found_and_near_wounded)
            return self.avoid_frontal_collision(command)

    def go_to_waypoint(self, x, y, found_and_near_wounded):
        # Compute angle error (for rotation control)
        dx = x - self.estimated_pose.position[0]
        dy = y - self.estimated_pose.position[1]
        epsilon = math.atan2(dy, dx) - self.estimated_pose.orientation
        epsilon = normalize_angle(epsilon)

        # Get the initial command from the PID controller for rotation
        command_path = self.pid_controller(
            {"forward": 1, "lateral": 0.0, "rotation": 0.0, "grasper": 1 if found_and_near_wounded else 0},
            epsilon,
            self.pid_params.Kp_angle,
            self.pid_params.Kd_angle,
            self.pid_params.Ki_angle,
            self.past_ten_errors_angle,
            "rotation",
            0.5
        )

        # Lateral adjustment (PID control) based on distance to the ideal path
        if self.indice_current_waypoint == 0:
            x_previous_waypoint, y_previous_waypoint = self.inital_point_path
        else:
            x_previous_waypoint, y_previous_waypoint = self.path[self.indice_current_waypoint - 1][0], \
                self.path[self.indice_current_waypoint - 1][1]

        epsilon_distance = compute_relative_distance_to_droite(x_previous_waypoint, y_previous_waypoint, x, y,
                                                               self.estimated_pose.position[0],
                                                               self.estimated_pose.position[1])
        command_path = self.pid_controller(
            command_path,
            epsilon_distance,
            self.pid_params.Kp_distance_1,
            self.pid_params.Kd_distance_1,
            self.pid_params.Ki_distance_1,
            self.past_ten_errors_distance,
            "lateral",
            0.5
        )

        # Compute the distance to the waypoint
        distance_to_waypoint = math.dist(self.grid._conv_world_to_grid(x, y),
                                         self.grid._conv_world_to_grid(*self.estimated_pose.position))

        # Apply a scaling factor for forward speed:
        # For instance, if the waypoint is closer than 10 units, scale the speed down linearly.
        print(distance_to_waypoint)
        slow_distance_threshold = 20.0  # Distance below which the drone should start slowing down
        min_speed_factor = 0.3  # Minimum fraction of the full speed you want when very close

        # Scale factor is 1.0 if far away and decreases as it gets closer.
        speed_scale = min(1.0, np.clip(distance_to_waypoint / slow_distance_threshold, min_speed_factor, 0.8))

        # Update forward command accordingly
        command_path["forward"] *= speed_scale

        return command_path

    def health_crit(self):
        if self.history_health[-1] < HealthParams.THRESHOLD_HEALTH:
            return True
        return False

    def state_update(self, found_wall, found_wounded, found_rescue_center, is_near_rescuing_drone, health_crit):
        """
        A visualisation of  the state machine is available at doc/Drone states
        """
        self.previous_state = self.state

        conditions = {
            "health_crit": health_crit or self.elapsed_timestep / self._misc_data.max_timestep_limit > 0.2,
            "health_drown": abs(self.history_health[-1] - self.history_health[0]) > 2,
            "returned_rescue_final": self.is_inside_return_area and (
                        health_crit or self.elapsed_timestep / self._misc_data.max_timestep_limit > 0.2),
            "lost_final_rescue": not self.is_inside_return_area,
            "found_wall": found_wall,
            "lost_wall": not found_wall,
            "found_wounded": found_wounded,
            "holding_wounded": bool(self.base.grasper.grasped_entities),
            "lost_wounded": not found_wounded and not self.base.grasper.grasped_entities,
            "found_rescue_center": found_rescue_center,
            "lost_rescue_center": (not self.base.grasper.grasped_entities) and (
                        not health_crit and not self.elapsed_timestep / self._misc_data.max_timestep_limit > 0.2),
            "no_frontiers_left": len(self.grid.frontiers) == 0,
            "waiting_time_over": self.step_waiting_count >= self.waiting_params.step_waiting and not (
                        health_crit or self.elapsed_timestep / self._misc_data.max_timestep_limit > 0.7),
            "is_near_rescuing_drone": is_near_rescuing_drone
        }

        STATE_TRANSITIONS = {
            self.State.WAITING: {
                "returned_rescue_final": self.State.STOP,
                # "health_drown": self.State.WAITING,
                "found_wounded": self.State.GRASPING_WOUNDED,
                "waiting_time_over": self.State.EXPLORING_FRONTIERS
            },
            self.State.STOP: {
                "returned_rescue_final": self.State.STOP,
                "lost_final_rescue": self.State.SEARCHING_RESCUE_CENTER
            },
            self.State.GRASPING_WOUNDED: {
                "lost_wounded": self.State.WAITING,
                "holding_wounded": self.State.SEARCHING_RESCUE_CENTER
            },
            self.State.SEARCHING_RESCUE_CENTER: {
                "health_drown": self.State.WAITING,
                "lost_rescue_center": self.State.WAITING,
                "found_rescue_center": self.State.GOING_RESCUE_CENTER
            },
            self.State.GOING_RESCUE_CENTER: {
                "health_drown": self.State.WAITING,
                "returned_rescue_final": self.State.WAITING,
                "lost_rescue_center": self.State.WAITING
            },
            self.State.EXPLORING_FRONTIERS: {
                "health_drown": self.State.WAITING,
                "health_crit": self.State.SEARCHING_RESCUE_CENTER,
                "found_wounded": self.State.GRASPING_WOUNDED,
                "no_frontiers_left": self.State.FOLLOWING_WALL,
                "is_near_rescuing_drone": self.State.WAITING
            },
            self.State.SEARCHING_WALL: {
                "health_drown": self.State.WAITING,
                "health_crit": self.State.SEARCHING_RESCUE_CENTER,
                "found_wounded": self.State.GRASPING_WOUNDED,
                "found_wall": self.State.FOLLOWING_WALL,
                "is_near_rescuing_drone": self.State.WAITING
            },
            self.State.FOLLOWING_WALL: {
                "health_drown": self.State.WAITING,
                "health_crit": self.State.SEARCHING_RESCUE_CENTER,
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

    def mapping(self, display=False):

        if self.timestep_count == 1:  # first iterations
            print("Starting control")
            start_x, start_y = self.measured_gps_position()  # never none ?
            print(f"Initial position: {start_x}, {start_y}")
            self.grid.set_initial_cell(start_x, start_y)
            self.last_position = (start_x, start_y)

        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                   self.measured_compass_angle(), self.odometer_values(), self.previous_position[-1],
                                   self.previous_orientation[-1], self.size_area)

        self.previous_position.append(self.estimated_pose.position)
        self.previous_orientation.append(self.estimated_pose.orientation)

        self.grid.update(pose=self.estimated_pose)

        if display and (self.timestep_count % 5 == 0):
            self.grid.display(self.grid.to_ternary_map(),
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

    def draw_point(self, point, color=arcade.color.GO_GREEN):
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
                                 float(pt1[1]), [125, 125, 125])
            pt2 = pt1

    def draw_top_layer(self):
        if self.visualisation_params.draw_path:
            self.draw_path(self.path)

        if self.state == self.State.EXPLORING_FRONTIERS:

            if self.visualisation_params.draw_frontier_points and self.next_frontier is not None:
                # print("VISUALISING")
                colors = [arcade.color.RED, arcade.color.BLUE, arcade.color.GREEN, arcade.color.YELLOW,
                          arcade.color.ORANGE, arcade.color.PURPLE]
                for i, f in enumerate(self.grid.frontiers):
                    for cell in f.cells:
                        point = self.grid._conv_grid_to_world(*cell) + self._half_size_array
                        self.draw_point(point, color=colors[i % len(colors)])

            if self.visualisation_params.draw_frontier_centroid and self.next_frontier_centroid is not None:
                self.draw_point(self.grid._conv_grid_to_world(
                    *self.next_frontier_centroid) + self._half_size_array)  # frame of reference change

    def visualise_actions(self):
        """
        It's mandatory to use draw_top_layer to draw anything on the interface
        """
        self.draw_top_layer()

