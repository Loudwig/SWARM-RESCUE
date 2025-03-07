from dataclasses import dataclass, field
from collections import deque
from typing import List, Tuple, Optional
from enum import Enum, auto
import math

@dataclass
class MappingParams:
    resolution: int = 8     # 8 to 1 factor from simulation pixels to grid (efficiency)
    display_map: bool = True
    display_binary_map = True

@dataclass
class TrackingParams:
    wounded_id_distance_threshold: float = 30.0

@dataclass
class WaitingStateParams:
    step_waiting: int = 20

@dataclass
class GraspingParams:
    grasping_speed: float = 0.3

@dataclass
class WallFollowingParams:
    dmax: int = 60
    dist_to_stay: int = 40
    speed_following_wall: float = 0.3
    speed_turning: float = 0.05

@dataclass
class PIDParams:
    Kp_angle: float = 4 / math.pi
    Kp_angle_1: float = 9 / math.pi
    Kd_angle: float = Kp_angle / 10
    Kd_angle_1: float = Kp_angle / 10
    Ki_angle: float = (1 / 10) * (1 / 20) * 2 / math.pi
    
    Kp_distance: float = 2 / abs(WallFollowingParams.dmax - WallFollowingParams.dist_to_stay)
    Kp_distance_1 : float = 2 / abs(10)
    Ki_distance: float = 1 / abs(WallFollowingParams.dist_to_stay) * 1 / 200
    Ki_distance_1: float = 1 / abs(10) * 1 / 20 * 1 / 10
    Kd_distance: float = 2 * Kp_distance
    Kd_distance_1: float = 2 * Kp_distance_1

    Kp_distance_2: float = 10
    Kd_distance_2: float = 0.01

@dataclass
class PathParams:
    distance_close_waypoint: int = 20
    max_inflation_obstacle: int = 5

@dataclass
class LogParams:
    record_log: bool = False
    log_file: str = "logs/log.txt"
    log_initialized: bool = False
    flush_interval: int = 50

@dataclass
class VisualisationParams:
    draw_path: bool = True
    draw_frontier_centroid: bool = True
    draw_frontier_points: bool = True

@dataclass  # Relative to grids.py
class GridParams:
    OBSTACLE: int = 1
    FREE: int = 0
    UNDISCOVERED: int = -2

    MIN_FRONTIER_SIZE: int = 6

    EVERY_N: int = 3
    LIDAR_DIST_CLIP: float = 40.0
    MAX_RANGE_LIDAR_SENSOR_FACTOR: float = 0.9
    EMPTY_ZONE_VALUE: float = -0.602
    OBSTACLE_ZONE_VALUE: float = 2.0
    FREE_ZONE_VALUE: float = -4.0

    THRESHOLD_MIN: float = -40.0
    THRESHOLD_MAX: float = 40.0
    WORLD_BORDERS_VALUE: float = THRESHOLD_MAX
    FRONTIER_ARTIFACT_RESET_VALUE: float = THRESHOLD_MAX

    # Used for the ternary map conversion
    FREE_THRESHOLD: float = 0
    OBSTACLE_THRESHOLD: float = 0

@dataclass
class BehaviourParams:
    try_not_couting_drone_as_obstacle: bool = True

@dataclass
class CommunicationParams:
    TIME_INTERVAL : int = 5