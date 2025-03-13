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
class WaitingStateParams:
    step_waiting: int = 20
    step_deadlock: int = 200

@dataclass
class WaitingDepartureStateParams:
    size_drone_group: int = 1
    departure_time_rate: float = 0.6
    departure_time_offset: int = 10

@dataclass
class GraspingParams:
    grasping_speed: float = 0.5
    grasping_dist: int = 30
    hampering_dist: int = 8

@dataclass
class WallFollowingParams:
    dmax: int = 60
    dist_to_stay: int = 30
    speed_following_wall: float = 0.3
    speed_turning: float = 0.05

@dataclass
class PIDParams:
    Kp_angle: float = 9 / math.pi
    Kd_angle: float = Kp_angle / 10
    Ki_angle: float = 0.0
    
    Kp_lateral : float = 0.3111
    Kd_lateral: float = 1.3667
    Ki_lateral: float = 0.0

    Kp_forward : float = 1.6
    Kd_forward: float = 11.0
    Ki_forward: float = 0

@dataclass
class PathParams:
    distance_close_waypoint: int = 25
    max_inflation_obstacle: int = 5
    max_inflation_grasping: int = 7

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
    FRONTIER_ARTIFACT_RESET_VALUE: float = 0

    # Used for the ternary map conversion
    FREE_THRESHOLD: float = -4
    OBSTACLE_THRESHOLD: float = 4

@dataclass
class HealthParams:
    THRESHOLD_HEALTH: int = 5
    THRESHOLD_TIMESTEP: float = 0.9

@dataclass
class CommunicationParams:
    GRID_SHARE_TIME_INTERVAL : int = 5