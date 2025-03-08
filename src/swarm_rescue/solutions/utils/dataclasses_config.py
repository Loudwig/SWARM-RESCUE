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
    drone_world_radius: float = 15.0
    inflated_drone_world_radius: float = 20.0
    drone_grid_radius: int = 3

@dataclass
class TrackingParams:
    wounded_id_add_distance_threshold: float = 20.0
    wounded_id_remove_distance_threshold: float = 50.0

@dataclass
class WaitingStateParams:
    step_waiting: int = 20

@dataclass
class WoundedRescueParams:
    grasping_speed: float = 0.3
    near_wounded_distance_threshold: float = 50.0

@dataclass
class WallFollowingParams:
    dmax: int = 60
    dist_to_stay: int = 40
    speed_following_wall: float = 0.3
    speed_turning: float = 0.05

@dataclass
class GoingRescueCenterParams:
    near_center_distance_threshold: float = 30.0

@dataclass
class ManagingCollisionParams:
    critical_collision_world_distance: float = 1.5 * MappingParams.drone_world_radius
    step_duration_managing_collision: int = 20

@dataclass
class PIDParams:
    Kp_angle: float = 9 / math.pi
    Kd_angle: float = Kp_angle / 10
    Ki_angle: float = (1 / 10) * (1 / 20) * 2 / math.pi
    
    Kp_distance : float = 2 / abs(10)
    Ki_distance: float = 1 / abs(10) * 1 / 20 * 1 / 10
    Kd_distance: float = 2 * Kp_distance

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
    draw_frontier_points: bool = False
    draw_unrescued_wounded: bool = True

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
    FRONTIER_ARTIFACT_RESET_VALUE: float = 1.0

    # Used for the ternary map conversion
    FREE_THRESHOLD: float = 0
    OBSTACLE_THRESHOLD: float = 0

@dataclass
class CommunicationParams:
    map_communication_start_timestep: int = 3500
    map_communication_minimum_interval: int = 300    # in timesteps

@dataclass
class StrategyParams:
    id_always_at_return_area: int = None
    id_always_exploring: int = 1