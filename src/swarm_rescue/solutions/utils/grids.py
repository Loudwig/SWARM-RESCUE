import numpy as np
from scipy.ndimage import label
import cv2
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR
from solutions.utils.pose import Pose
from spg_overlay.utils.grid import Grid
from solutions.utils.messages import DroneMessage
from solutions.utils.astar import *
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from solutions.utils.dataclasses_config import *




class OccupancyGrid(Grid):
    """Self updating occupancy grid"""

    OBSTACLE = GridParams.OBSTACLE
    FREE = GridParams.FREE
    UNDISCOVERED = GridParams.UNDISCOVERED

    class Frontier:

        MIN_FRONTIER_SIZE = GridParams.MIN_FRONTIER_SIZE

        def __init__(self, cells):
            """
            Initialize a frontier with a list of grid cells.
            :param cells: List of tuples [(x1, y1), (x2, y2), ...]
            """
            self.cells = cells

        def compute_centroid(self):
            """
            Compute the centroid of the frontier.
            """
            if self.cells.size == 0:
                return None
            x_coords, y_coords = zip(*self.cells)
            return np.array([sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)], dtype=int)

        def size(self):
            """
            Return the number of cells in the frontier.
            """
            return len(self.cells)

    def __init__(self,
                 size_area_world,
                 resolution: float,
                 lidar,semantic):
        super().__init__(size_area_world=size_area_world,
                         resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution

        self.lidar = lidar

        self.semantic = semantic

        self.x_max_grid: int = int(self.size_area_world[0] / self.resolution
                                   + 0.5)
        self.y_max_grid: int = int(self.size_area_world[1] / self.resolution
                                   + 0.5)

        self.initial_cell = None
        self.initial_cell_value = None

        WORLD_BORDERS_VALUE = GridParams.WORLD_BORDERS_VALUE
        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))
        # Set the value of all border cells to WORLD_BORDERS_VALUE so they are considered as obstacles
        self.grid[[0, -1], :] = WORLD_BORDERS_VALUE
        self.grid[:, [0, -1]] = WORLD_BORDERS_VALUE

        self.zoomed_grid = np.empty((self.x_max_grid, self.y_max_grid))

        self.frontier_connectivity_structure = np.ones((3, 3), dtype=int)  # Connects points that are adjacent (even diagonally)
        self.frontiers = []

    def set_initial_cell(self, world_x, world_y):
        """
        Store the cell that corresponds to the initial drone position 
        This should be called once the drone initial position is known.
        """
        cell_x, cell_y = self._conv_world_to_grid(world_x, world_y)
        
        if 0 <= cell_x < self.x_max_grid and 0 <= cell_y < self.y_max_grid:
            self.initial_cell = (cell_x, cell_y)
    
    def to_ternary_map(self):
        """
        Convert the probabilistic occupancy grid into a ternary grid.
        OBSTACLE = 1
        FREE = 0
        UNDISCOVERED = -2
        Cells with value > OBSTACLE_THRESHOLD are considered obstacles.
        Cells with value < FREE_THRESHOLD are considered free.
        Cells with value = 0 are considered undiscovered
        """
        OBSTACLE_THRESHOLD = GridParams.OBSTACLE_THRESHOLD
        FREE_THRESHOLD = GridParams.FREE_THRESHOLD

        ternary_map = np.zeros_like(self.grid, dtype=int)
        ternary_map[self.grid > OBSTACLE_THRESHOLD] = self.OBSTACLE
        ternary_map[self.grid < FREE_THRESHOLD] = self.FREE
        ternary_map[self.grid == 0] = self.UNDISCOVERED
        return ternary_map

    def to_median_map(self):
        median_map = np.zeros_like(self.grid,dtype=int)
        copy = np.float32(self.grid)
        filtered = cv2.medianBlur(copy,3,)
        seuil = 2
        median_map[filtered > seuil] = self.OBSTACLE
        median_map[abs(filtered) <= 0] = self.UNDISCOVERED 
        return median_map
    
    def update_with_sensor_data(self, pose:Pose, other_drones_poses=None):
        """
        Uses a ray casting algorithm to update the values of cells in the grid. 
        """
        to_update = []

        EVERY_N = GridParams.EVERY_N
        LIDAR_DIST_CLIP = GridParams.LIDAR_DIST_CLIP
        MAX_RANGE_LIDAR_SENSOR_FACTOR = GridParams.MAX_RANGE_LIDAR_SENSOR_FACTOR
        EMPTY_ZONE_VALUE = GridParams.EMPTY_ZONE_VALUE
        OBSTACLE_ZONE_VALUE = GridParams.OBSTACLE_ZONE_VALUE
        FREE_ZONE_VALUE = GridParams.FREE_ZONE_VALUE

        lidar_dist = self.lidar.get_sensor_values()[::EVERY_N].copy()   # Distance of each ray to the first obstacle it encounters (or max range if it doesn't)
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()  # Angle of each ray
        
        # Used to go from rays to points on the grid
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        # Any ray that has an associated lidar_dist greater than this threshold is considered to have no obstacle
        no_obstacle_ray_distance_threshold = MAX_RANGE_LIDAR_SENSOR * MAX_RANGE_LIDAR_SENSOR_FACTOR
        # Ensure coherent values to balance noise on lidar values
        processed_lidar_dist = np.clip(lidar_dist - LIDAR_DIST_CLIP, 0, no_obstacle_ray_distance_threshold)

        points_x = pose.position[0] + np.multiply(processed_lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(processed_lidar_dist, sin_rays)

        for pt_x, pt_y in zip(points_x, points_y):
            self.add_value_along_line(pose.position[0], pose.position[1], pt_x, pt_y, EMPTY_ZONE_VALUE)

        # Rays that collide obstacles are those that verify lidar_dist[ray] < max_confidence_range
        select_collision = lidar_dist < no_obstacle_ray_distance_threshold 
        
        points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)
        
        ###############################
        if BehaviourParams().try_not_couting_drone_as_obstacle:
                zone_drone_x , zone_drone_y = self.compute_near_drones_zone(pose)
                epsilon = 3
                for ind,v in enumerate(select_collision):
                    if select_collision[ind] == True:
                        if self.list_any_comparaison_int(abs(zone_drone_x - points_x[ind]),epsilon) and self.list_any_comparaison_int(abs(zone_drone_y - points_y[ind]),epsilon): 
                            select_collision[ind] =  False
        ###############################

        points_x = points_x[select_collision]
        points_y = points_y[select_collision]

        self.add_points(points_x, points_y, OBSTACLE_ZONE_VALUE)

        # the current position of the drone is free !
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)
    
    def update_with_communication(self, list_communicated_grids):
        for communicated_grid in list_communicated_grids:
            # Add only the communicated grid information that is not already known
            unexplored_mask = (self.grid == 0)
            self.grid += communicated_grid * unexplored_mask

    def full_update(self, pose:Pose, communicated_grid):
        """
        Uses both type of updates and threshold the values in the grid so they don't diverge.
        """
        self.update_with_sensor_data(pose)
        self.update_with_communication(communicated_grid)

        # Threshold values in the grid
        self.grid = np.clip(self.grid, GridParams.THRESHOLD_MIN, GridParams.THRESHOLD_MAX)

        self.zoomed_grid = self.grid.copy()
        
        new_zoomed_size = (int(self.size_area_world[1] * 0.5),
                           int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_zoomed_size,
                                      interpolation=cv2.INTER_NEAREST)
    
    def frontiers_update(self):
        ternary_map = self.to_ternary_map()

        # Différences sur les axes X et Y
        diff_x = np.diff(ternary_map, axis=1)
        diff_y = np.diff(ternary_map, axis=0)

        # Détection des frontières entre FREE et UNDISCOVERED
        boundaries_x = np.abs(diff_x) == 2
        boundaries_y = np.abs(diff_y) == 2

        # Combinaison des résultats
        boundaries_map = np.pad(boundaries_x, ((0, 0), (0, 1))) | np.pad(boundaries_y, ((0, 1), (0, 0)))

        labeled_array, num_features = label(boundaries_map, self.frontier_connectivity_structure)

        # Extraction des points de chaque frontière
        frontiers = [np.argwhere(labeled_array == i) for i in range(1, num_features + 1)]
        self.frontiers = [self.Frontier(cells) for cells in frontiers if len(cells) >= self.Frontier.MIN_FRONTIER_SIZE]

    def delete_frontier_artifacts(self, frontier):
        """
        Set to THRESHOLD_MAX (which relates to OBSTACLE) in the grid all cells of frontier
        """
        print("Deleting frontier artifacts")
        if frontier is not None:
            for cell in frontier.cells:
                self.grid[*cell] = GridParams.FRONTIER_ARTIFACT_RESET_VALUE
    
    def closest_largest_frontier(self, pose: Pose):
        """
        Returns the centroid of the frontier with best interest considering both distance to pose and size.
        IN GRID COORDINATES.
        """
        self.frontiers_update()
        if not self.frontiers:
            return None

        pos_drone_grid = np.array(self._conv_world_to_grid(*pose.position))
        frontiers_with_size = [
            (frontier, frontier.compute_centroid(), frontier.size()) for frontier in self.frontiers
        ]
        
        def interest_measure(frontier_data):
            _, centroid, size = frontier_data
            distance = np.linalg.norm(centroid - pos_drone_grid)
            return distance / (size + 1)**2
        
        closest_frontier, closest_centroid, _ = min(frontiers_with_size, key=interest_measure, default=(None, None))
        return (closest_frontier, closest_centroid)

    def compute_safest_path(self, start_cell, target_cell, max_inflation):
        """
        Returns the path, if it exists, that joins drone's position to target_cell
        while approaching the least possible any wall
        start_cell, target_cell : GRID COORDINATES
        """
        MAP = self.to_ternary_map()

        for inflation in range(max_inflation, 0, -1):   # Decreasing inflation to find the safest path
            MAP_inflated = inflate_obstacles(MAP, inflation)
            start_x, start_y = next_point_free(MAP_inflated, *start_cell, max_inflation - inflation)
            end_x, end_y = next_point_free(MAP_inflated, *target_cell, max_inflation - inflation)

            path = a_star_search(MAP_inflated, (start_x, start_y), (end_x, end_y))

            if path:
                path_simplified = self.simplify_path(path, MAP_inflated) or [start_cell]
                return [self._conv_grid_to_world(x, y) for x, y in path_simplified]
        
        return None

    def simplify_path(self, path, MAP):
        path_simplified = simplify_collinear_points(path)
        path_line_of_sight = simplify_by_line_of_sight(path_simplified, MAP)
        return ramer_douglas_peucker(path_line_of_sight, 0.5)