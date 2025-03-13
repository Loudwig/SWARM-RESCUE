import numpy as np
from scipy.ndimage import label
import cv2
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR
from solutions.utils.pose import Pose
from spg_overlay.utils.grid import Grid
from solutions.utils.astar import *
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from solutions.utils.dataclasses_config import *

from sklearn.cluster import DBSCAN


class OccupancyGrid(Grid):
    """Self updating occupancy grid"""
    OBSTACLE = GridParams.OBSTACLE
    FREE = GridParams.FREE
    UNDISCOVERED = GridParams.UNDISCOVERED

    class Frontier:
        def __init__(self, cells):
            """
            Initialize a frontier with a list of grid cells.
            :param cells: List of tuples [(x1, y1), (x2, y2), ...]
            """
            self.cells = cells

        def centroid(self):
            """
            Compute the centroid of the frontier.
            """
            if self.cells.size == 0:
                return None
            return np.mean(self.cells, axis=0).astype(int)


        def cell_closest_to_centroid(self):
            """
            Compute the cell closest to the centroid of the frontier among cells of the frontier.
            """
            if self.cells.size == 0:
                return None
            return self.cells[np.argmin(np.linalg.norm(self.cells - self.centroid(), axis=1))]

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

        WORLD_BORDERS_VALUE = GridParams.WORLD_BORDERS_VALUE
        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))
        # Set the value of all border cells to WORLD_BORDERS_VALUE so they are considered as obstacles
        self.grid[[0, -1], :] = WORLD_BORDERS_VALUE
        self.grid[:, [0, -1]] = WORLD_BORDERS_VALUE

        self.zoomed_grid = np.empty((self.x_max_grid, self.y_max_grid))

        self.frontier_connectivity_structure = np.ones((3, 3), dtype=int)  # Connects points that are adjacent (even diagonally)
        self.frontiers = []
    
    def to_ternary_map(self):
        """
        Convert the probabilistic occupancy grid into a ternary grid.
        Cells with value >= OBSTACLE_THRESHOLD are considered obstacles.
        Cells with value <= FREE_THRESHOLD are considered free.
        Cells with value = 0 are considered undiscovered
        """
        OBSTACLE_THRESHOLD = GridParams.OBSTACLE_THRESHOLD
        FREE_THRESHOLD = GridParams.FREE_THRESHOLD

        ternary_map = np.zeros_like(self.grid, dtype=int)
        ternary_map[self.grid >= OBSTACLE_THRESHOLD] = self.OBSTACLE
        ternary_map[self.grid <= FREE_THRESHOLD] = self.FREE
        ternary_map[(self.grid < OBSTACLE_THRESHOLD) & (self.grid > FREE_THRESHOLD)] = self.UNDISCOVERED
        return ternary_map

    def to_binary_map(self):
        """
        Convert the probabilistic occupancy grid into a binary grid.
        Cells with value > OBSTACLE_THRESHOLD are considered obstacles.
        Cells with value < FREE_THRESHOLD are considered free.
        """
        OBSTACLE_THRESHOLD = GridParams.OBSTACLE_THRESHOLD
        FREE_THRESHOLD = GridParams.FREE_THRESHOLD

        binary_map = np.zeros_like(self.grid, dtype=int)
        binary_map[self.grid >= OBSTACLE_THRESHOLD] = self.OBSTACLE
        binary_map[self.grid <= FREE_THRESHOLD] = self.FREE
        # Binary map : undiscovered are considered obstacles
        binary_map[ not(self.grid >= OBSTACLE_THRESHOLD or self.grid <= FREE_THRESHOLD) ] = self.OBSTACLE
        return binary_map
    
    def update(self, pose: Pose):
        """
        Returns the list of things to update on the grid
        Uses a ray casting algorithm with the lidar data
        """

        EVERY_N = GridParams.EVERY_N
        LIDAR_DIST_CLIP = GridParams.LIDAR_DIST_CLIP
        MAX_RANGE_LIDAR_SENSOR_FACTOR = GridParams.MAX_RANGE_LIDAR_SENSOR_FACTOR
        EMPTY_ZONE_VALUE = GridParams.EMPTY_ZONE_VALUE
        OBSTACLE_ZONE_VALUE = GridParams.OBSTACLE_ZONE_VALUE
        FREE_ZONE_VALUE = GridParams.FREE_ZONE_VALUE
        THRESHOLD_MIN = GridParams.THRESHOLD_MIN
        THRESHOLD_MAX = GridParams.THRESHOLD_MAX

        lidar_dist = self.lidar.get_sensor_values()[::EVERY_N].copy()   # Distance of each ray to the first obstacle it encounters (or max range if it doesn't)
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()  # Angle of each ray
        
        # Used to go from rays to points on the grid
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        # Any ray that has an associated lidar_dist greater than this threshold is considered to have no obstacle
        no_obstacle_ray_distance_threshold = MAX_RANGE_LIDAR_SENSOR * MAX_RANGE_LIDAR_SENSOR_FACTOR

        # Ensure coherent values to balance noise on lidar values
        processed_lidar_dist = np.clip(lidar_dist - LIDAR_DIST_CLIP, 0, no_obstacle_ray_distance_threshold)
        points_x = pose.position[0] + np.multiply(processed_lidar_dist,
                                                  cos_rays)
        points_y = pose.position[1] + np.multiply(processed_lidar_dist,
                                                  sin_rays)

        for pt_x, pt_y in zip(points_x, points_y):
            self.add_value_along_line(pose.position[0], pose.position[1], pt_x, pt_y, EMPTY_ZONE_VALUE)

        # Rays that collide obstacles are those that verify lidar_dist[ray] < max_confidence_range
        select_collision = lidar_dist < no_obstacle_ray_distance_threshold 
        
        points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)

        zone_drone_x , zone_drone_y = self.compute_near_drones_zone(pose)
        epsilon = 3
        for ind,v in enumerate(select_collision):
            if select_collision[ind] == True:
                if self.list_any_comparison_int(abs(zone_drone_x - points_x[ind]),epsilon) and self.list_any_comparison_int(abs(zone_drone_y - points_y[ind]),epsilon):
                    select_collision[ind] =  False
        
        points_x = points_x[select_collision]
        points_y = points_y[select_collision]

        self.add_points(points_x, points_y, OBSTACLE_ZONE_VALUE)
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)
        self.zoomed_grid = self.grid.copy()
        
        new_zoomed_size = (int(self.size_area_world[1] * 0.5),
                           int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_zoomed_size,
                                      interpolation=cv2.INTER_NEAREST)

    def frontiers_update(self):
        ternary_map = self.to_ternary_map()
        rows, cols = ternary_map.shape

        threshold = 2

        frontier_mask = np.zeros_like(ternary_map, dtype=bool)

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for i in range(rows):
            for j in range(cols):
                if ternary_map[i, j] != 0:
                    continue

                candidate_found = False
                valid_candidate = True

                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                        continue

                    if ternary_map[ni, nj] == -2:
                        candidate_found = True
                        for step in range(1, threshold + 1):
                            check_i = i + di * (step + 1)
                            check_j = j + dj * (step + 1)
                            if check_i < 0 or check_i >= rows or check_j < 0 or check_j >= cols:
                                break
                            if ternary_map[check_i, check_j] == 1:
                                valid_candidate = False
                                break
                        if not valid_candidate:
                            break

                # Mark the cell as a valid frontier if it touches unknown and passes the extended check
                if candidate_found and valid_candidate:
                    frontier_mask[i, j] = True

        # Label connected regions in the frontier mask using the provided connectivity structure
        labeled_array, num_features = label(frontier_mask, self.frontier_connectivity_structure)

        # Extract points from each connected component and filter by minimum frontier size
        frontiers = [np.argwhere(labeled_array == i) for i in range(1, num_features + 1)]
        self.frontiers = [self.Frontier(cells) for cells in frontiers if len(cells) >= GridParams.MIN_FRONTIER_SIZE]

    def cluster_frontiers_dbscan(self, eps=2, min_samples=3):
        """
        Clusters all frontier cells using DBSCAN and returns a list of clusters.
        Each cluster is represented as a dictionary with a centroid, the points, and cluster size.
        """
        self.frontiers_update()
        # Gather all frontier cells from the frontier objects
        all_frontier_cells = []
        for frontier in self.frontiers:
            for cell in frontier.cells:
                all_frontier_cells.append(cell)
        all_frontier_cells = np.array(all_frontier_cells)
        if len(all_frontier_cells) == 0:
            return []
        # Apply DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(all_frontier_cells)
        labels = db.labels_
        clusters = []
        for label_val in set(labels):
            if label_val == -1:  # Noise
                continue
            cluster_points = all_frontier_cells[labels == label_val]
            new_frontier = self.Frontier(cluster_points)
            clusters.append(new_frontier)
        return clusters

    def delete_frontier_artifacts(self, frontier):
        """
        Set to THRESHOLD_MAX (which relates to OBSTACLE) in the grid all cells of frontier
        """
        print("Deleting frontier artifacts")
        if frontier is not None:
            for cell in frontier.cells:
                self.grid[cell] = GridParams.FRONTIER_ARTIFACT_RESET_VALUE
    
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
            (frontier, frontier.centroid(), frontier.size()) for frontier in self.frontiers
        ]
        
        def interest_measure(frontier_data):
            _, centroid, size = frontier_data
            distance = np.linalg.norm(centroid - pos_drone_grid)
            return distance / (size + 1)**2
        
        closest_frontier, closest_centroid, _ = min(frontiers_with_size, key=interest_measure, default=(None, None))
        return (closest_frontier, closest_centroid)
    
    def pos_closest_to_centroid(self, frontier: Frontier):
        """
        Returns the position of the frontier closest to its centroid.
        """
        return self._conv_grid_to_world(*frontier.cell_closest_to_centroid())

    def compute_safest_path(self, start_pos, target_pos, max_inflation):
        """
        Returns the path, if it exists, that joins drone's position to target_cell
        while approaching the least possible any wall
        start_pos, target_pos : WORLD COORDINATES
        start_cell, target_cell : GRID COORDINATES
        """
        MAP = self.to_ternary_map()

        start_cell, target_cell = self._conv_world_to_grid(*start_pos), self._conv_world_to_grid(*target_pos)

        for inflation in range(max_inflation, 0, -1):   # Decreasing inflation to find the safest path
            MAP_inflated = inflate_obstacles(MAP, inflation)
            start_x, start_y = next_point_free(MAP_inflated, *start_cell, max_inflation - inflation)
            end_x, end_y = next_point_free(MAP_inflated, *target_cell, max_inflation - inflation)

            path = a_star_search(MAP_inflated, (start_x, start_y), (end_x, end_y))

            if path:
                path_simplified = self.simplify_path(path, MAP_inflated) or [start_cell]
                return [self._conv_grid_to_world(x, y) for x, y in path_simplified]
        
        return None

    def path_distance(self, path) :
        if path is None:
            return 100000
        diff = [math.dist(path[i+1],path[i]) for i in range(len(path)-1)]
        return np.sum(diff)

    def simplify_path(self, path, MAP):
        path_simplified = simplify_collinear_points(path)
        path_line_of_sight = simplify_by_line_of_sight(path_simplified, MAP)
        return ramer_douglas_peucker(path_line_of_sight, 0.5)
    
    def compute_near_drones_zone(self,pose:Pose):
        detection_semantic = self.semantic.get_sensor_values().copy()
        zone_drone_x = []
        zone_drone_y = []
        for data in detection_semantic:
            if (data.entity_type == DroneSemanticSensor.TypeEntity.DRONE):
                cos_rays = np.cos(data.angle + pose.orientation)
                sin_rays = np.sin(data.angle + pose.orientation)
                
                zone_drone_x.append(pose.position[0] + np.multiply(data.distance, cos_rays))
                zone_drone_y.append(pose.position[1] + np.multiply(data.distance, sin_rays))
        return zone_drone_x,zone_drone_y
    
    def list_any_comparison_int(self,L,i):
        for x in L : 
            if x < i : return True
        return False


    def merge_maps(self, other_map,confiance): # other_map is not a class but just the grid.
        """
        Merge the other map into the current map
        """
        self.grid = self.grid*(1-confiance) + other_map*(confiance)