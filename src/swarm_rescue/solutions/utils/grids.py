import numpy as np
from scipy.ndimage import label
import cv2
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR
from solutions.utils.pose import Pose
from spg_overlay.utils.grid import Grid
from solutions.utils.messages import DroneMessage

class OccupancyGrid(Grid):
    """Self updating occupancy grid"""

    OBSTACLE = 1
    FREE = 0
    UNDISCOVERED = -2

    class Frontier:
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
            return np.array([sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)])

        def size(self):
            """
            Return the number of cells in the frontier.
            """
            return len(self.cells)

    def __init__(self,
                 size_area_world,
                 resolution: float,
                 lidar):
        super().__init__(size_area_world=size_area_world,
                         resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution

        self.lidar = lidar

        self.x_max_grid: int = int(self.size_area_world[0] / self.resolution
                                   + 0.5)
        self.y_max_grid: int = int(self.size_area_world[1] / self.resolution
                                   + 0.5)

        self.initial_cell = None
        self.initial_cell_value = None

        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))
        self.zoomed_grid = np.empty((self.x_max_grid, self.y_max_grid))

        self.frontiers = []

    def set_initial_cell(self, world_x, world_y):
        """
        Store the cell that corresponds to the initial drone position 
        This should be called once the drone initial position is known.
        """
        cell_x, cell_y = self._conv_world_to_grid(world_x, world_y)
        
        if 0 <= cell_x < self.x_max_grid and 0 <= cell_y < self.y_max_grid:
            self.initial_cell = (cell_x, cell_y)
    
    def to_binary_map(self):
        """
        Convert the probabilistic occupancy grid into a binary grid.
        OBSTACLE = 1
        FREE = 0
        UNDISCOVERED = -2
        Cells with value > 0 are considered obstacles.
        Cells with value < 0 are considered free.
        Cells with value = 0 are considered undiscovered
        """
        #print(np.count_nonzero(self.grid < 0))
        binary_map = np.zeros_like(self.grid, dtype=int)
        binary_map[self.grid > 0] = self.OBSTACLE
        binary_map[self.grid == 0] = self.UNDISCOVERED
        return binary_map
    
    def to_update(self, pose: Pose):
        """
        Returns the list of things to update on the grid
        """
        to_update = []

        EVERY_N = 3
        LIDAR_DIST_CLIP = 40.0
        EMPTY_ZONE_VALUE = -0.602
        OBSTACLE_ZONE_VALUE = 2.0
        FREE_ZONE_VALUE = -4.0

        lidar_dist = self.lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()

        # Compute cos and sin of the absolute angle of the lidar
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        max_range = MAX_RANGE_LIDAR_SENSOR * 0.9 # pk ? 

        # For empty zones
        # points_x and point_y contains the border of detected empty zone
        # We use a value a little bit less than LIDAR_DIST_CLIP because of the
        # noise in lidar
        lidar_dist_empty = np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0)
        # All values of lidar_dist_empty_clip are now <= max_range
        lidar_dist_empty_clip = np.minimum(lidar_dist_empty, max_range)
        points_x = pose.position[0] + np.multiply(lidar_dist_empty_clip,
                                                  cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist_empty_clip,
                                                  sin_rays)

        for pt_x, pt_y in zip(points_x, points_y):
            to_update.append(DroneMessage(
                            subject=DroneMessage.Subject.MAPPING,
                            code=DroneMessage.Code.LINE,
                            arg=(pose.position[0], pose.position[1], pt_x, pt_y, EMPTY_ZONE_VALUE))
                            )

        # For obstacle zones, all values of lidar_dist are < max_range
        select_collision = lidar_dist < max_range

        points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)

        points_x = points_x[select_collision]
        points_y = points_y[select_collision]

        
        to_update.append(DroneMessage(
                        subject=DroneMessage.Subject.MAPPING,
                        code=DroneMessage.Code.POINTS,
                        arg=(points_x, points_y, OBSTACLE_ZONE_VALUE))
                        )

        # the current position of the drone is free !
        to_update.append(DroneMessage(
                        subject=DroneMessage.Subject.MAPPING,
                        code=DroneMessage.Code.POINTS,
                        arg=(pose.position[0], pose.position[1], FREE_ZONE_VALUE))
                        )

        return to_update

    def update(self, to_update):
        """
        Bayesian map update with new observation
        lidar : lidar data
        pose : corrected pose in world coordinates
        """
        THRESHOLD_MIN = -40
        THRESHOLD_MAX = 40

        for message in to_update:
            # Ensure the message is a valid DroneMessage instance
            if not isinstance(message, DroneMessage):
                raise ValueError("Invalid message type. Expected a DroneMessage instance.")

            code = message.code
            arg = message.arg

            if code == DroneMessage.Code.LINE:
                self.add_value_along_line(*arg)
            elif code == DroneMessage.Code.POINTS:
                self.add_points(*arg)
            else:
                raise ValueError(f"Unknown code in DroneMessage: {code}")

        # Threshold values in the grid
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)


        # # Restore the initial cell value # That could have been set to free or empty
        # if self.initial_cell and self.initial_cell_value is not None:
        #     cell_x, cell_y = self.initial_cell
        #     if 0 <= cell_x < self.x_max_grid and 0 <= cell_y < self.y_max_grid:
        #         self.grid[cell_x, cell_y] = self.initial_cell_value

        # compute zoomed grid for displaying
        self.zoomed_grid = self.grid.copy()
        
        new_zoomed_size = (int(self.size_area_world[1] * 0.5),
                           int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_zoomed_size,
                                      interpolation=cv2.INTER_NEAREST)
    
    def frontiers_update(self):
        binary_map = self.to_binary_map()

        # Différences sur les axes X et Y
        diff_x = np.diff(binary_map, axis=1)
        diff_y = np.diff(binary_map, axis=0)

        # Détection des frontières entre FREE et UNDISCOVERED
        boundaries_x = np.abs(diff_x) == 2
        boundaries_y = np.abs(diff_y) == 2

        # Combinaison des résultats
        boundaries_map = np.pad(boundaries_x, ((0, 0), (0, 1))) | np.pad(boundaries_y, ((0, 1), (0, 0)))
        boundaries_map = boundaries_map * (binary_map==self.UNDISCOVERED)    # Frontier with width one

        structure = [[1,1,1]]*3 # Two boundary points are in the same frontier if they are adjacent (even diagonally)
        labeled_array, num_features = label(boundaries_map, structure)

        # Extraction des points de chaque frontière
        frontiers = [np.argwhere(labeled_array == i) for i in range(1, num_features + 1)]
        self.frontiers = [self.Frontier(cells) for cells in frontiers]
    
    def closest_centroid_frontier(self, pose: Pose):
        """
        Return the centroid of the frontier that is closest to pose
        """
        x_world,y_world = pose.position[0],pose.position[1]
        pos_drone_grid = np.array(self._conv_world_to_grid(x_world,y_world))
        centroid_closest_frontier_grid = min( (frontier.compute_centroid() for frontier in self.frontiers),
                                        key=lambda c: np.linalg.norm(c - pos_drone_grid) )
        centroid_closest_frontier_world = self._conv_grid_to_world(*centroid_closest_frontier_grid)
        return centroid_closest_frontier_world