import numpy as np
from spg_overlay.utils.grid import Grid

class OccupancyGrid(Grid):
    """Simple occupancy grid"""

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
        self.position_grid = np.zeros((self.x_max_grid, self.y_max_grid))
        self.zoomed_grid = np.empty((self.x_max_grid, self.y_max_grid))
        
        self.grid_score = 0
        self.grid_previous_score = 0
        self.exploration_score = 0
    
    def compute_grid_score(self):
        """
        Compute the score of the grid
        """
        seuil = 0.2
        self.grid_score = np.sum(abs(self.grid) >= seuil) / (self.x_max_grid * self.y_max_grid)

    def compute_exploration_score(self):
        self.exploration_score = self.grid_score - self.grid_previous_score

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
        1 = obstacle
        0 = free
        Cells with value >= 0 are considered obstacles.
        Cells with value < 0 are considered free.
        """
        #print(np.count_nonzero(self.grid < 0))
        binary_map = np.zeros_like(self.grid, dtype=int)
        binary_map[self.grid >= 0] = 1
        return binary_map
    
    def update_grid(self, pose: Pose):

        self.grid_previous_score = self.grid_score

        self.position_grid = np.zeros((self.x_max_grid, self.y_max_grid))
        self.position_grid[self._conv_world_to_grid(pose.position[0]), self._conv_world_to_grid(pose.position[1])] = 1

        """
        Bayesian map update with new observation
        lidar : lidar data
        pose : corrected pose in world coordinates
        """
        normalisation = 40 
        THRESHOLD_MIN = -40/normalisation
        THRESHOLD_MAX = 40/normalisation

        EVERY_N = 3
        LIDAR_DIST_CLIP = 40.0
        EMPTY_ZONE_VALUE = -0.602/normalisation
        OBSTACLE_ZONE_VALUE = 2.0/normalisation
        FREE_ZONE_VALUE = -4.0/normalisation



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
            self.add_value_along_line(pose.position[0], pose.position[1],
                                      pt_x, pt_y,
                                      EMPTY_ZONE_VALUE)

        # For obstacle zones, all values of lidar_dist are < max_range
        select_collision = lidar_dist < max_range

        points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)

        points_x = points_x[select_collision]
        points_y = points_y[select_collision]


        self.add_points(points_x, points_y, OBSTACLE_ZONE_VALUE)

        # the current position of the drone is free !
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)

        # threshold values
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)
        self.compute_grid_score()
        self.compute_exploration_score()
        # compute zoomed grid for displaying
        self.zoomed_grid = self.grid.copy()
        
        new_zoomed_size = (int(self.size_area_world[1] * 0.5),
                           int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_zoomed_size,
                                      interpolation=cv2.INTER_NEAREST)
