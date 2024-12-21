import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import sys
from pathlib import Path
from enum import Enum
import cv2
from typing import List, Type



# Insert the parent directory of the current file's directory into sys.path.
# This allows Python to locate modules that are one level above the current
# script, in this case spg_overlay.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maps.map_intermediate_01 import MyMapIntermediate01

from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.grid import Grid


class OccupancyGrid(Grid):
    """Enhanced occupancy grid for exploration."""

    def __init__(self, size_area_world, resolution: float, lidar):
        super().__init__(size_area_world=size_area_world, resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution

        self.lidar = lidar

        self.x_max_grid: int = int(self.size_area_world[0] / self.resolution + 0.5)
        self.y_max_grid: int = int(self.size_area_world[1] / self.resolution + 0.5)

        self.initial_cell = None

        # Initialize grid with low values (unexplored regions)
        self.grid = np.full((self.x_max_grid, self.y_max_grid), -5.0)  # Start with low probability
        self.zoomed_grid = np.empty((self.x_max_grid, self.y_max_grid))

    def set_initial_cell(self, world_x, world_y):
        """Store the cell that corresponds to the initial drone position."""
        cell_x, cell_y = self._conv_world_to_grid(world_x, world_y)
        if 0 <= cell_x < self.x_max_grid and 0 <= cell_y < self.y_max_grid:
            self.initial_cell = (cell_x, cell_y)

    def to_binary_map(self):
        """
        Convert the probabilistic occupancy grid into a binary grid.
        1 = obstacle
        0 = free
        """
        binary_map = np.zeros_like(self.grid, dtype=int)
        binary_map[self.grid >= 0] = 1
        return binary_map

    def update_grid(self, pose: Pose):
        """
        Bayesian map update with new observation
        """
        EVERY_N = 3
        LIDAR_DIST_CLIP = 40.0
        EMPTY_ZONE_VALUE = -0.6
        OBSTACLE_ZONE_VALUE = 2.0
        FREE_ZONE_VALUE = -4.0

        lidar_dist = self.lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()

        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        max_range = MAX_RANGE_LIDAR_SENSOR * 0.9

        # For empty zones
        lidar_dist_empty = np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0)
        lidar_dist_empty_clip = np.minimum(lidar_dist_empty, max_range)
        points_x = pose.position[0] + np.multiply(lidar_dist_empty_clip, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist_empty_clip, sin_rays)

        for pt_x, pt_y in zip(points_x, points_y):
            self.add_value_along_line(pose.position[0], pose.position[1], pt_x, pt_y, EMPTY_ZONE_VALUE)

        # For obstacle zones
        select_collision = lidar_dist < max_range
        points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)
        points_x = points_x[select_collision]
        points_y = points_y[select_collision]
        self.add_points(points_x, points_y, OBSTACLE_ZONE_VALUE)

        # Mark the current position as free
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)

        # Clip values
        self.grid = np.clip(self.grid, -10, 10)

        # Update zoomed grid for visualization
        self.zoomed_grid = cv2.resize(self.grid, (self.x_max_grid // 2, self.y_max_grid // 2), interpolation=cv2.INTER_NEAREST)

    def is_unexplored(self, world_x, world_y):
        """
        Check if the given world position corresponds to an unexplored grid cell.
        """
        cell_x, cell_y = self._conv_world_to_grid(world_x, world_y)
        if 0 <= cell_x < self.x_max_grid and 0 <= cell_y < self.y_max_grid:
            return self.grid[cell_x, cell_y] < 0  # Negative values indicate unexplored areas
        return False

    def compute_exploration_reward(self, pose: Pose):
        """
        Compute a reward based on exploration of unexplored areas.
        """
        cell_x, cell_y = self._conv_world_to_grid(*pose.position)
        if 0 <= cell_x < self.x_max_grid and 0 <= cell_y < self.y_max_grid:
            value = self.grid[cell_x, cell_y]
            if value < 0:  # Reward exploration of unexplored areas
                self.grid[cell_x, cell_y] += 1  # Mark as explored
                return 10  # Exploration reward
        return -1  # Penalize revisiting explored areas



# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 1e-3
ENTROPY_BETA = 0.01
NB_EPISODES = 400
MAX_STEPS = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim=4):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.to(device)

    def forward(self, x):
        logits = self.fc(x)
        continuous_actions = torch.tanh(logits[:, :3])*1.0  # Continuous actions
        discrete_action = torch.sigmoid(logits[:, 3])  # Grasp action
        return continuous_actions, discrete_action


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.to(device)

    def forward(self, x):
        return self.fc(x)


class MyDrone(DroneAbstract):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.policy_net = policy_net
        self.state = self.Activity.SEARCHING_WOUNDED
        resolution = 8
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())
        self.pose = Pose()


    def control(self):
        lidar_data = preprocess_lidar(self.lidar_values())
        semantic_data = preprocess_semantic(self.semantic_values())
        state = np.concatenate([lidar_data, semantic_data])
        action, _ = select_action(self.policy_net, state)
        return process_actions(action)


    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass


def preprocess_lidar(lidar_values):
    """Groups lidar rays into 18 sectors and computes mean distance per sector."""
    if lidar_values is None or len(lidar_values) == 0:
        return np.zeros(18)  # Default to zeros if no data

    num_sectors = 18
    sector_size = len(lidar_values) // num_sectors

    aggregated = [np.mean(lidar_values[i * sector_size:(i + 1) * sector_size]) for i in range(num_sectors)]
    aggregated = np.nan_to_num(aggregated, nan=300, posinf=300, neginf=0)  # Handle NaNs or inf
    aggregated = np.clip(aggregated, 0, 300) / 300.0  # Normalize between 0 and 1
    return aggregated


def preprocess_semantic(semantic_values):
    """Processes semantic sensor output into usable features."""
    wounded_detected = []
    rescue_center_detected = []

    if semantic_values:
        for data in semantic_values:
            if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                wounded_detected.append((data.distance, data.angle))
            elif data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                rescue_center_detected.append((data.distance, data.angle))

    # Encode as feature vector: [nearest_wounded_distance, nearest_wounded_angle, rescue_center_distance, rescue_center_angle]
    nearest_wounded = min(wounded_detected, default=(300, 0), key=lambda x: x[0])
    nearest_rescue_center = min(rescue_center_detected, default=(300, 0), key=lambda x: x[0])

    processed = [
        nearest_wounded[0] / 300.0,  # Distance normalized
        nearest_wounded[1] / np.pi,  # Angle normalized to [-1, 1]
        nearest_rescue_center[0] / 300.0,
        nearest_rescue_center[1] / np.pi
    ]

    return processed


def process_actions(actions):
    return {
        "forward": actions[0],
        "lateral": actions[1],
        "rotation": actions[2],
        "grasper": actions[3]  # 0 or 1 for grasping
    }


def select_action(policy_net, state):
    state = torch.FloatTensor(state).to(device).unsqueeze(0)
    continuous_actions, discrete_action_prob = policy_net(state)
    # Clipping for numerical stability
    discrete_action_prob = torch.clamp(discrete_action_prob, 1e-6, 1 - 1e-6)

    continuous_actions = continuous_actions[0].cpu().detach().numpy()
    discrete_action = int(discrete_action_prob[0].item() > 0.5)  # Threshold 0.5
    action = np.concatenate([continuous_actions, [discrete_action]])

    log_prob = torch.log(discrete_action_prob if discrete_action == 1 else 1 - discrete_action_prob)
    return action, log_prob

def find_largest_opening(lidar_values):
    """
    Find the angle and width of the largest opening from LiDAR data.
    """
    num_sectors = 18  # Divide LiDAR data into sectors
    sector_size = len(lidar_values) // num_sectors
    aggregated_distances = [
        np.mean(lidar_values[i * sector_size:(i + 1) * sector_size]) for i in range(num_sectors)
    ]

    # Find the sector with the largest average distance
    max_distance = max(aggregated_distances)
    max_index = aggregated_distances.index(max_distance)

    # Convert sector index to angle (radians)
    sector_angle = 2 * np.pi * max_index / num_sectors

    return sector_angle, max_distance


def compute_reward(is_collision, lidar_data, occupancy_grid, pose, time_penalty):
    reward = 0

    # Penalize collisions
    if is_collision:
        reward -= 1000

    # Reward exploration
    exploration_reward = occupancy_grid.compute_exploration_reward(pose)
    reward += exploration_reward

    # Reward movement toward the largest opening
    largest_opening_angle, _ = find_largest_opening(lidar_data)
    angular_difference = abs(largest_opening_angle - pose.orientation)
    angular_difference = min(angular_difference, 2 * np.pi - angular_difference)
    reward += 5 * (1 - angular_difference / np.pi)

    if min(lidar_data)*300<40:
        reward -= 100

    # Penalize idling
    reward -= time_penalty

    return reward




def compute_returns(rewards,found):
    returns = [1000 if found else 0+rewards[-1]]
    g = 0
    for reward in reversed(rewards[:-1]):
        g = reward + GAMMA * g
        returns.insert(0, g)
    return returns




policy_net = PolicyNetwork(state_dim=22, action_dim=4)  # 18 Lidar + 2 Semantic
value_net = ValueNetwork(state_dim=22)
optimizer_policy = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
optimizer_value = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)
policy_net.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

def optimize_batch(states, actions, returns, batch_size=16):
    """
    Optimizes the policy and value networks in smaller batches to avoid memory overload.
    """
    # Convert lists to numpy arrays for efficient slicing and tensor conversion
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    returns = np.array(returns, dtype=np.float32)

    dataset_size = len(states)
    for start_idx in range(0, dataset_size, batch_size):
        end_idx = min(start_idx + batch_size, dataset_size)

        # Batch slicing
        states_batch = torch.tensor(states[start_idx:end_idx], device=device)
        actions_batch = torch.tensor(actions[start_idx:end_idx], device=device)
        returns_batch = torch.tensor(returns[start_idx:end_idx], device=device)

        # Forward pass
        continuous_pred, discrete_pred = policy_net(states_batch)
        values = value_net(states_batch).squeeze()

        # Separate continuous and discrete actions
        continuous_actions_t = actions_batch[:, :3]
        discrete_actions_t = actions_batch[:, 3].long()

        # Compute losses
        continuous_loss = nn.functional.mse_loss(continuous_pred, continuous_actions_t)
        discrete_log_probs = torch.log(discrete_pred + 1e-8) * discrete_actions_t + \
                             torch.log(1 - discrete_pred + 1e-8) * (1 - discrete_actions_t)
        discrete_loss = -discrete_log_probs.mean()

        advantages = returns_batch - values.detach()
        policy_loss = (continuous_loss + discrete_loss * ENTROPY_BETA).mean()

        value_loss = nn.functional.mse_loss(values, returns_batch)

        # Optimize Policy Network
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        # Optimize Value Network
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

        # Clear GPU memory (if applicable)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



def train():
    map_training = MyMapIntermediate01()
    base_playground = map_training.construct_playground(drone_type=MyDrone)
    rewards_per_episode = []

    for episode in range(NB_EPISODES):
        playground = base_playground
        for drone in map_training.drones:
            drone.occupancy_grid = OccupancyGrid(size_area_world=drone.size_area, resolution=10.0, lidar=drone.lidar())
            #drone.occupancy_grid.set_initial_cell(drone.measured_gps_position()[0], drone.measured_gps_position()[1])
        # gui = GuiSR(playground=playground, the_map=map_training, draw_semantic_rays=True)
        done = False
        states, actions, rewards = [], [], []
        total_reward = 0
        step = 0
        nb_rescue = 0

        while not done and step < MAX_STEPS:
            step += 1
            playground.step()
            # gui.run()  # Run GUI for visualization

            for drone in map_training.drones:
                lidar_data = preprocess_lidar(drone.lidar_values())
                semantic_data = preprocess_semantic(drone.semantic_values())
                state = np.concatenate([lidar_data, semantic_data])
                action, _ = select_action(drone.policy_net, state)
                current_angle = drone.measured_compass_angle()
                drone.occupancy_grid.update_grid(drone.pose)
                reward = compute_reward(
                    is_collision=min(drone.lidar_values()) < 20,
                    lidar_data=lidar_data,
                    occupancy_grid=drone.occupancy_grid,
                    pose=drone.pose,
                    time_penalty=1
                )
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                total_reward += reward
                nb_rescue += int(drone.is_inside_return_area)

                done = lidar_data[0]*300<40

        # Optimize the policy and value networks in batches
        returns = compute_returns(rewards, done)
        optimize_batch(states, actions, returns, batch_size=64)
        rewards_per_episode.append(total_reward)

        if episode % 10 == 1:
            print(f"Episode {episode}, Reward: {total_reward}, Mean Last 100 Rewards: {np.mean(rewards_per_episode[-100:])}")
            torch.save(policy_net.state_dict(), 'policy_net.pth')
            torch.save(value_net.state_dict(), 'value_net.pth')
            #cv2.imshow("Occupancy Grid", map_training.drones[0].occupancy_grid.zoomed_grid)
            #cv2.waitKey(1)

        if np.mean(rewards_per_episode[-100:]) > 10000:
            print(f"Training solved in {episode} episodes!")
            break


if __name__ == "__main__":
    train()