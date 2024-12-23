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
import matplotlib.pyplot as plt
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


# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 1e-4
ENTROPY_BETA = 0.1
NB_EPISODES = 500
MAX_STEPS = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OccupancyGrid(Grid):
    """Enhanced occupancy grid for exploration and obstacle detection."""

    def __init__(self, size_area_world, resolution: float, lidar):
        super().__init__(size_area_world=size_area_world, resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution
        self.lidar = lidar

        self.x_max_grid: int = int(self.size_area_world[0] / self.resolution + 0.5)
        self.y_max_grid: int = int(self.size_area_world[1] / self.resolution + 0.5)

        self.grid = np.full((self.x_max_grid, self.y_max_grid), -5.0)
        self.last_position = None
        self.visited_zones = set()

    def update_grid(self, pose: Pose, step):
        """
        Update the occupancy grid based on the current position and lidar data.
        """
        EVERY_N = 3  # Use every nth lidar ray for efficiency
        LIDAR_DIST_CLIP = 40.0
        EMPTY_ZONE_VALUE = -0.5
        OBSTACLE_ZONE_VALUE = 2.0
        FREE_ZONE_VALUE = 1.0

        lidar_dist = self.lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()

        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        max_range = MAX_RANGE_LIDAR_SENSOR * 0.9

        # Update empty zones
        lidar_dist_empty = np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0)
        lidar_dist_empty_clip = np.minimum(lidar_dist_empty, max_range)
        points_x_empty = pose.position[0] + np.multiply(lidar_dist_empty_clip, cos_rays)
        points_y_empty = pose.position[1] + np.multiply(lidar_dist_empty_clip, sin_rays)

        for pt_x, pt_y in zip(points_x_empty, points_y_empty):
            self.add_value_along_line(
                pose.position[0], pose.position[1], pt_x, pt_y, EMPTY_ZONE_VALUE
            )

        # Update obstacle zones
        select_collision = lidar_dist < max_range
        points_x_obstacle = pose.position[0] + np.multiply(lidar_dist[select_collision], cos_rays[select_collision])
        points_y_obstacle = pose.position[1] + np.multiply(lidar_dist[select_collision], sin_rays[select_collision])
        self.add_points(points_x_obstacle, points_y_obstacle, OBSTACLE_ZONE_VALUE)

        # Mark the current position as free
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)

        # Clip grid values to avoid runaway updates
        if step % 50 == 0:
            self.grid = np.clip(self.grid, -10, 10)

    def compute_exploration_reward(self, x, y):
        """
        Compute a reward for exploring new or less explored areas.
        Penalize revisits or idling.
        """
        cell_x, cell_y = self._conv_world_to_grid(x, y)
        if 0 <= cell_x < self.x_max_grid and 0 <= cell_y < self.y_max_grid:
            cell_value = self.grid[cell_x, cell_y]
            if cell_value < 0:  # Reward exploration of new areas
                self.grid[cell_x, cell_y] += 1  # Mark as explored
                return 10  # Positive reward for exploration
            else:
                return -1  # Penalty for revisiting well-explored areas
        return -10  # Large penalty for going out of bounds

    def compute_movement_reward(self, pose: Pose):
        """
        Compute a reward based on the movement of the drone.
        Reward movement to new zones, penalize idling.
        """
        current_position = self._conv_world_to_grid(pose.position[0], pose.position[1])
        if self.last_position is None:
            self.last_position = current_position
            return 0  # No reward/penalty on the first movement

        if current_position == self.last_position:
            return -5  # Penalize staying in the same zone

        self.last_position = current_position
        return 5  # Reward moving to a new zone

    def mark_zone_as_visited(self, world_x, world_y):
        """Mark the zone as visited and return if it's a new zone."""
        cell_x, cell_y = self._conv_world_to_grid(world_x, world_y)
        if 0 <= cell_x < self.x_max_grid and 0 <= cell_y < self.y_max_grid:
            zone = (cell_x//10, cell_y//10)
            if zone not in self.visited_zones:
                self.visited_zones.add(zone)
                return True  # New zone
        return False  # Already visited or out of bounds



class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim=4):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim + 3)  # 3 continuous means + 3 log_stds + 1 discrete logits
        )
        self.to(device)

    def forward(self, x):
        logits = self.fc(x)
        means = torch.tanh(logits[:, :3])  # Means for continuous actions (bounded in [-1, 1])
        log_stds = torch.clamp(logits[:, 3:6], -20, 2)  # Log stds for continuous actions
        discrete_logits = logits[:, 6:]  # Logits for discrete action
        return means, log_stds, discrete_logits




class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
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
        resolution = 2
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

    def get_orientation(self):
        return self.measured_compass_angle()

    def update_pose(self):
        pos = self.measured_gps_position()
        angle = self.measured_compass_angle()
        self.pose.position = pos
        self.pose.orientation = angle

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass


def preprocess_lidar(lidar_values):
    if lidar_values is None or len(lidar_values) == 0:
        return np.zeros(90)  # Default to zeros if no data

    num_sectors = 90
    sector_size = len(lidar_values) // num_sectors

    aggregated = [np.mean(lidar_values[i * sector_size:(i + 1) * sector_size]) for i in range(num_sectors)]
    aggregated = np.nan_to_num(aggregated, nan=300, posinf=300, neginf=0)  # Handle NaNs or inf
    aggregated = np.clip(aggregated, 0, 300) / 300.0  # Normalize between 0 and 1

    assert not np.isnan(aggregated).any(), "Preprocessed LiDAR data contains NaN"
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



def select_action(policy, state):
    state = torch.FloatTensor(state).to(device).unsqueeze(0)
    means, log_stds, discrete_logits = policy(state)

    # Sample continuous actions
    stds = torch.exp(log_stds)
    sampled_continuous_actions = means + torch.randn_like(means) * stds

    # Clamp continuous actions to valid range
    continuous_actions = torch.clamp(sampled_continuous_actions, -1.0, 1.0)

    # Compute log probabilities for continuous actions
    log_probs_continuous = -0.5 * (((sampled_continuous_actions - means) / (stds + 1e-8)) ** 2 + 2 * log_stds + math.log(2 * math.pi))
    log_probs_continuous = log_probs_continuous.sum(dim=1)

    # Compute discrete action probability
    discrete_action_prob = torch.sigmoid(discrete_logits).squeeze()
    discrete_action = int(discrete_action_prob.item() > 0.5)

    # Log probability for discrete action
    log_prob_discrete = torch.log(discrete_action_prob + 1e-8) if discrete_action == 1 else torch.log(1 - discrete_action_prob + 1e-8)

    # Combine actions
    action = torch.cat([continuous_actions, torch.tensor([[discrete_action]], device=device)], dim=1).squeeze()
    log_prob = log_probs_continuous + log_prob_discrete
    return action.detach().cpu().numpy(), log_prob



def find_largest_opening(lidar_values):
    """
    Find the angle and width of the largest opening from LiDAR data.
    """
    num_sectors = 2  # Divide LiDAR data into sectors
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

def compute_reward(is_collision, found, occupancy_grid, pose, time_penalty):
    reward = 0

    # Penalize collisions heavily
    if is_collision:
        reward -= 50

    # Reward movement toward the largest opening
    #largest_opening_angle, _ = find_largest_opening(lidar_data)
    #angular_difference = abs(largest_opening_angle - orientation)
    #angular_difference = min(angular_difference, 2 * np.pi - angular_difference)
    #reward += 2 * (1 - angular_difference / np.pi)

    # Reward for entering new zones
    x, y = pose.position[0], pose.position[1]
    if occupancy_grid.mark_zone_as_visited(x, y):
        reward += 20  # Reward for exploring a new zone
    else:
        reward -= 1

    if found:
        reward += 100

    # Penalize idling or lack of movement
    reward -= time_penalty

    return reward


def compute_returns(rewards):
    returns = []
    g = 0
    for reward in reversed(rewards):
        g = reward + GAMMA * g
        returns.insert(0, g)
    return returns




policy_net = PolicyNetwork(state_dim=94, action_dim=4)  # 90 Lidar + 4 Semantic
value_net = ValueNetwork(state_dim=94)
optimizer_policy = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
optimizer_value = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)
policy_net.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)


def optimize_batch(states, actions, returns, batch_size=16, entropy_beta=0.1):
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    returns = np.array(returns, dtype=np.float32)

    dataset_size = len(states)
    for start_idx in range(0, dataset_size, batch_size):
        end_idx = min(start_idx + batch_size, dataset_size)

        states_batch = torch.tensor(states[start_idx:end_idx], device=device)
        actions_batch = torch.tensor(actions[start_idx:end_idx], device=device)
        returns_batch = torch.tensor(returns[start_idx:end_idx], device=device)

        # Forward pass
        means, log_stds, discrete_logits = policy_net(states_batch)
        values = value_net(states_batch).squeeze()

        # Separate continuous and discrete actions
        continuous_actions_t = actions_batch[:, :3]
        discrete_actions_t = actions_batch[:, 3].long()

        # Continuous action log probabilities
        stds = torch.exp(log_stds)
        log_probs_continuous = -0.5 * (((continuous_actions_t - means) / (stds + 1e-8)) ** 2 + 2 * log_stds + math.log(2 * math.pi))
        log_probs_continuous = log_probs_continuous.sum(dim=1)

        # Discrete action log probabilities
        discrete_action_prob = torch.sigmoid(discrete_logits).squeeze()
        log_probs_discrete = torch.log(discrete_action_prob + 1e-8) * discrete_actions_t + \
                             torch.log(1 - discrete_action_prob + 1e-8) * (1 - discrete_actions_t)

        # Total log probabilities
        log_probs = log_probs_continuous + log_probs_discrete

        # Compute advantages
        advantages = returns_batch - values.detach()

        # Policy loss
        policy_loss = -(log_probs * advantages).mean()

        # Entropy regularization
        entropy_loss = -entropy_beta * (log_stds + 0.5 * math.log(2 * math.pi * math.e)).sum(dim=1).mean()
        total_policy_loss = policy_loss + entropy_loss

        # Value loss
        value_loss = nn.functional.mse_loss(values, returns_batch)

        # Backpropagation
        optimizer_policy.zero_grad()
        total_policy_loss.backward()
        optimizer_policy.step()

        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()


def visualize_occupancy_grid(grid, episode, step):
    """
    Visualize the occupancy grid using Matplotlib.
    """
    plt.imshow(grid.T, origin='lower', cmap='gray', interpolation='nearest')
    plt.colorbar(label="Occupancy Value")
    plt.title(f"Occupancy Grid - Episode {episode}, Step {step}")
    plt.xlabel("X-axis (grid)")
    plt.ylabel("Y-axis (grid)")
    plt.show(block=False)
    plt.pause(0.1)
    plt.clf()


def train():
    map_training = MyMapIntermediate01()
    playground = map_training.construct_playground(drone_type=MyDrone)
    rewards_per_episode = []

    for episode in range(NB_EPISODES):
        playground.reset()
        for drone in map_training.drones:
            drone.occupancy_grid = OccupancyGrid(size_area_world=drone.size_area, resolution=10.0, lidar=drone.lidar())
        #gui = GuiSR(playground=playground, the_map=map_training, draw_semantic_rays=True)
        done = False
        states, actions, rewards = [], [], []
        total_reward = 0
        step = 0
        nb_rescue = 0
        actions_drones = {}
        playground.step()

        while not done and step < MAX_STEPS and all([drone.drone_health>0 for drone in map_training.drones]):
            step += 1
            # gui.run()  # Run GUI for visualization
            for drone in map_training.drones:
                lidar_data = preprocess_lidar(drone.lidar_values())
                semantic_data = preprocess_semantic(drone.semantic_values())
                state = np.concatenate([lidar_data, semantic_data])
                action, _ = select_action(drone.policy_net, state)
                actions_drones[drone] = process_actions(action)
                orientation = drone.get_orientation()

                reward = compute_reward(
                    is_collision=min(drone.lidar_values()) < 20,
                    found = semantic_data[0]*300 <40,
                    occupancy_grid=drone.occupancy_grid,
                    pose=drone.pose,
                    time_penalty=1
                )
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                total_reward += reward
                nb_rescue += int(drone.is_inside_return_area)

                done = semantic_data[0]*300<100
                if done:
                    print("found wounded !")
            playground.step(actions_drones)
            for drone in map_training.drones:
                drone.update_pose()  # Update drone pose after the step
                drone.occupancy_grid.update_grid(drone.pose, step)

        if any([drone.drone_health<=0 for drone in map_training.drones]):
            map_training = MyMapIntermediate01()
            playground = map_training.construct_playground(drone_type=MyDrone)
        # Optimize the policy and value networks in batches
        returns = compute_returns(rewards)
        optimize_batch(states, actions, returns, batch_size=16)
        rewards_per_episode.append(total_reward)

        del states, actions, rewards

        if episode % 10 == 1:
            print(f"Episode {episode}, Reward: {total_reward}, Mean Last 100 Rewards: {np.mean(rewards_per_episode[-100:])}")
            torch.save(policy_net.state_dict(), 'policy_net.pth')
            torch.save(value_net.state_dict(), 'value_net.pth')
            #visualize_occupancy_grid(map_training.drones[0].occupancy_grid.grid, episode, step)
            #cv2.imshow("Occupancy Grid", map_training.drones[0].occupancy_grid.zoomed_grid)
            #cv2.waitKey(1)

        if np.mean(rewards_per_episode[-100:]) > 10000:
            print(f"Training solved in {episode} episodes!")
            break


if __name__ == "__main__":
    train()