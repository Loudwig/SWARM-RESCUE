import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import sys
from pathlib import Path
from enum import Enum
import matplotlib.pyplot as plt
import gc
from typing import List, Type

# Insert the parent directory of the current file's directory into sys.path.
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
ENTROPY_BETA = 0.01
EPSILON_CLIP = 0.2
LAM = 0.95
NB_EPISODES = 400
MAX_STEPS = 200
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
        self.last_position = Pose()
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
        #if step % 50 == 0:
            #self.grid = np.clip(self.grid, -10, 10)

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
            zone = (cell_x//20, cell_y//20)
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
            nn.Linear(128, action_dim + 3)  # 3 continuous + 3 log_stds + 1 discrete logits
        )
        self.to(device)

    def forward(self, x):
        logits = self.fc(x)
        means = torch.tanh(logits[:, :3])
        log_stds = torch.clamp(logits[:, 3:6], -20, 2)
        discrete_logits = logits[:, 6:]
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
        self.init_pose = Pose()


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
        return np.zeros(45)  # Default to zeros if no data

    num_sectors = 45
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


def compute_reward(drone, is_collision, found, occupancy_grid, pose, semantic_data, time_penalty):
    reward = 0

    # Penalize collisions heavily
    if is_collision:
        reward -= 100

    # Reward exploration
    x, y = round(pose.position[0]), round(pose.position[1])


    # Reward movement to new zones
    current_position = x//10, y//10
    last_zone = occupancy_grid.last_position.position[0]//10, occupancy_grid.last_position.position[1]//10
    if last_zone == current_position:
        reward -= 5  # Penalize staying in the same position
    else:
        reward += 5  # Small reward for movement

    distance_from_origin = np.sqrt((x-drone.init_pose.position[0]) ** 2 + (y-drone.init_pose.position[1]) ** 2)
    reward += 0.1 * distance_from_origin  # Scale the reward for distance

    if occupancy_grid.mark_zone_as_visited(x, y):
        reward += 5 * distance_from_origin  # Reward for exploring a new zone
    else:
        reward -= 10  # Penalize revisiting explored areas

    # Reward progress towards wounded
    nearest_wounded = semantic_data[0]*300  # Assume preprocessed semantic data contains distance
    if nearest_wounded < 300:
        progress_toward_wounded = max(0, 300 - nearest_wounded) / 300.0  # Normalize
        reward += 100 * progress_toward_wounded  # Reward scaled by progress

    # Reward for finding the wounded
    if found:
        reward += 500  # Large reward for completing the objective

    # Penalize idleness
    reward -= time_penalty

    return reward



policy_net = PolicyNetwork(state_dim=49, action_dim=4)
value_net = ValueNetwork(state_dim=49)
optimizer_policy = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
optimizer_value = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)

# PPO Functions
def compute_gae(rewards, values, next_values, dones):
    advantages = []
    gae = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * next_values[step] * (1 - dones[step]) - values[step]
        gae = delta + GAMMA * LAM * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns

def optimize_ppo(states, actions, log_probs, returns, advantages, batch_size=16):
    dataset_size = len(states)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    for start_idx in range(0, dataset_size, batch_size):
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]

        states_batch = torch.tensor(np.array(states)[batch_indices], dtype=torch.float32).to(device)
        actions_batch = torch.FloatTensor(np.array(actions)[batch_indices]).to(device)
        log_probs_batch = torch.tensor(
            np.array([log_prob.detach().cpu().numpy() for log_prob in log_probs])[batch_indices],
            dtype=torch.float32).to(device)
        returns_batch = torch.FloatTensor(np.array(returns)[batch_indices]).to(device)
        advantages_batch = torch.FloatTensor(np.array(advantages)[batch_indices]).to(device)

        # Forward pass through networks
        means, log_stds, discrete_logits = policy_net(states_batch)
        values = value_net(states_batch).squeeze()

        # Compute new log probabilities
        stds = torch.exp(log_stds)
        new_log_probs_continuous = -0.5 * (((actions_batch[:, :3] - means) / stds) ** 2 + 2 * log_stds + math.log(2 * math.pi))
        new_log_probs_continuous = new_log_probs_continuous.sum(dim=1)
        discrete_action_prob = torch.sigmoid(discrete_logits).squeeze()
        new_log_probs_discrete = torch.log(discrete_action_prob + 1e-8) * actions_batch[:, 3] + \
                                 torch.log(1 - discrete_action_prob + 1e-8) * (1 - actions_batch[:, 3])

        new_log_probs = new_log_probs_continuous + new_log_probs_discrete
        ratios = torch.exp(new_log_probs - log_probs_batch)

        # PPO loss
        surr1 = ratios * advantages_batch
        surr2 = torch.clamp(ratios, 1 - EPSILON_CLIP, 1 + EPSILON_CLIP) * advantages_batch
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy regularization
        entropy = -0.5 * (1 + 2 * log_stds + math.log(2 * math.pi)).mean()
        policy_loss -= ENTROPY_BETA * entropy

        # Value loss
        value_loss = nn.functional.mse_loss(values, returns_batch)

        # Backpropagation
        optimizer_policy.zero_grad()
        policy_loss.backward()
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


def compute_returns(rewards, gamma=0.99):
    """
    Compute the cumulative discounted rewards (returns) for each timestep.

    Args:
        rewards (list or np.array): A list of rewards from an episode.
        gamma (float): Discount factor (default: 0.99).

    Returns:
        list: A list of cumulative returns for each timestep.
    """
    returns = []
    g = 0  # This will hold the running cumulative reward
    for reward in reversed(rewards):
        g = reward + gamma * g  # Accumulate discounted reward
        returns.insert(0, g)  # Insert at the beginning to reverse the order
    return returns


def train():
    map_training = MyMapIntermediate01()
    playground = map_training.construct_playground(drone_type=MyDrone)
    rewards_per_episode = []

    batch_size = 128  # Mini-batch size for optimization

    for episode in range(NB_EPISODES):
        playground.reset()
        for drone in map_training.drones:
            drone.grid = OccupancyGrid(size_area_world=drone.size_area, resolution=10.0, lidar=drone.lidar())
            drone.init_pose.position = drone.measured_gps_position()
        batch_states, batch_actions, batch_log_probs, batch_rewards = [], [], [], []
        total_reward, step = 0, 0
        done = False

        while not done and step < MAX_STEPS and all([drone.drone_health>0 for drone in map_training.drones]):
            step += 1
            actions_drones = {}
            for drone in map_training.drones:
                lidar_data = preprocess_lidar(drone.lidar_values())
                semantic_data = preprocess_semantic(drone.semantic_values())
                state = np.concatenate([lidar_data, semantic_data])
                action, log_prob = select_action(drone.policy_net, state)

                actions_drones[drone] = process_actions(action)
                reward = compute_reward(
                    drone=drone,
                    is_collision=min(drone.lidar_values()) < 40,
                    found=semantic_data[0] * 300 < 200,
                    occupancy_grid=drone.grid,
                    pose=drone.pose,
                    semantic_data=semantic_data,
                    time_penalty=1
                )

                batch_states.append(state)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                batch_rewards.append(reward)
                total_reward += reward
                done = semantic_data[0]*300<100
                if done:
                    print("found wounded !")

            playground.step(actions_drones)
            for drone in map_training.drones:
                drone.grid.last_position.position = drone.pose.position
                drone.update_pose()  # Update drone pose after the step
                drone.grid.update_grid(drone.pose, step)

            # Incrementally optimize networks when enough data is collected
            if len(batch_rewards) >= batch_size or done or step == MAX_STEPS:
                batch_returns = compute_returns(batch_rewards)
                batch_advantages = [ret - val for ret, val in zip(batch_returns, batch_rewards)]
                optimize_ppo(batch_states, batch_actions, batch_log_probs, batch_returns, batch_advantages)

                # Clear batch data after optimization
                batch_states, batch_actions, batch_log_probs, batch_rewards = [], [], [], []

        rewards_per_episode.append(total_reward)



        if any([drone.drone_health<=0 for drone in map_training.drones]):
            map_training = MyMapIntermediate01()
            playground = map_training.construct_playground(drone_type=MyDrone)

        # Log performance and visualize progress


        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {total_reward}")
            print(map_training.drones[0].grid.visited_zones)
            torch.save(policy_net.state_dict(), 'policy_net.pth')
            torch.save(value_net.state_dict(), 'value_net.pth')
            visualize_occupancy_grid(map_training.drones[0].grid.grid, episode, step)

        for drone in map_training.drones:
            del drone.grid.grid

        if np.mean(rewards_per_episode[-10:]) > 10000:
            print(f"Solved in {episode} episodes!")
            break


if __name__ == "__main__":
    train()
