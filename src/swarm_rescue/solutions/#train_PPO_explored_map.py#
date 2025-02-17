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

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 1e-4
ENTROPY_BETA = 0.05
EPSILON_CLIP = 0.2
LAM = 0.95
NB_EPISODES = 1000
MAX_STEPS = 800
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import cv2
from spg.view import TopDownView


def _create_black_white_image(img_playground):
    """
    Converts the playground image to a binary map (walls=1, free space=0).
    """
    map_color = cv2.normalize(src=img_playground, dst=None, alpha=0, beta=255,
                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    map_gray = cv2.cvtColor(map_color, cv2.COLOR_BGR2GRAY)
    _, binary_map = cv2.threshold(map_gray, 127, 1, cv2.THRESH_BINARY)
    return binary_map


class ExploredMapOptimized:
    """
    GPU-accelerated version of the ExploredMap class.
    Tracks explored areas and computes exploration scores using PyTorch tensors.
    """

    def __init__(self, playground, device_available=device):
        """
        Initializes the explored map using the playground.
        Args:
            playground: An instance of ClosedPlayground.
            device: Device for computation (e.g., "cuda" or "cpu").
        """
        self.device = torch.device(device_available)
        self._initialize_map(playground)
        self._reset_maps()
        self._cached_score = None  # Cache for exploration score

        # Increase the kernel radius if you want the "explored zone" to be bigger around the visited line.
        # This effectively reduces the need to visit every cell to count as "explored."
        self._precomputed_kernel = self._create_circular_kernel(radius=200)

    def _initialize_map(self, playground):
        """
        Initializes the base map using the playground's top-down view.
        Args:
            playground: An instance of ClosedPlayground.
        """
        view = TopDownView(playground=playground, zoom=1)
        view.update()
        img_playground = cv2.flip(view.get_np_img(), 0)
        binary_map = _create_black_white_image(img_playground)

        # --------------------
        # COARSER MAP RESIZING
        # --------------------
        # If your map is e.g. 2000x2000, resizing by factor 0.5 => 1000x1000
        # You can adapt interpolation / factor as you like.
        height, width = binary_map.shape
        resize_factor = 0.5
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)
        binary_map_small = cv2.resize(
            binary_map,
            (new_width, new_height),
            interpolation=cv2.INTER_NEAREST
        )

        # Convert the binary map to a PyTorch tensor for GPU acceleration
        self.map_playground = torch.tensor(binary_map_small, dtype=torch.float32, device=self.device)
        self._map_shape = self.map_playground.shape
        self._total_reachable_pixels = torch.sum(self.map_playground == 0).float().item()

        # Store factors to convert real pose coords -> map idx
        # so we know how to update lines in the coarser map.
        self._orig_width = width
        self._orig_height = height
        self._resize_factor = resize_factor

    def _reset_maps(self):
        """
        Resets the exploration maps and counters.
        """
        self._map_explo_lines = torch.ones(self._map_shape, dtype=torch.float32, device=self.device)
        self._map_explo_zones = torch.zeros(self._map_shape, dtype=torch.float32, device=self.device)
        self._cached_score = None  # Reset cached score

    def reset(self):
        """
        Resets all internal maps to initial states.
        """
        self._reset_maps()

    def update_drones(self, drones):
        """
        Updates the exploration lines based on the drone positions.
        Args:
            drones: List of DroneAbstract instances with positions.
        """
        height, width = self._map_explo_lines.shape
        for drone in drones:
            position = drone.true_position()
            # Convert real position to map grid in the smaller map
            # The + width/2, + height/2 logic is typical for center-based indexing
            grid_x = int((position[0] + self._orig_width / 2) * self._resize_factor)
            grid_y = int((-position[1] + self._orig_height / 2) * self._resize_factor)

            if 0 <= grid_x < width and 0 <= grid_y < height:
                self._map_explo_lines[grid_y, grid_x] = 0.0  # Mark as visited

    def compute_score(self, force_update=False):
        """
        Computes the exploration score as the percentage of reachable pixels explored.
        Optimized to avoid unnecessary data transfer and computation overhead.
        Args:
            force_update: Force recalculation of the exploration score.
        Returns:
            float: Exploration score.
        """
        if not force_update and self._cached_score is not None:
            return self._cached_score

        # Perform erosion in one step using the precomputed kernel
        eroded_map = torch.nn.functional.conv2d(
            self._map_explo_lines.unsqueeze(0).unsqueeze(0),
            self._precomputed_kernel,
            padding="same"
        ).squeeze()

        # Update the exploration zones incrementally
        self._map_explo_zones = 1.0 - eroded_map
        self._map_explo_zones[self.map_playground == 1] = 0.0  # Exclude walls

        # Efficient pixel counting using GPU
        explored_pixels = torch.count_nonzero(self._map_explo_zones > 0).float()
        total_reachable_pixels = torch.tensor(self._total_reachable_pixels, device=self.device)

        # Compute score directly on GPU
        self._cached_score = (explored_pixels / total_reachable_pixels).item() if total_reachable_pixels > 0 else 0.0
        return self._cached_score

    def _create_circular_kernel(self, radius):
        """
        Creates a circular kernel with a given radius for erosion.
        Args:
            radius: Radius of the circle.
        Returns:
            torch.Tensor: A 2D circular kernel.
        """
        y, x = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=self.device),
            torch.arange(-radius, radius + 1, device=self.device),
            indexing="ij"
        )
        kernel = (x ** 2 + y ** 2 <= radius ** 2).float()
        kernel /= kernel.sum()  # Normalize for convolution
        return kernel.unsqueeze(0).unsqueeze(0)

    def get_pretty_map_explo_lines(self):
        """
        Returns a visualization of the explored lines.
        Returns:
            numpy.ndarray: A grayscale image with walls, free space, and explored lines.
        """
        pretty_map = torch.zeros_like(self.map_playground, dtype=torch.float32)
        pretty_map[self.map_playground == 1] = 255  # Walls
        pretty_map[self._map_explo_lines == 0] = 128  # Explored lines
        return pretty_map.cpu().numpy()

    def get_pretty_map_explo_zones(self):
        """
        Returns a visualization of the explored zones.
        Returns:
            numpy.ndarray: A grayscale image with walls, free space, and explored zones.
        """
        pretty_map = torch.zeros_like(self.map_playground, dtype=torch.float32)
        pretty_map[self.map_playground == 1] = 255  # Walls
        pretty_map[self._map_explo_zones > 0] = 128  # Explored zones
        return pretty_map.cpu().numpy()


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


# ------------------------
# COARSER LIDAR PREPROCESS
# ------------------------
def preprocess_lidar(lidar_values):
    """
    Instead of 180 or 45 aggregated rays, let's reduce the LiDAR to e.g. 15 sectors.
    This drastically reduces how precisely the drone perceives its surroundings.
    """
    if lidar_values is None or len(lidar_values) == 0:
        return np.zeros(15)  # Default to zeros if no data

    num_sectors = 15  # Decrease from 45 to 15
    # Round down for the sector size
    sector_size = max(1, len(lidar_values) // num_sectors)

    aggregated = []
    for i in range(num_sectors):
        start_idx = i * sector_size
        end_idx = (i + 1) * sector_size
        # Safely slice within array
        segment = lidar_values[start_idx:end_idx]
        aggregated.append(np.mean(segment))

    # Fallback if leftover rays exist in the end (only if len(lidar_values) % num_sectors != 0)
    if len(aggregated) < num_sectors:
        aggregated += [300] * (num_sectors - len(aggregated))

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
    log_probs_continuous = -0.5 * (
        ((sampled_continuous_actions - means) / (stds + 1e-8)) ** 2
        + 2 * log_stds
        + math.log(2 * math.pi)
    )
    log_probs_continuous = log_probs_continuous.sum(dim=1)

    # Compute discrete action probability
    discrete_action_prob = torch.sigmoid(discrete_logits).squeeze()
    discrete_action = int(discrete_action_prob.item() > 0.5)

    # Log probability for discrete action
    log_prob_discrete = (
        torch.log(discrete_action_prob + 1e-8)
        if discrete_action == 1
        else torch.log(1 - discrete_action_prob + 1e-8)
    )

    # Combine actions
    action = torch.cat(
        [continuous_actions, torch.tensor([[discrete_action]], device=device)], dim=1
    ).squeeze()
    log_prob = log_probs_continuous + log_prob_discrete
    return action.detach().cpu().numpy(), log_prob


def compute_reward(drone, is_collision, found, explored_map, last_explo, pose, semantic_data, time_penalty):
    reward = 0

    # Penalize collisions heavily
    if is_collision:
        reward -= 100

    # Exploration reward
    exploration_score = explored_map.compute_score(force_update=True) - last_explo
    reward += 10 * exploration_score  # Higher weight for exploration

    # Penalize revisiting the same area
    # NOTE: We do it coarsely as well if needed,
    # but here let's just keep it simple:
    grid_x, grid_y = int(pose.position[0]), int(pose.position[1])
    if (0 <= grid_x < explored_map._orig_width) and (0 <= grid_y < explored_map._orig_height):
        pass  # or any additional checks if you need them

    # Reward progress towards the goal (e.g., finding wounded)
    nearest_wounded = semantic_data[0] * 300
    if nearest_wounded < 300:
        progress_toward_wounded = max(0, 300 - nearest_wounded) / 300.0
        reward += 100 * progress_toward_wounded

    # Reward for completing the objective
    if found:
        reward += 500

    # Penalize idleness
    reward -= time_penalty

    return reward, exploration_score


policy_net = PolicyNetwork(state_dim=19, action_dim=4)
value_net = ValueNetwork(state_dim=19)
optimizer_policy = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
optimizer_value = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)


def compute_gae(rewards, values, next_values, dones):
    advantages = []
    gae = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * next_values[step] * (1 - dones[step]) - values[step]
        gae = delta + GAMMA * LAM * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def optimize_ppo(states, actions, log_probs, returns, advantages, batch_size=1024):
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
            dtype=torch.float32
        ).to(device)
        returns_batch = torch.FloatTensor(np.array(returns)[batch_indices]).to(device)
        advantages_batch = torch.FloatTensor(np.array(advantages)[batch_indices]).to(device)

        # Forward pass through networks
        means, log_stds, discrete_logits = policy_net(states_batch)
        values = value_net(states_batch).squeeze()

        # Compute new log probabilities
        stds = torch.exp(log_stds)
        new_log_probs_continuous = -0.5 * (
            ((actions_batch[:, :3] - means) / stds) ** 2
            + 2 * log_stds
            + math.log(2 * math.pi)
        )
        new_log_probs_continuous = new_log_probs_continuous.sum(dim=1)
        discrete_action_prob = torch.sigmoid(discrete_logits).squeeze()
        new_log_probs_discrete = (
            torch.log(discrete_action_prob + 1e-8) * actions_batch[:, 3]
            + torch.log(1 - discrete_action_prob + 1e-8) * (1 - actions_batch[:, 3])
        )

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


def compute_returns(rewards, gamma=0.99):
    """
    Compute the cumulative discounted rewards (returns) for each timestep.
    """
    returns = []
    g = 0
    for reward in reversed(rewards):
        g = reward + gamma * g
        returns.insert(0, g)
    return returns


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
        self.pose = Pose()
        self.init_pose = Pose()

    def control(self):
        lidar_data = preprocess_lidar(self.lidar_values())
        semantic_data = preprocess_semantic(self.semantic_values())
        # state_dim = 19 => 15 LiDAR + 4 semantic
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


def train():
    map_training = MyMapIntermediate01()
    playground = map_training.construct_playground(drone_type=MyDrone)
    explored_map = ExploredMapOptimized(playground=playground, device_available=device)
    rewards_per_episode = []

    batch_size = 1024  # Mini-batch size for optimization

    for episode in range(NB_EPISODES):
        playground.reset()
        explored_map.reset()
        for drone in map_training.drones:
            drone.init_pose.position = drone.measured_gps_position()

        batch_states, batch_actions, batch_log_probs, batch_rewards = [], [], [], []
        total_reward, step, explo_score = 0, 0, 0
        done = False

        while not done and step < MAX_STEPS and all([drone.drone_health > 0 for drone in map_training.drones]):
            step += 1
            actions_drones = {}
            for drone in map_training.drones:
                lidar_data = preprocess_lidar(drone.lidar_values())
                semantic_data = preprocess_semantic(drone.semantic_values())
                state = np.concatenate([lidar_data, semantic_data])
                action, log_prob = select_action(drone.policy_net, state)

                actions_drones[drone] = process_actions(action)
                explored_map.update_drones([drone])

                # Compute reward with exploration score update
                if step % 2 == 0 or done:
                    reward, explo_score = compute_reward(
                        drone=drone,
                        is_collision=min(drone.lidar_values()) < 15,
                        found=(semantic_data[0] * 300 < 80),
                        explored_map=explored_map,
                        last_explo=explo_score,
                        pose=drone.pose,
                        semantic_data=semantic_data,
                        time_penalty=1
                    )
                else:
                    reward = -1  # Minor penalty for time

                batch_states.append(state)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                batch_rewards.append(reward)
                total_reward += reward

                done = (semantic_data[0] * 300 < 80)
                if done:
                    print("Found wounded!")

            playground.step(actions_drones)
            for drone in map_training.drones:
                drone.update_pose()  # Update drone pose after the step

            # Incrementally optimize networks when enough data is collected
            if len(batch_rewards) >= batch_size or done or step == MAX_STEPS:
                batch_returns = compute_returns(batch_rewards)
                batch_advantages = [ret - rew for ret, rew in zip(batch_returns, batch_rewards)]
                optimize_ppo(batch_states, batch_actions, batch_log_probs, batch_returns, batch_advantages)

                # Clear batch data after optimization
                batch_states, batch_actions, batch_log_probs, batch_rewards = [], [], [], []

        rewards_per_episode.append(total_reward)

        # Rebuild scenario if the drone is destroyed or we ended the episode
        if any([drone.drone_health <= 0 for drone in map_training.drones]):
            map_training = MyMapIntermediate01()
            playground = map_training.construct_playground(drone_type=MyDrone)

        # Logging
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward}")
            torch.save(policy_net.state_dict(), 'policy_net_ppo.pth')
            torch.save(value_net.state_dict(), 'value_net_ppo.pth')

        # Early stopping if it’s “solved” for your criteria
        if np.mean(rewards_per_episode[-10:]) > 10000:
            print(f"Solved in {episode} episodes!")
            break


if __name__ == "__main__":
    train()
