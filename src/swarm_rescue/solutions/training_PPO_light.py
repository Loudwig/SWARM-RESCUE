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
MAX_STEPS = 300
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Policy and Value Networks
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim=4):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim + 3)
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



# Reward Function

def compute_reward(drone, is_collision, pose, lidar_data, last_position, time_penalty):
    reward = 0

    # Penalize collisions
    if is_collision:
        return -100, pose.position  # Immediate heavy penalty

    # Reward movement away from previous position
    dist_moved = np.linalg.norm(np.array(pose.position) - np.array(last_position))
    reward += 5 * dist_moved  # Encourage movement

    # Reward for detecting unexplored areas (using LiDAR data)
    #unexplored_lidar = np.sum(lidar_data < 0.8)  # Count LiDAR hits close to walls/unexplored zones
    #reward += 10 * unexplored_lidar

    # Penalize idleness
    reward -= time_penalty

    return reward, pose.position

# PPO Functions
def compute_returns(rewards, gamma=0.99):
    returns = []
    g = 0
    for reward in reversed(rewards):
        g = reward + gamma * g
        returns.insert(0, g)
    return returns

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
            dtype=torch.float32).to(device)
        returns_batch = torch.FloatTensor(np.array(returns)[batch_indices]).to(device)
        advantages_batch = torch.FloatTensor(np.array(advantages)[batch_indices]).to(device)

        means, log_stds, discrete_logits = policy_net(states_batch)
        values = value_net(states_batch).squeeze()

        stds = torch.exp(log_stds)
        new_log_probs_continuous = -0.5 * (((actions_batch[:, :3] - means) / stds) ** 2 + 2 * log_stds + math.log(2 * math.pi))
        new_log_probs_continuous = new_log_probs_continuous.sum(dim=1)
        discrete_action_prob = torch.sigmoid(discrete_logits).squeeze()
        new_log_probs_discrete = torch.log(discrete_action_prob + 1e-8) * actions_batch[:, 3] + \
                                 torch.log(1 - discrete_action_prob + 1e-8) * (1 - actions_batch[:, 3])

        new_log_probs = new_log_probs_continuous + new_log_probs_discrete
        ratios = torch.exp(new_log_probs - log_probs_batch)

        surr1 = ratios * advantages_batch
        surr2 = torch.clamp(ratios, 1 - EPSILON_CLIP, 1 + EPSILON_CLIP) * advantages_batch
        policy_loss = -torch.min(surr1, surr2).mean()

        entropy = -0.5 * (1 + 2 * log_stds + math.log(2 * math.pi)).mean()
        policy_loss -= ENTROPY_BETA * entropy

        value_loss = nn.functional.mse_loss(values, returns_batch)

        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

# Training
policy_net = PolicyNetwork(state_dim=49, action_dim=4)
value_net = ValueNetwork(state_dim=49)
optimizer_policy = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
optimizer_value = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)

def train():
    map_training = MyMapIntermediate01()
    playground = map_training.construct_playground(drone_type=MyDrone)
    rewards_per_episode = []
    batch_size = 1024  # Larger batch size for PPO stability
    last_positions = {}

    for episode in range(NB_EPISODES):
        playground.reset()
        for drone in map_training.drones:
            drone.init_pose.position = drone.measured_gps_position()
        last_positions = {drone: drone.init_pose.position for drone in map_training.drones}
        batch_states, batch_actions, batch_log_probs, batch_rewards = [], [], [], []
        total_reward, step, done = 0, 0, False

        while not done and step < MAX_STEPS and all([drone.drone_health > 0 for drone in map_training.drones]):
            step += 1
            actions_drones = {}
            for drone in map_training.drones:
                lidar_data = preprocess_lidar(drone.lidar_values())
                semantic_data = preprocess_semantic(drone.semantic_values())
                state = np.concatenate([lidar_data, semantic_data])
                action, log_prob = select_action(drone.policy_net, state)

                actions_drones[drone] = process_actions(action)
                reward, last_positions[drone] = compute_reward(
                    drone=drone,
                    is_collision=min(drone.lidar_values()) < 15,
                    pose=drone.pose,
                    lidar_data=lidar_data,
                    last_position=last_positions[drone],
                    time_penalty=1
                )

                batch_states.append(state)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                batch_rewards.append(reward)
                total_reward += reward

            playground.step(actions_drones)
            for drone in map_training.drones:
                drone.update_pose()

            if len(batch_rewards) >= batch_size or done or step == MAX_STEPS:
                rewards_normalized = (batch_rewards - np.mean(batch_rewards)) / (np.std(batch_rewards) + 1e-8)
                batch_returns = compute_returns(rewards_normalized)
                batch_advantages = batch_returns - np.mean(batch_returns)
                optimize_ppo(batch_states, batch_actions, batch_log_probs, batch_returns, batch_advantages)

                batch_states, batch_actions, batch_log_probs, batch_rewards = [], [], [], []

        rewards_per_episode.append(total_reward)

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {total_reward}")
            torch.save(policy_net.state_dict(), 'policy_net_ppo_simple.pth')
            torch.save(value_net.state_dict(), 'value_net_ppo_simple.pth')

        if np.mean(rewards_per_episode[-10:]) > 10000:
            print(f"Solved in {episode} episodes!")
            break

if __name__ == "__main__":
    train()
