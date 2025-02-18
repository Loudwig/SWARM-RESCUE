"""
Adapted PPO Implementation for a Multiagent Frontier Drone
-----------------------------------------------------------

This file adapts your PPO code to the frontier-based drone specification.
The RL policy is used during exploration. The drone class MyDronePPO inherits from
the frontier-based drone (MyDroneFrontex) and uses PPO for the EXPLORING_RL state.
"""

import math
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from enum import Enum, auto
from collections import deque
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# 1. PPO Networks & Helper Functions
# =============================================================================

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim=4):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            # Output: first 3 for continuous means, next 3 for log std, last 1 for discrete logits
            nn.Linear(128, action_dim + 3 + 1)
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


def preprocess_lidar(lidar_values):
    """
    Aggregates the LiDAR sensor values into 15 sectors, normalized between 0 and 1.
    """
    if lidar_values is None or len(lidar_values) == 0:
        return np.zeros(15)  # default if no data

    num_sectors = 15
    sector_size = max(1, len(lidar_values) // num_sectors)
    aggregated = []
    for i in range(num_sectors):
        start_idx = i * sector_size
        end_idx = (i + 1) * sector_size
        segment = lidar_values[start_idx:end_idx]
        aggregated.append(np.mean(segment))
    if len(aggregated) < num_sectors:
        aggregated += [300] * (num_sectors - len(aggregated))
    aggregated = np.nan_to_num(aggregated, nan=300, posinf=300, neginf=0)
    aggregated = np.clip(aggregated, 0, 300) / 300.0
    return aggregated


def preprocess_semantic(semantic_values):
    """
    Process semantic sensor data into 4 features:
      - normalized nearest wounded distance and angle,
      - normalized rescue center distance and angle.
    """
    wounded_detected = []
    rescue_center_detected = []
    if semantic_values:
        for data in semantic_values:
            if data.entity_type == getattr(
                    __import__("spg_overlay.entities.drone_distance_sensors", fromlist=["DroneSemanticSensor"]),
                    "DroneSemanticSensor").TypeEntity.WOUNDED_PERSON and not data.grasped:
                wounded_detected.append((data.distance, data.angle))
            elif data.entity_type == getattr(
                    __import__("spg_overlay.entities.drone_distance_sensors", fromlist=["DroneSemanticSensor"]),
                    "DroneSemanticSensor").TypeEntity.RESCUE_CENTER:
                rescue_center_detected.append((data.distance, data.angle))
    nearest_wounded = min(wounded_detected, default=(300, 0), key=lambda x: x[0])
    nearest_rescue_center = min(rescue_center_detected, default=(300, 0), key=lambda x: x[0])
    processed = [
        nearest_wounded[0] / 300.0,  # distance normalized
        nearest_wounded[1] / np.pi,  # angle normalized (approx. [-1,1])
        nearest_rescue_center[0] / 300.0,
        nearest_rescue_center[1] / np.pi
    ]
    return processed


def select_action(policy, state):
    state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
    means, log_stds, discrete_logits = policy(state_tensor)
    stds = torch.exp(log_stds)
    sampled_continuous = means + torch.randn_like(means) * stds
    continuous_actions = torch.clamp(sampled_continuous, -1.0, 1.0)
    # Log-probabilities for continuous part
    log_probs_cont = -0.5 * (((sampled_continuous - means) / (stds + 1e-8)) ** 2 +
                             2 * log_stds + math.log(2 * math.pi))
    log_probs_cont = log_probs_cont.sum(dim=1)
    # Discrete action (e.g., for grasper)
    discrete_prob = torch.sigmoid(discrete_logits).squeeze()
    discrete_action = 1 if discrete_prob.item() > 0.5 else 0
    log_prob_disc = torch.log(discrete_prob + 1e-8) if discrete_action == 1 else torch.log(1 - discrete_prob + 1e-8)
    total_log_prob = log_probs_cont + log_prob_disc
    # Combine actions: [forward, lateral, rotation, grasper]
    action = torch.cat([continuous_actions, torch.tensor([[discrete_action]], device=device)], dim=1)
    return action.detach().cpu().numpy().squeeze(), total_log_prob


def process_actions(action_array):
    """
    Convert the action vector into a command dictionary.
    The vector format is assumed to be:
      [forward, lateral, rotation, grasper]
    """
    return {
        "forward": float(action_array[0]),
        "lateral": float(action_array[1]),
        "rotation": float(action_array[2]),
        "grasper": int(action_array[3])
    }


# PPO hyperparameters
GAMMA = 0.99
LEARNING_RATE = 1e-4
ENTROPY_BETA = 0.05
EPSILON_CLIP = 0.2
LAM = 0.95
NB_EPISODES = 1000
MAX_STEPS = 800


# Functions to compute returns and advantages

def compute_returns(rewards, gamma=GAMMA):
    returns = []
    g = 0
    for reward in reversed(rewards):
        g = reward + gamma * g
        returns.insert(0, g)
    return returns


def compute_gae(rewards, values, next_values, dones):
    advantages = []
    gae = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * next_values[step] * (1 - dones[step]) - values[step]
        gae = delta + GAMMA * LAM * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def optimize_ppo(states, actions, log_probs, returns, advantages, policy_net, value_net,
                 optimizer_policy, optimizer_value, batch_size=1024):
    dataset_size = len(states)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    for start_idx in range(0, dataset_size, batch_size):
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]
        states_batch = torch.tensor(np.array(states)[batch_indices], dtype=torch.float32).to(device)
        actions_batch = torch.tensor(np.array(actions)[batch_indices], dtype=torch.float32).to(device)
        log_probs_batch = torch.tensor(np.array([lp.detach().cpu().numpy() for lp in log_probs])[batch_indices],
                                       dtype=torch.float32).to(device)
        returns_batch = torch.tensor(np.array(returns)[batch_indices], dtype=torch.float32).to(device)
        advantages_batch = torch.tensor(np.array(advantages)[batch_indices], dtype=torch.float32).to(device)

        # Forward pass
        means, log_stds, discrete_logits = policy_net(states_batch)
        values_pred = value_net(states_batch).squeeze()
        stds = torch.exp(log_stds)
        # Continuous log probs
        new_log_probs_cont = -0.5 * (((actions_batch[:, :3] - means) / (stds + 1e-8)) ** 2 +
                                     2 * log_stds + math.log(2 * math.pi))
        new_log_probs_cont = new_log_probs_cont.sum(dim=1)
        # Discrete log probs
        discrete_prob = torch.sigmoid(discrete_logits).squeeze()
        new_log_probs_disc = torch.log(discrete_prob + 1e-8) * actions_batch[:, 3] + \
                             torch.log(1 - discrete_prob + 1e-8) * (1 - actions_batch[:, 3])
        new_log_probs = new_log_probs_cont + new_log_probs_disc
        ratios = torch.exp(new_log_probs - log_probs_batch)
        surr1 = ratios * advantages_batch
        surr2 = torch.clamp(ratios, 1 - EPSILON_CLIP, 1 + EPSILON_CLIP) * advantages_batch
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy = -0.5 * (1 + 2 * log_stds + math.log(2 * math.pi)).mean()
        policy_loss -= ENTROPY_BETA * entropy
        value_loss = nn.functional.mse_loss(values_pred, returns_batch)

        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()


# =============================================================================
# 2. Adapted Drone Class (Using PPO for Exploration)
# =============================================================================

# Import your frontier code base – make sure the import path is correct.
from spg_overlay.entities.drone_abstract import DroneAbstract
from solutions.utils.pose import Pose
from solutions.utils.grids import OccupancyGrid


# (Other imports from your frontier code as needed)

class MyDroneFrontex(DroneAbstract):
    """
    (Your original frontier-based drone code.)
    For brevity, only the essential structure is kept here.
    """

    class State(Enum):
        WAITING = auto()
        SEARCHING_WALL = auto()
        FOLLOWING_WALL = auto()
        EXPLORING_FRONTIERS = auto()
        # We add a new state for RL-based exploration:
        EXPLORING_RL = auto()
        GRASPING_WOUNDED = auto()
        SEARCHING_RESCUE_CENTER = auto()
        GOING_RESCUE_CENTER = auto()

    def __init__(self, identifier=None, misc_data=None, **kwargs):
        super().__init__(identifier=identifier, misc_data=misc_data, **kwargs)
        # Mapping and navigation initialization (as in your frontier code)
        self.mapping_params = type("MappingParams", (), {"resolution": 5, "display_map": False})
        self.estimated_pose = Pose()
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=self.mapping_params.resolution,
                                  lidar=self.lidar())
        self.previous_position = deque(maxlen=1)
        self.previous_position.append((0, 0))
        self.previous_orientation = deque(maxlen=1)
        self.previous_orientation.append(0)
        self.state = self.State.WAITING
        self.previous_state = self.State.WAITING
        self.waiting_params = type("WaitingStateParams", (), {"step_waiting": 50})
        self.step_waiting_count = 0
        # (Other initializations as in your original code)
        self.timestep_count = 0

    def mapping(self, display=False):
        # (Mapping update as in your original code)
        if self.timestep_count == 1:
            start_x, start_y = self.measured_gps_position()
            self.grid.set_initial_cell(start_x, start_y)
        self.estimated_pose = Pose(np.array(self.measured_gps_position()),
                                   self.measured_compass_angle(),
                                   self.odometer_values(),
                                   self.previous_position[-1],
                                   self.previous_orientation[-1],
                                   self.size_area)
        self.previous_position.append(self.estimated_pose.position)
        self.previous_orientation.append(self.estimated_pose.orientation)
        grid_update_informations = self.grid.to_update(pose=self.estimated_pose)
        if self.communicator:
            for msg in self.communicator.received_messages:
                grid_update_informations += msg[1]
        self.grid.update(grid_update_informations)
        if display and (self.timestep_count % 5 == 0):
            self.grid.display(self.grid.zoomed_grid, self.estimated_pose,
                              title=f"Drone {self.identifier} zoomed occupancy grid")

    def process_lidar_sensor(self, lidar):
        lidar_values = lidar.get_sensor_values()
        if lidar_values is None:
            return (False, 0, 0)
        ray_angles = lidar.ray_angles
        min_dist = min(lidar_values)
        angle_nearest = ray_angles[np.argmin(lidar_values)]
        near_obstacle = (min_dist < 30)  # example threshold
        epsilon_wall_angle = angle_nearest - np.pi / 2
        return near_obstacle, epsilon_wall_angle, min_dist

    def process_semantic_sensor(self):
        semantic_values = self.semantic_values()
        # (Simplified processing – return values needed for state update)
        found_wounded = any(getattr(data, "entity_type", None) ==
                            getattr(__import__("spg_overlay.entities.drone_distance_sensors",
                                               fromlist=["DroneSemanticSensor"]),
                                    "DroneSemanticSensor").TypeEntity.WOUNDED_PERSON
                            for data in semantic_values)
        found_rescue_center = any(getattr(data, "entity_type", None) ==
                                  getattr(__import__("spg_overlay.entities.drone_distance_sensors",
                                                     fromlist=["DroneSemanticSensor"]),
                                          "DroneSemanticSensor").TypeEntity.RESCUE_CENTER
                                  for data in semantic_values)
        # For simplicity, set angles to zero:
        return found_wounded, found_rescue_center, 0, 0, False

    def state_update(self, found_wall, found_wounded, found_rescue_center):
        # A simple state transition: after waiting, switch to RL-based exploration.
        self.previous_state = self.state
        if self.state == self.State.WAITING and self.step_waiting_count >= self.waiting_params.step_waiting:
            self.state = self.State.EXPLORING_RL
        # (Other transitions as in your original code)
        if self.state != self.previous_state and self.state == self.State.WAITING:
            self.step_waiting_count = 0

    def control(self):
        self.timestep_count += 1
        self.mapping(display=self.mapping_params.display_map)
        found_wall, epsilon_wall_angle, min_dist = self.process_lidar_sensor(self.lidar())
        found_wounded, found_rescue_center, _, _, _ = self.process_semantic_sensor()
        self.state_update(found_wall, found_wounded, found_rescue_center)
        # Use RL for exploration if in the EXPLORING_RL state.
        if self.state == self.State.EXPLORING_RL:
            return self.handle_exploring_rl()
        # Otherwise, fall back on a simple waiting command.
        return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

    # ----------
    # This method will be overridden in MyDronePPO.
    def handle_exploring_rl(self):
        raise NotImplementedError("This drone should use the PPO exploration handler.")


# Now we create a new drone class that uses PPO for exploration.
class MyDronePPO(MyDroneFrontex):
    def __init__(self, identifier=None, misc_data=None, **kwargs):
        super().__init__(identifier=identifier, misc_data=misc_data, **kwargs)
        # Initialize PPO networks
        self.state_dim = 19  # 15 LiDAR + 4 semantic features
        self.action_dim = 4  # forward, lateral, rotation, grasper
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim)
        self.value_net = ValueNetwork(self.state_dim)
        # (Optionally, load pretrained weights here)

    def get_rl_state(self):
        # Build the state vector from sensor data using the PPO preprocessing functions.
        lidar_data = preprocess_lidar(self.lidar().get_sensor_values())
        semantic_data = preprocess_semantic(self.semantic_values())
        state = np.concatenate([lidar_data, semantic_data])
        return state

    def handle_exploring_rl(self):
        state = self.get_rl_state()
        action, _ = select_action(self.policy_net, state)
        return process_actions(action)


# =============================================================================
# 3. Training Loop
# =============================================================================

# Import your simulation environment. For example, using your MyMapIntermediate01:
from maps.map_intermediate_01 import MyMapIntermediate01


def compute_reward(drone, is_collision, found, last_explo, time_penalty):
    reward = 0
    if is_collision:
        reward -= 100
    # Here, we simply reward exploration progress; you can compute an exploration score if available.
    reward += 1  # reward per step
    if found:
        reward += 500
    reward -= time_penalty
    return reward


def train():
    # Instantiate the simulation environment with our PPO drone.
    map_training = MyMapIntermediate01()
    playground = map_training.construct_playground(drone_type=MyDronePPO)

    # Optimizers for PPO
    optimizer_policy = optim.Adam(playground.drones[0].policy_net.parameters(), lr=LEARNING_RATE)
    optimizer_value = optim.Adam(playground.drones[0].value_net.parameters(), lr=LEARNING_RATE)

    rewards_per_episode = []

    for episode in range(NB_EPISODES):
        playground.reset()  # Reset simulation
        # Reset any exploration map if needed.
        batch_states, batch_actions, batch_log_probs, batch_rewards = [], [], [], []
        total_reward = 0
        step = 0
        done = False

        # (You might also want to reset drone poses, etc.)
        while not done and step < MAX_STEPS and all(drone.drone_health > 0 for drone in playground.drones):
            step += 1
            actions_drones = {}
            for drone in playground.drones:
                # Get RL state from the drone (using our PPO-based method)
                state = drone.get_rl_state()
                action, log_prob = select_action(drone.policy_net, state)
                actions_drones[drone] = process_actions(action)
                # Here we assume collision if a LiDAR reading is very low:
                is_collision = min(drone.lidar().get_sensor_values()) < 15
                # For reward, we can check if a wounded person is “found” (example threshold)
                semantic = preprocess_semantic(drone.semantic_values())
                found = semantic[0] * 300 < 80  # if wounded is near
                # A simple time penalty:
                reward = compute_reward(drone, is_collision, found, 0, time_penalty=1)
                batch_states.append(state)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                batch_rewards.append(reward)
                total_reward += reward
                # End episode if the drone finds a wounded
                if found:
                    done = True

            # Step the simulation with the chosen actions
            playground.step(actions_drones)
            for drone in playground.drones:
                # Update the drone pose if needed
                if hasattr(drone, "update_pose"):
                    drone.update_pose()

            # Optionally, perform PPO optimization when enough samples have been collected.
            if len(batch_rewards) >= 1024 or done or step == MAX_STEPS:
                values = []
                next_values = []
                dones = []
                for state in batch_states:
                    state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
                    values.append(playground.drones[0].value_net(state_tensor).item())
                    next_values.append(playground.drones[0].value_net(state_tensor).item())
                    dones.append(0)  # You can update this based on terminal conditions
                advantages, returns = compute_gae(batch_rewards, values, next_values, dones)
                optimize_ppo(batch_states, batch_actions, batch_log_probs, returns, advantages,
                             playground.drones[0].policy_net, playground.drones[0].value_net,
                             optimizer_policy, optimizer_value)
                batch_states, batch_actions, batch_log_probs, batch_rewards = [], [], [], []

        rewards_per_episode.append(total_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
            # Save models if desired:
            torch.save(playground.drones[0].policy_net.state_dict(), 'policy_net_ppo.pth')
            torch.save(playground.drones[0].value_net.state_dict(), 'value_net_ppo.pth')
        if np.mean(rewards_per_episode[-10:]) > 10000:
            print(f"Solved in {episode} episodes!")
            break


if __name__ == "__main__":
    train()
