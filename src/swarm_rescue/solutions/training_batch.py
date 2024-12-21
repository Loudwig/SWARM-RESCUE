import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import sys
from pathlib import Path
from enum import Enum
from typing import List, Type



# Insert the parent directory of the current file's directory into sys.path.
# This allows Python to locate modules that are one level above the current
# script, in this case spg_overlay.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maps.map_intermediate_01 import MyMapIntermediate01

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.utils.misc_data import MiscData


from spg_overlay.entities.rescue_center import RescueCenter
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.map_abstract import MapAbstract

class MyMapTraining(MapAbstract):
    def __init__(self):
        super().__init__()

        self._size_area = (800, 800)
        self._rescue_center = RescueCenter(size=(100, 100))
        self._rescue_center_pos = ((0, 150), 0)

        # Wounded persons setup
        self._number_wounded_persons = 2
        self._wounded_persons_pos = []
        self._wounded_persons = []

        for i in range(self._number_wounded_persons):
            x = random.uniform(-150, 150)
            y = random.uniform(-150, 150)
            pos = ((x, y), random.uniform(-math.pi, math.pi))
            self._wounded_persons_pos.append(pos)

        # Drone positions
        self._number_drones = 1
        self._drones_pos = [((-100, -100), random.uniform(-math.pi, math.pi))]
        self._drones: List[DroneAbstract] = []

    def construct_playground(self, drone_type):
        # Create a new playground
        playground = ClosedPlayground(size=self._size_area)

        # Add a fresh instance of the rescue center
        self._rescue_center = RescueCenter(size=(100, 100))
        playground.add(self._rescue_center, self._rescue_center_pos)

        # Add wounded persons (reinitialize the list)
        self._wounded_persons = []
        for pos in self._wounded_persons_pos:
            wounded_person = WoundedPerson(rescue_center=self._rescue_center)
            self._wounded_persons.append(wounded_person)
            playground.add(wounded_person, pos)

        # Add drones (reinitialize the list)
        self._drones = []
        misc_data = MiscData(size_area=self._size_area,
                             number_drones=self._number_drones,
                             max_timestep_limit=self._max_timestep_limit,
                             max_walltime_limit=self._max_walltime_limit)
        for i, pos in enumerate(self._drones_pos):
            drone = drone_type(identifier=i, misc_data=misc_data)
            self._drones.append(drone)
            playground.add(drone, pos)

        return playground

    def get_drones(self):
        return self._drones

    def get_number_drones(self):
        return self._number_drones


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
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
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


def compute_reward(is_collision, is_rescue, near_wounded, lidar_data, current_angle, time_penalty):
    """
    Compute reward based on multiple factors:
    1. Avoid collisions.
    2. Encourage reaching the rescue area.
    3. Incentivize moving toward the largest LiDAR opening.
    4. Penalize time spent idling.
    """
    reward = 0

    # Penalize collisions
    if is_collision:
        reward -= 50

    # Reward for successfully rescuing and reaching the rescue area
    if is_rescue and near_wounded:
        reward += 50

    # Reward movement toward the largest opening
    largest_opening_angle, _ = find_largest_opening(lidar_data)
    angular_difference = abs(largest_opening_angle - current_angle)
    angular_difference = min(angular_difference, 2 * np.pi - angular_difference)  # Ensure smallest angular difference
    reward += 5 * (1 - angular_difference / np.pi)  # Scale reward: closer to the opening direction gets higher reward

    # Penalize idling or spending too much time
    reward -= time_penalty

    return reward



def compute_returns(rewards,returned):
    returns = [returned+rewards[-1]]
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
        # gui = GuiSR(playground=playground, the_map=map_training, draw_semantic_rays=True)
        done = False
        states, actions, rewards = [], [], []
        total_reward = 0
        step = 0
        nb_rescue = 0

        while step < MAX_STEPS:
            step += 1
            playground.step()
            # gui.run()  # Run GUI for visualization

            for drone in map_training.drones:
                lidar_data = preprocess_lidar(drone.lidar_values())
                semantic_data = preprocess_semantic(drone.semantic_values())
                state = np.concatenate([lidar_data, semantic_data])
                action, _ = select_action(drone.policy_net, state)
                current_angle = drone.measured_compass_angle()
                reward = compute_reward(
                    is_collision=min(drone.lidar_values()) < 40,
                    is_rescue=action[3],
                    near_wounded=lidar_data[0]*300<40,
                    lidar_data=lidar_data,
                    current_angle=current_angle,
                    time_penalty=0.1
                )
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                total_reward += reward
                nb_rescue += int(drone.is_inside_return_area)

            done = nb_rescue == map_training.number_drones

        # Optimize the policy and value networks in batches
        returns = compute_returns(rewards, map_training.compute_score_health_returned())
        optimize_batch(states, actions, returns, batch_size=64)
        rewards_per_episode.append(total_reward)

        if episode % 10 == 1:
            print(f"Episode {episode}, Reward: {total_reward}, Mean Last 100 Rewards: {np.mean(rewards_per_episode[-100:])}")
            torch.save(policy_net.state_dict(), 'policy_net.pth')
            torch.save(value_net.state_dict(), 'value_net.pth')

        if np.mean(rewards_per_episode[-100:]) > 10000:
            print(f"Training solved in {episode} episodes!")
            break


if __name__ == "__main__":
    train()