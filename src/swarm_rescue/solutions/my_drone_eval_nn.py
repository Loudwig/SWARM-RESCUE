import numpy as np
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from enum import Enum



# Insert the parent directory of the current file's directory into sys.path.
# This allows Python to locate modules that are one level above the current
# script, in this case spg_overlay.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(os.path.dirname(__file__), 'policy_net.pth')


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



class MyDroneEval(DroneAbstract):
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
        self.policy_net = PolicyNetwork(state_dim=22, action_dim=4)  # 18 Lidar + 2 Semantic
        self.policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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
