import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

# ----------------
# Hyperparameters
# ----------------
GAMMA = 0.99
LEARNING_RATE = 1e-4
ENTROPY_BETA = 0.05
EPSILON_CLIP = 0.2
LAM = 0.95
NB_EPISODES = 1000
MAX_STEPS = 800
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# PPO networks
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim=4):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim + 3)  # (3 continuous means + 3 log_stds) + 1 discrete logit
        )
        self.to(device)

    def forward(self, x):
        """
        Returns:
          means: shape [batch, 3] for forward, lateral, rotation
          log_stds: shape [batch, 3]
          discrete_logits: shape [batch, 1]
        """
        logits = self.fc(x)  # shape: [batch, (3 + 3 + 1) = 7]
        means = torch.tanh(logits[:, :3])  # range [-1,1]
        log_stds = torch.clamp(logits[:, 3:6], -20, 2)  # clamp to avoid extreme stds
        discrete_logits = logits[:, 6:]  # shape [batch, 1]
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


# ----------------------------------------
# Preprocess: LiDAR (FULL 180 rays) + semantic
# ----------------------------------------
def preprocess_lidar(lidar_values):
    """
    Use ALL 180 LiDAR rays (assuming your environment provides exactly 180).
    Normalize each distance from [0..300] => [0..1].
    """
    if not any(lidar_values) or len(lidar_values) == 0:
        return np.zeros(181)

    # Clip to [0, 300], then normalize
    clipped = np.clip(lidar_values, 0, 300)
    normalized = clipped / 300.0
    return normalized


def preprocess_semantic(semantic_values):
    """
    We only need: nearest wounded distance & angle, nearest rescue center distance & angle.
    Each is normalized. If none found, default to (300, 0).
    """
    wounded_detected = []
    rescue_center_detected = []

    if semantic_values:
        for data in semantic_values:
            if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                wounded_detected.append((data.distance, data.angle))
            elif data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                rescue_center_detected.append((data.distance, data.angle))

    nearest_wounded = min(wounded_detected, default=(300, 0), key=lambda x: x[0])
    nearest_rescue = min(rescue_center_detected, default=(300, 0), key=lambda x: x[0])

    # [dist_wounded, angle_wounded, dist_rescue, angle_rescue]
    # Dist normalized to [0..1], angle normalized to [-1..1].
    processed = [
        nearest_wounded[0] / 300.0,
        nearest_wounded[1] / np.pi,  # angle in [-pi..pi], so dividing by pi => [-1..1]
        nearest_rescue[0] / 300.0,
        nearest_rescue[1] / np.pi
    ]
    return processed


# ------------------------
# Action (3 continuous + 1 discrete)
# ------------------------
def process_actions(action_tensor):
    """
    action_tensor = [forward, lateral, rotation, grasper_discrete]
    The continuous actions are in [-1..1], you can scale them if you want faster movement.
    """
    return {
        "forward": float(action_tensor[0]),  # continuous in [-1..1]
        "lateral": float(action_tensor[1]),  # continuous in [-1..1]
        "rotation": float(action_tensor[2]), # continuous in [-1..1]
        "grasper": float(action_tensor[3])   # 0 or 1 (discrete)
    }


def select_action(policy_net, state):
    """
    Sample from the policy's distribution. Returns (action, log_prob).
    """
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    means, log_stds, discrete_logits = policy_net(state_t)

    # Continuous part
    stds = torch.exp(log_stds)
    noise = torch.randn_like(means)
    sampled_continuous = means + noise * stds
    clamped_continuous = torch.clamp(sampled_continuous, -1.0, 1.0)

    # Log prob of continuous
    log_probs_continuous = -0.5 * (
        ((sampled_continuous - means) / (stds + 1e-8))**2
        + 2*log_stds
        + math.log(2*math.pi)
    )
    log_probs_continuous = log_probs_continuous.sum(dim=1)  # sum over 3 dims

    # Discrete part (grasper)
    discrete_prob = torch.sigmoid(discrete_logits).squeeze()
    discrete_action = (discrete_prob > 0.5).float()  # 0 or 1
    discrete_action = discrete_action.unsqueeze(-1)
    log_prob_discrete = torch.log(
        torch.where(discrete_action > 0.5, discrete_prob, 1.0 - discrete_prob) + 1e-8
    )

    # Combine
    log_prob = log_probs_continuous + log_prob_discrete
    action = torch.cat([clamped_continuous, discrete_action.unsqueeze(0)], dim=-1)
    return action.detach().cpu().numpy().flatten(), log_prob.squeeze()


# -----------------------------------
# Frontier + Distance Based Reward
# -----------------------------------
def compute_reward(
    drone,
    last_lidar,
    new_lidar,
    last_wounded_dist,
    new_wounded_dist,
    found_wounded,
    is_collision,
    time_penalty=0.1
):
    """
    Reward components:

    1) Time Penalty: -0.1 each step, to discourage stalling.
    2) Collision Penalty: -50 if collision (LiDAR min < threshold).
    3) Frontier Reward: Gains from discovering new free space in LiDAR.
       For each ray that sees further than last step, add a small reward.
    4) Distance Reward: If we get closer to the wounded, reward > 0.
       distance_delta = (last_wounded_dist - new_wounded_dist).
       scale it, e.g. * 10.
    5) Found Wounded Reward: +500 once we cross a threshold.

    Returns: (reward, done)
    """
    reward = 0.0

    # 1) Time penalty
    reward -= time_penalty

    # 2) Collision penalty
    if is_collision:
        reward -= 50

    # 3) Frontier reward
    #    sum of positive differences in LiDAR = newly seen free space
    if last_lidar is not None and new_lidar is not None:
        # frontier = Σ max(0, new_lidar[i] - last_lidar[i])
        differences = new_lidar - last_lidar
        frontier_gain = np.sum(np.clip(differences, 0, None))
        # scale frontier gain
        reward += 2.0 * frontier_gain

    # 4) Distance-based approach to wounded
    if (last_wounded_dist is not None) and (new_wounded_dist is not None):
        distance_delta = last_wounded_dist - new_wounded_dist
        reward += 10.0 * distance_delta

    # 5) Found wounded
    if found_wounded:
        reward += 500

    # If found wounded => done
    done = bool(found_wounded)
    return reward, done


# -------------------------
# PPO-related Functions
# -------------------------
def compute_returns(rewards, gamma=0.99):
    """
    Simple discounted returns for each step in an episode.
    """
    returns = []
    g = 0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.insert(0, g)
    return returns

def compute_gae(rewards, values, next_values, dones):
    """
    If you prefer GAE-lambda, implement here.
    For brevity, we might just do the simple compute_returns approach.
    """
    advantages = []
    gae = 0.0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * next_values[step] * (1 - dones[step]) - values[step]
        gae = delta + GAMMA * LAM * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [v + adv for v, adv in zip(values, advantages)]
    return advantages, returns


def optimize_ppo(
    states, actions, log_probs_old, returns, advantages,
    policy_net, value_net,
    optimizer_policy, optimizer_value,
    batch_size=1024
):
    """
    Standard PPO minibatch update
    """
    dataset_size = len(states)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    for start_idx in range(0, dataset_size, batch_size):
        end_idx = start_idx + batch_size
        batch_idx = indices[start_idx:end_idx]

        sb = torch.tensor(np.array(states)[batch_idx], dtype=torch.float32, device=device)
        ab = torch.tensor(np.array(actions)[batch_idx], dtype=torch.float32, device=device)
        old_logpb = torch.tensor(np.array([lp.item() for lp in log_probs_old])[batch_idx],
                                 dtype=torch.float32, device=device)
        rb = torch.tensor(np.array(returns)[batch_idx], dtype=torch.float32, device=device)
        advb = torch.tensor(np.array(advantages)[batch_idx], dtype=torch.float32, device=device)

        # Forward pass
        means, log_stds, disc_logits = policy_net(sb)
        vals = value_net(sb).squeeze()

        # Recompute log_probs
        stds = torch.exp(log_stds)
        # continuous
        continuous_acts = ab[:, :3]
        discrete_acts = ab[:, 3]  # 0 or 1

        # log_prob continuous
        z = (continuous_acts - means) / (stds + 1e-8)
        logp_cont = -0.5 * (z**2 + 2*log_stds + math.log(2*math.pi))
        logp_cont = logp_cont.sum(dim=1)

        # log_prob discrete
        disc_prob = torch.sigmoid(disc_logits).squeeze()
        logp_disc = torch.log(
            torch.where(discrete_acts > 0.5, disc_prob, 1 - disc_prob) + 1e-8
        )

        new_logp = logp_cont + logp_disc

        ratio = torch.exp(new_logp - old_logpb)

        # PPO clipped objective
        surr1 = ratio * advb
        surr2 = torch.clamp(ratio, 1 - EPSILON_CLIP, 1 + EPSILON_CLIP) * advb
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy for regularization (continuous only)
        # or combine with discrete, but keep it simple:
        entropy_cont = 0.5 * (log_stds.shape[1] * (1 + math.log(2*math.pi)) + 2*log_stds.sum(dim=1))
        entropy_loss = entropy_cont.mean()
        policy_loss -= ENTROPY_BETA * entropy_loss

        # Value loss
        value_loss = nn.functional.mse_loss(vals, rb)

        # Backprop
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()


# --------------------
# Drone Implementation
# --------------------
class MyDrone(DroneAbstract):
    class Activity(Enum):
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.policy_net = policy_net
        self.pose = Pose()
        self.init_pose = Pose()

        # We'll store last step's LiDAR & wounded distance
        self.last_lidar = None
        self.last_wounded_dist = None

    def control(self):
        # Grab sensor data
        raw_lidar = self.lidar_values()  # 180 rays
        if raw_lidar is None:
            raw_lidar = [300]*180  # fallback

        lidar_data = preprocess_lidar(raw_lidar)
        semantic_data = preprocess_semantic(self.semantic_values())

        # Build state
        state = np.concatenate([lidar_data, semantic_data])  # 180 + 4 = 184

        # PPO policy => action
        action_np, _ = select_action(self.policy_net, state)

        # Return dictionary to environment
        return process_actions(action_np)

    def update_pose(self):
        # Update the internal pose estimate
        pos = self.measured_gps_position()
        angle = self.measured_compass_angle()
        self.pose.position = pos
        self.pose.orientation = angle

    def define_message_for_all(self):
        pass


# -------------
# Instantiate
# -------------
policy_net = PolicyNetwork(state_dim=185, action_dim=4)
value_net = ValueNetwork(state_dim=185)
optimizer_policy = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
optimizer_value = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)


# -------------
# Training Loop
# -------------
def train():
    # Create environment
    map_training = MyMapIntermediate01()
    playground = map_training.construct_playground(drone_type=MyDrone)

    rewards_history = []

    # For PPO
    batch_states = []
    batch_actions = []
    batch_log_probs = []
    batch_rewards = []
    batch_dones = []
    batch_values = []
    batch_next_values = []

    batch_size = 2048  # try bigger batch if you have GPU memory
    global_step = 0

    for episode in range(NB_EPISODES):
        playground.reset()
        drone = map_training.drones[0]  # If you have only one drone
        drone.last_lidar = None
        drone.last_wounded_dist = None

        episode_reward = 0.0
        done = False
        step = 0

        while not done and step < MAX_STEPS and drone.drone_health > 0:
            step += 1
            global_step += 1

            # Collect observation from drone
            raw_lidar = drone.lidar_values() if any(drone.lidar_values()) else [300]*180
            new_lidar = preprocess_lidar(raw_lidar)

            semantic_data = preprocess_semantic(drone.semantic_values())
            wounded_dist = semantic_data[0] * 300  # up to 300
            found_wounded = (wounded_dist < 60)  # threshold

            # Collision check
            is_collision = (min(raw_lidar) < 15)

            # Build state and action
            state = np.concatenate([new_lidar, semantic_data])  # shape= [180 + 4=184]

            action_np, log_prob = select_action(policy_net, state)
            action_dict = process_actions(action_np)

            # Query value function (for advantage calc)
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                value_pred = value_net(state_t).item()

            # Step environment
            playground.step({drone: action_dict})
            drone.update_pose()

            # Compute reward
            reward, done_flag = compute_reward(
                drone=drone,
                last_lidar=drone.last_lidar,
                new_lidar=new_lidar,
                last_wounded_dist=drone.last_wounded_dist,
                new_wounded_dist=wounded_dist,
                found_wounded=found_wounded,
                is_collision=is_collision,
                time_penalty=0.1
            )

            episode_reward += reward

            # Save transition
            batch_states.append(state)
            batch_actions.append(action_np)
            batch_log_probs.append(log_prob)
            batch_rewards.append(reward)
            batch_dones.append(float(done_flag))
            batch_values.append(value_pred)

            # For next_value placeholder, we do it after the loop or next step
            # We'll fill batch_next_values after we get next state's value

            # Update drone's "last" sensors
            drone.last_lidar = new_lidar
            drone.last_wounded_dist = wounded_dist

            # If done
            if done_flag:
                done = True
                break

        # End of episode => compute next value as 0 if done
        next_value = 0.0
        if not done:  # not found wounded or not dead, but time step ended
            # get value from final state
            last_state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            next_value = value_net(last_state_t).item()

        # fill next_values for each step in episode
        for _ in range(step):
            batch_next_values.append(next_value)

        # Logging
        rewards_history.append(episode_reward)
        if (episode+1) % 10 == 0:
            avg_last_10 = np.mean(rewards_history[-10:])
            print(f"[Episode {episode+1}] Reward: {episode_reward:.2f}, Avg(10): {avg_last_10:.2f}")
            torch.save(policy_net.state_dict(), "policy_net_ppo_no_map.pth")
            torch.save(value_net.state_dict(), "value_net_ppo_no_map.pth")

        # PPO update once we have enough data
        if len(batch_states) >= batch_size:
            # We can do GAE or simpler returns
            # 1) Convert to arrays
            arr_rewards = np.array(batch_rewards, dtype=np.float32)
            arr_dones = np.array(batch_dones, dtype=np.float32)
            arr_values = np.array(batch_values, dtype=np.float32)
            arr_next_values = np.array(batch_next_values, dtype=np.float32)

            # 2) compute advantages
            advantages, returns_ = compute_gae(
                rewards=arr_rewards,
                values=arr_values,
                next_values=arr_next_values,
                dones=arr_dones
            )

            # 3) optimize
            optimize_ppo(
                states=batch_states,
                actions=batch_actions,
                log_probs_old=batch_log_probs,
                returns=returns_,
                advantages=advantages,
                policy_net=policy_net,
                value_net=value_net,
                optimizer_policy=optimizer_policy,
                optimizer_value=optimizer_value,
                batch_size=1024  # mini-minibatch size
            )

            # Clear batch
            batch_states.clear()
            batch_actions.clear()
            batch_log_probs.clear()
            batch_rewards.clear()
            batch_dones.clear()
            batch_values.clear()
            batch_next_values.clear()

        # Early stopping if drone is destroyed
        if drone.drone_health <= 0:
            print("Drone destroyed, resetting environment.")
            map_training = MyMapIntermediate01()
            playground = map_training.construct_playground(drone_type=MyDrone)

        # If you have your own success condition:
        if np.mean(rewards_history[-10:]) > 20000:  # example threshold
            print(f"Solved in {episode+1} episodes!")
            break

    print("Training finished!")


if __name__ == "__main__":
    train()
