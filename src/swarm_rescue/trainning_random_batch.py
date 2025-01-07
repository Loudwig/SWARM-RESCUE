from enum import Enum
from collections import deque
import math
from typing import Optional
import cv2
import numpy as np
import arcade
import torch
import sys
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from maps.map_intermediate_01 import MyMapIntermediate01 as M1
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.entities.rescue_center import RescueCenter
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.utils.utils import circular_mean, normalize_angle
from solutions.utils.pose import Pose
from solutions.utils.grid import OccupancyGrid
from solutions.utils.astar import *
from solutions.utils.NetworkPolicy import NetworkPolicy
from solutions.utils.NetworkValue import NetworkValue
from solutions.my_drone_A2C_trainning import MyDroneHulk

from torch.utils.data import Dataset, DataLoader

class DroneDataset(Dataset):
    def __init__(self, states_maps, states_vectors, actions, returns):
        # Convert lists to tensors
        self.states_maps = torch.stack([torch.FloatTensor(s).squeeze(0) for s in states_maps])
        self.states_vectors = torch.stack([torch.FloatTensor(s).squeeze(0) for s in states_vectors])
        self.actions = torch.stack([torch.FloatTensor(a) for a in actions])
        self.returns = torch.FloatTensor(returns)

    def __len__(self):
        return len(self.returns)

    def __getitem__(self, idx):
        return (self.states_maps[idx], 
                self.states_vectors[idx], 
                self.actions[idx], 
                self.returns[idx])

GAMMA = 0.99
LEARNING_RATE = 5e-6
ENTROPY_BETA = 0.1
NB_EPISODES = 1200
MAX_STEPS = 400

LossValue = []
LossPolicy = []
LossEntropy = []
LossOutbound = []
LossWeightsValue = []
LossExploration = []
LossWeightsPolicy = []



def compute_returns(rewards):
    returns = []
    g = 0
    for reward in reversed(rewards):
        g = reward + GAMMA * g
        returns.insert(0, g)
    return returns

def optimize_batch(states_map_batch, states_vector_batch, actions_batch, returns_batch, 
                  policy_net, value_net, optimizer_policy, optimizer_value, device):
    # Move tensors to device and ensure proper dimensions
    states_map_batch = states_map_batch.to(device)
    states_vector_batch = states_vector_batch.to(device)
    actions_batch = actions_batch.to(device)
    returns_batch = returns_batch.to(device)

    # Forward pass
    means, log_stds = policy_net(states_map_batch, states_vector_batch)
    values = value_net(states_map_batch, states_vector_batch).squeeze()

    # Ensure proper dimensions for values
    if values.dim() == 0:
        values = values.unsqueeze(0)
    if returns_batch.dim() == 0:
        returns_batch = returns_batch.unsqueeze(0)

    # Continuous action log probabilities with error checking
    stds = torch.exp(log_stds.clamp(min=-20, max=2))  # Prevent numerical instability
    log_probs_continuous = -0.5 * (((actions_batch - means) / (stds + 1e-8)) ** 2 + 2 * log_stds + math.log(2 * math.pi))
    log_probs_continuous = log_probs_continuous.sum(dim=1)
    
    # Safer tanh correction term
    tanh_correction = torch.sum(torch.log(1 - actions_batch.pow(2) + 1e-6), dim=1)
    log_probs = log_probs_continuous - tanh_correction

    # Compute advantages with proper normalization
    advantages = returns_batch - values.detach()
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss with proper scaling
    policy_loss = -(log_probs * advantages).mean()

    # Action penalty with proper clamping
    max_action_value = 1.0
    penalty_weight = 0.1  # Reduced from 10000 to prevent overshadowing other losses
    action_penalty = penalty_weight * torch.sum(torch.clamp(actions_batch.abs() - (max_action_value - 0.1), min=0) ** 2)

    # Entropy regularization
    entropy_loss = -ENTROPY_BETA * (log_stds + 0.5 * math.log(2 * math.pi * math.e)).sum(dim=1).mean()
    total_policy_loss = policy_loss + entropy_loss

    # Value loss
    value_loss = nn.functional.mse_loss(values, returns_batch)

    # L2 regularization (weight decay)
    l2_lambda = 1e-1  # Regularization strength
    l2_policy_loss = sum(param.pow(2.0).sum() for param in policy_net.parameters())
    l2_value_loss = sum(param.pow(2.0).sum() for param in value_net.parameters())

    # Add L2 regularization to the losses
    total_policy_loss += l2_lambda * l2_policy_loss
    value_loss += l2_lambda * l2_value_loss

    LossPolicy.append(total_policy_loss.item())
    LossValue.append(value_loss.item())
    LossEntropy.append(entropy_loss.item())
    LossOutbound.append(action_penalty.item())
    LossWeightsPolicy.append(l2_lambda* l2_policy_loss.item())
    LossExploration.append(policy_loss.item())
    LossWeightsValue.append(l2_lambda* l2_value_loss.item())

    # Backpropagation
    optimizer_policy.zero_grad()
    total_policy_loss.backward()
    optimizer_policy.step()

    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

def select_action(policy_net, state_map,state_vector):
        
        state_map = torch.FloatTensor(state_map).to(device)
        state_vector = torch.FloatTensor(state_vector).to(device)
        means, log_stds = policy_net(state_map,state_vector)

        # Sample continuous actions
        stds = torch.exp(log_stds)
        sampled_continuous_actions = means + torch.randn_like(means) * stds

        # print(f"actions before tanh: {sampled_continuous_actions}")
        continuous_actions = torch.tanh(sampled_continuous_actions)
        # print(f"actions after tanh: {continuous_actions}")
        
        # Compute log probabilities 
        # log prob with tanh squashing and gaussian prob
        log_probs_continuous = -0.5 * (((sampled_continuous_actions - means) / (stds + 1e-8)) ** 2 + 2 * log_stds + math.log(2 * math.pi))
        log_probs_continuous = log_probs_continuous.sum(dim=1)

        log_probs_continuous = log_probs_continuous - torch.sum(torch.log(1 - continuous_actions ** 2 + 1e-6), dim=1)

        # Combine actions
        action = continuous_actions[0]
        #print(f"action: {action}")
        
        if torch.isnan(action).any():
            print("NaN detected in action")
            return [0, 0, 0], None
        
        log_prob = log_probs_continuous 
        return action.detach().cpu(), log_prob


def train():
    print("Training...")
    map_training = M1()
    playground = map_training.construct_playground(drone_type=MyDroneHulk)
    policy_net = NetworkPolicy(h=map_training.drones[0].grid.grid.shape[0],w=map_training.drones[0].grid.grid.shape[1])
    value_net = NetworkValue(h=map_training.drones[0].grid.grid.shape[0],w=map_training.drones[0].grid.grid.shape[1])

    optimizer_policy = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    optimizer_value = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)
    policy_net.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    rewards_per_episode = []
    
    for episode in range(NB_EPISODES):
        # reinintialisation de la map pour chaque drone à chaque épisode

        playground.reset()
        for drone in map_training.drones:
            drone.grid.reset()
        #gui = GuiSR(playground=playground, the_map=map_training, draw_semantic_rays=True)
        done = False
        states_map,states_vector, actions, rewards = [], [], [],[]
        total_reward = 0
        step = 0
        nb_rescue = 0
        actions_drones = {}
        playground.step()

        while not done and step < MAX_STEPS and all([drone.drone_health>0 for drone in map_training.drones]):
            step += 1
            # gui.run()  # Run GUI for visualization
            for drone in map_training.drones:
                
                drone.timestep_count = step
                drone.showMaps(display_zoomed_position_grid=True,display_zoomed_grid=True)
                # maps is the occupation map and the position map as a tensor of shape (1, 2, h, w)
                maps = torch.tensor([drone.grid.grid, drone.grid.position_grid], dtype=torch.float32, device=device).unsqueeze(0)
                global_state = torch.tensor([drone.estimated_pose.position[0], drone.estimated_pose.position[1], drone.estimated_pose.orientation, drone.estimated_pose.vitesse_X, drone.estimated_pose.vitesse_Y, drone.estimated_pose.vitesse_angulaire], dtype=torch.float32, device=device).unsqueeze(0)
                

                states_map.append(maps)
                states_vector.append(global_state)


                action, _ = select_action(policy_net,maps, global_state)
                actions_drones[drone] = drone.process_actions(action)
                actions.append(action)

            playground.step(actions_drones)

            for drone in map_training.drones:
                
                drone.update_map_pose_speed() # conséquence de l'action
                found_wounded = drone.process_semantic_sensor()
                min_dist = drone.process_lidar_sensor(drone.lidar())
                is_collision = min_dist < 10
                reward = drone.compute_reward(is_collision, found_wounded, 1,actions_drones[drone])
                rewards.append(reward)
                total_reward += reward

                done = found_wounded
                if done:
                    print("found wounded !")

            
            
        if any([drone.drone_health<=0 for drone in map_training.drones]):
            map_training = M1()
            playground = map_training.construct_playground(drone_type=MyDroneHulk)
        
        # Optimize the policy and value networks in batches
        returns = compute_returns(rewards)
        rewards_per_episode.append(total_reward)

        dataset = DroneDataset(states_map, states_vector, actions, returns)
        data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        for batch in data_loader:
            states_map_b, states_vector_b, actions_b, returns_b = batch
            optimize_batch(states_map_b, states_vector_b, actions_b, returns_b, policy_net, value_net, optimizer_policy, optimizer_value, device)

        del states_map,states_vector, actions, rewards

        if episode % 5 == 1:
            
            print(f"Episode {episode}, Reward: {total_reward}, Mean Last 5 Rewards: {np.mean(rewards_per_episode[-5:])}")
            torch.save(policy_net.state_dict(), 'solutions/utils/trained_models/policy_net.pth')
            torch.save(value_net.state_dict(), 'solutions/utils/trained_models/value_net.pth')
            
            plot = False
            
            # Plot the losses
            if plot or episode % 1000 ==1:
                plt.figure(figsize=(14, 7))
                
                plt.subplot(2, 3, 1)
                plt.plot(LossPolicy)
                plt.title('Policy Loss')
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
                plt.grid()

                plt.subplot(2, 3, 2)
                plt.plot(LossValue)
                plt.title('Value Loss')
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
                plt.grid()

                plt.subplot(2, 3, 3)
                plt.plot(LossEntropy)
                plt.title('Entropy Loss')
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
                plt.grid()

                plt.subplot(2, 3, 4)
                plt.plot(LossOutbound)
                plt.title('Outbound Loss')
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
                plt.grid()

                plt.subplot(2, 3, 5)
                plt.plot(LossWeightsPolicy)
                plt.title('Weights Policy Loss')
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
                plt.grid()

                plt.subplot(2, 3, 6)
                plt.plot(LossExploration)
                plt.title('Exploration Loss')
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
                plt.grid()

                plt.tight_layout()
                plt.show()

            
        if np.mean(rewards_per_episode[-5:]) > 10000:
            print(f"Training solved in {episode} episodes!")
            break


if __name__ == "__main__":
    train()