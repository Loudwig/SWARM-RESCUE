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
import time
import os
import csv


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

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
generator = torch.Generator(device=device)



class DroneDataset:
    def __init__(self, states_maps, states_vectors, actions, returns):
        # Ensure all elements in states_maps are tensors on the correct device
        self.states_maps = torch.stack([
            s.squeeze(0) if isinstance(s, torch.Tensor) else torch.FloatTensor(s).squeeze(0)
            for s in states_maps
        ])
        
        # Similarly handle states_vectors, actions, and returns if needed
        self.states_vectors = torch.stack([sv.squeeze(0) for sv in states_vectors])
        self.actions = torch.stack(actions)
        self.returns = torch.FloatTensor(returns)

    def __len__(self):
        return len(self.states_maps)

    def __getitem__(self, idx):
        return self.states_maps[idx], self.states_vectors[idx], self.actions[idx], self.returns[idx]

GAMMA = 0.99
LEARNING_RATE = 5e-6
ENTROPY_BETA = 0.1
NB_EPISODES = 2000
MAX_STEPS = 400
BATCH_SIZE = 8

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
def select_action(policy_net, state_map, state_vector):
    # Debugging and type-checking for state_map
    if isinstance(state_map, list):
        state_map = np.array(state_map)
        #print('Converted state_map to numpy array.')
    elif isinstance(state_map, torch.Tensor):
        #print('State_map is already a tensor.')
        pass
    #print('State map shape before processing:', state_map.shape)

    if not isinstance(state_map, torch.Tensor):
        state_map = torch.FloatTensor(state_map)  # Convert to tensor if not already
    state_map = state_map  # Ensure it's on the correct device

    #print('State map shape after processing:', state_map.shape)
    #print('State map device:', state_map.device)

    # Debugging and type-checking for state_vector
    if isinstance(state_vector, list):
        state_vector = np.array(state_vector)
        #print('Converted state_vector to numpy array.')
    elif isinstance(state_vector, torch.Tensor):
        #print('State_vector is already a tensor.')
        pass
    #print('State vector shape before processing:', state_vector.shape)

    if not isinstance(state_vector, torch.Tensor):
        state_vector = torch.FloatTensor(state_vector)  # Convert to tensor if not already
    state_vector = state_vector  # Ensure it's on the correct device

    #print('State vector shape after processing:', state_vector.shape)
    #print('State vector device:', state_vector.device)

    # Policy network forward pass
    means, log_stds = policy_net(state_map, state_vector)

    # Sample continuous actions
    stds = torch.exp(log_stds)
    sampled_continuous_actions = means + torch.randn_like(means) * stds
    continuous_actions = torch.tanh(sampled_continuous_actions)

    # Compute log probabilities
    log_probs_continuous = -0.5 * (
        ((sampled_continuous_actions - means) / (stds + 1e-8)) ** 2 +
        2 * log_stds + math.log(2 * math.pi)
    )
    log_probs_continuous = log_probs_continuous.sum(dim=1)
    log_probs_continuous -= torch.sum(torch.log(1 - continuous_actions ** 2 + 1e-6), dim=1)

    # Combine actions
    action = continuous_actions[0]
    if torch.isnan(action).any():
        print("NaN detected in action")
        return [0, 0, 0], None

    log_prob = log_probs_continuous

    return action.detach(), log_prob




def train(n_frames_stack=4):
    
    print("Training...")

    # Create a unique folder name based on hyperparameters and timestamp
    current_time = time.strftime("%Y%m%d-%H%M%S")
    folder_name = f"solutions/trained_models/run_lr_{LEARNING_RATE}_episodes_{NB_EPISODES}_{current_time}"
    os.makedirs(folder_name, exist_ok=True)

    # Save hyperparameters to a text file
    hyperparams_path = os.path.join(folder_name, "hyperparams.txt")
    with open(hyperparams_path, 'w') as f:
        f.write(f"LEARNING_RATE = {LEARNING_RATE}\n")
        f.write(f"ENTROPY_BETA = {ENTROPY_BETA}\n")
        f.write(f"NB_EPISODES = {NB_EPISODES}\n")
        f.write(f"MAX_STEPS = {MAX_STEPS}\n")
        f.writ(f"BATCH_SIZE = {BATCH_SIZE}\n")
        f.write(f"Other hyperparams...\n")

    map_training = M1()
    playground = map_training.construct_playground(drone_type=MyDroneHulk)
    print("Using device:", device)
    policy_net = NetworkPolicy(h=map_training.drones[0].grid.grid.shape[0],w=map_training.drones[0].grid.grid.shape[1],frame_stack=n_frames_stack).to(device)
    value_net = NetworkValue(h=map_training.drones[0].grid.grid.shape[0],w=map_training.drones[0].grid.grid.shape[1],frame_stack=n_frames_stack).to(device)

    optimizer_policy = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    optimizer_value = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)
    policy_net.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    value_net.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    # Initialize frame buffers
    frame_buffers = {drone: deque(maxlen=n_frames_stack) for drone in map_training.drones}
    global_state_buffer = { drone : deque(maxlen= n_frames_stack) for drone in map_training.drones}
    rewards_per_episode = []
    done = False

    for episode in range(NB_EPISODES):
        playground.reset()
        
        # resetting all needs to be done for each drone
        for drone in map_training.drones:
            drone.grid.reset()
            
            if drone in frame_buffers: 
                frame_buffers[drone].clear()
            else : # if the drone was destroyed a new drone is created. 
                frame_buffers[drone] = deque(maxlen=n_frames_stack)
            
            # Initialize with dummy frames
            dummy_maps = torch.zeros((1, 2,map_training.drones[0].grid.grid.shape[0], map_training.drones[0].grid.grid.shape[1]), device=device)
            for _ in range(n_frames_stack):
                frame_buffers[drone].append(dummy_maps)

            if drone in global_state_buffer: 
                global_state_buffer[drone].clear()
            else :
                global_state_buffer[drone] = deque(maxlen=n_frames_stack)
            
            # initialize with dummy global states

            dummy_state = torch.zeros((6), device=device)
            for _ in range(n_frames_stack):
                global_state_buffer[drone].append(dummy_state)

        # initialisations    
        states_map = []
        states_vector = []
        actions = []
        rewards = []
        step = 0
        actions_drones = {drone: [0, 0, 0] for drone in map_training.drones}  # Initialize with neutral actions
        total_reward = 0


        while not done and step < MAX_STEPS and all([drone.drone_health>0 for drone in map_training.drones]):
            step += 1
            
            for drone in map_training.drones:
                drone.timestep_count = step
                #drone.showMaps(display_zoomed_position_grid=True, display_zoomed_grid=True)
                
                # Get current frame
                maps = torch.tensor([drone.grid.grid, drone.grid.position_grid], 
                                  dtype=torch.float32, device=device).unsqueeze(0)
                

                global_state = drone.process_state_before_network(drone.estimated_pose.position[0],
                    drone.estimated_pose.position[1],
                    drone.estimated_pose.orientation,
                    drone.estimated_pose.vitesse_X,
                    drone.estimated_pose.vitesse_Y,
                    drone.estimated_pose.vitesse_angulaire)
                
                # Update frame buffer
                frame_buffers[drone].append(maps)
                global_state_buffer[drone].append(global_state) # global_state_buffer[drone] = deque([tensor([x,y,...],tensor([..],))]
                # Stack frames for network input
                stacked_frames = torch.cat(list(frame_buffers[drone]), dim=1)
                # Stack global states for network input
                stacked_global_states = torch.cat(list(global_state_buffer[drone])).unsqueeze(0) # global_states = tensor([x1,y1,..,x2,y2,...) dim = 
                
                states_map.append(stacked_frames)
                states_vector.append(stacked_global_states)
                
                #print(f"states vector{type(states_vector)}")
                # Select action using stacked frames
                action, _ = select_action(policy_net, stacked_frames, stacked_global_states)
                # action : tensor([1,-1,1])
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
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,generator = generator)
        for batch in data_loader:
            states_map_b, states_vector_b, actions_b, returns_b = batch
            optimize_batch(states_map_b, states_vector_b, actions_b, returns_b, policy_net, value_net, optimizer_policy, optimizer_value, device)

        del states_map,states_vector, actions, rewards

        if np.mean(rewards_per_episode[-5:]) > 10000:
            print(f"Training solved in {episode} episodes!")
            break
    

    # After training ends, store the loss arrays in a CSV file
    losses_csv_path = os.path.join(folder_name, "losses.csv")
    with open(losses_csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Header
        csv_writer.writerow(["Step", "PolicyLoss", "ValueLoss", "EntropyLoss", "OutboundLoss", "WeightsPolicyLoss", "ExplorationLoss","RewardPerEpisode"])
        # Rows
        for i in range(len(LossPolicy)):
            csv_writer.writerow([
                i,
                LossPolicy[i],
                LossValue[i],
                LossEntropy[i],
                LossOutbound[i],
                LossWeightsPolicy[i],
                LossExploration[i],
            ])

    # idem with rewards per episode : 
    rewards_csv_path = os.path.join(folder_name, "rewards_per_episode.csv")
    with open(rewards_csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Header
        csv_writer.writerow(["Step", "TotalReward"])
        # Rows
        for i, reward in enumerate(rewards_per_episode):
            csv_writer.writerow([i, reward])
            
    print("Training complete. Files saved in:", folder_name)

if __name__ == "__main__":
    train()