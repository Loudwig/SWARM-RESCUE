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
import random


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
from solutions.utils.NetworkValuebootstrap import NetworkValue
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
        
        #print(f"actions before dataset {actions}")

        self.actions = torch.stack(actions)
        #print(f"actions after dataset {self.actions}")
        #print(f"returns before DroneDataset {returns}")
        self.returns = torch.FloatTensor(returns)
        #print(f"returns after DroneDataset {self.returns}")


    def __len__(self):
        return len(self.states_maps)

    def __getitem__(self, idx):
        return self.states_maps[idx], self.states_vectors[idx], self.actions[idx], self.returns[idx]

GAMMA = 0.99
LEARNING_RATE = 1e-5
ENTROPY_BETA = 0.15
NB_EPISODES = 500
MAX_STEPS = 64*4 # multiple du batch size c'est mieux sinon des fois on a des batchs pas de la même taille.
BATCH_SIZE = 32 # prendre des puissance de 2
UPDATE_VALUE_NET_PERIOD = 64 # periode d'update du value netork pendant un episode (le policy network lui ne s'update que à la fin de l'épisode)
BATCH_SIZE_VALUE = 32 # batch size pour l'update du value network

LossValue = []
LossPolicy = []
LossEntropy = []
LossOutbound = []
LossWeightsValue = []
LossExploration = []
LossWeightsPolicy = []


def optimize_batch(states_map_batch, states_vector_batch, actions_batch, returns_batch, 
                  policy_net, value_net, optimizer_policy, optimizer_value, device,mode):
    """
    Optimize networks based on collected experience
    
    Args:
        mode (str): Update mode
            - "both": Update both policy and value networks
            - "value": Update only value network
            - "policy": Update only policy network
    """

    if mode == "both" : 
        # Move tensors to device and ensure proper dimensions
        states_map_batch = states_map_batch.to(device)
        states_vector_batch = states_vector_batch.to(device)
        actions_batch = actions_batch.to(device)
        returns_batch = returns_batch.to(device)
        #print(f"return batch  {returns_batch }")

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
        advantages = returns_batch.detach() - values.detach()
        
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss with proper scaling
        policy_loss = -(log_probs * advantages).mean() 
        #print(log_probs)
        max_action_value = 1.0
        penalty_weight = 0.001  # Reduced from 10000 to prevent overshadowing other losses
        action_penalty = penalty_weight * torch.sum(torch.clamp(actions_batch.abs() - (max_action_value - 0.3), min=0) ** 2)

        # Entropy regularization
        entropy_loss = -ENTROPY_BETA * (log_stds + 0.5 * math.log(2 * math.pi * math.e)).sum(dim=1).mean()
        total_policy_loss = policy_loss + entropy_loss # + action_penalty

        # Value loss
        value_loss = nn.functional.mse_loss(values, returns_batch)

        # L2 regularization (weight decay)
        l2_lambda = 1e-6
        l2_policy_loss = sum(torch.sum(param ** 2) for param in policy_net.parameters())
        l2_value_loss = sum(torch.sum(param ** 2) for param in value_net.parameters())

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
        total_norm = 0.0
        for p in policy_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item()**2
        total_norm = total_norm**0.5
        #print(f"[DEBUG] Norme L2 du gradient Policy = {total_norm:.4f}")
        #torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
        optimizer_policy.step()

        optimizer_value.zero_grad()
        value_loss.backward()
        #torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.5)
        optimizer_value.step()
    
    elif mode == "value" : 
        returns_batch = returns_batch.to(device)
        values = value_net(states_map_batch, states_vector_batch).squeeze()
        # Ensure proper dimensions for values
        if values.dim() == 0:
            values = values.unsqueeze(0)
        if returns_batch.dim() == 0:
            returns_batch = returns_batch.unsqueeze(0)
        value_loss = nn.functional.mse_loss(values, returns_batch)

        # L2 regularization (weight decay)
        l2_lambda = 1e-5
        l2_value_loss = sum(torch.sum(param ** 2) for param in value_net.parameters())

        # Add L2 regularization to the losses
        value_loss += l2_lambda * l2_value_loss
        value_loss.backward()
        #torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.5)
        optimizer_value.step()
    
    else : 
        raise ValueError("mode non conforme")
    
def select_action(policy_net, state_map, state_vector):
    # Debugging and type-checking for state_map
    if isinstance(state_map, list):
        state_map = np.array(state_map)
        #print('Converted state_map to numpy array.')
    elif isinstance(state_map, torch.Tensor):
        #print('State_map is already a tensor.')
        pass

    if not isinstance(state_map, torch.Tensor):
        state_map = torch.FloatTensor(state_map)  # Convert to tensor if not already
    state_map = state_map  # Ensure it's on the correct device

    # Debugging and type-checking for state_vector
    if isinstance(state_vector, list):
        state_vector = np.array(state_vector)
        #print('Converted state_vector to numpy array.')
    elif isinstance(state_vector, torch.Tensor):
        #print('State_vector is already a tensor.')
        pass

    if not isinstance(state_vector, torch.Tensor):
        state_vector = torch.FloatTensor(state_vector)  # Convert to tensor if not already
    state_vector = state_vector  # Ensure it's on the correct device

    # Policy network forward pass
    means, log_stds = policy_net(state_map, state_vector)

    # Sample continuous actions
    stds = torch.exp(log_stds)
    sampled_continuous_actions = means + torch.randn_like(means) * stds # utilisation de rsample ? 
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

def train(n_frames_stack=1,n_frame_skip=1,grid_resolution = 8):
    
    print("Training bootstrap")
    print("Using device:", device)

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
        f.write(f"BATCH_SIZE = {BATCH_SIZE}\n")
        f.write(f"frame_stacking = {n_frames_stack}\n")
        f.write(f"frame_skipping = {n_frame_skip}\n")
        f.write(f"grid_resolution = {grid_resolution}\n")
        f.write(f"Other hyperparams...\n")

    map_training = M1()
    playground = map_training.construct_playground(drone_type=MyDroneHulk)

    h_dummy = int(map_training.drones[0].grid.size_area_world[0] / grid_resolution
                                   + 0.5)
    w_dummy  = int(map_training.drones[0].grid.size_area_world[1] / grid_resolution
                                   + 0.5)

    policy_net = NetworkPolicy(h=h_dummy,w=w_dummy,frame_stack=n_frames_stack).to(device)
    value_net = NetworkValue(h=h_dummy,w=w_dummy).to(device)

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
            drone.grid = OccupancyGrid(size_area_world=drone.size_area,
                                resolution=grid_resolution,
                                lidar=drone.lidar()) # On reset la grille + mettre la bonne resolution.
            
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
        next_states_map = []
        next_states_vector = []
        actions = []
        rewards = []
        step = 0
        actions_drones = {drone: drone.process_actions([0, 0, 0]) for drone in map_training.drones}  # Initialize with neutral actions
        total_reward = 0
        done = False


        while not done and step < MAX_STEPS and all([drone.drone_health>0 for drone in map_training.drones]):
            step += 1
            # attendre 30 steps avant de commencer à bouger
            if step < 70:
                #print("Waiting")
                
                if step < 50 : 
                    i = random.random()
                else : 
                    i = 0
                #print(i)
                for drone in map_training.drones:
                    drone.timestep_count = step
                    #drone.showMaps(display_zoomed_position_grid=True, display_zoomed_grid=True)
                    actions_drones = {drone: drone.process_actions([0, 0, i]) for drone in map_training.drones}
                    drone.update_map_pose_speed()
                playground.step(actions_drones)
            else : 
                #print("Moving")
                if step % n_frame_skip == 0:
                    for drone in map_training.drones:
                        drone.timestep_count = step
                        #drone.showMaps(display_zoomed_position_grid=True, display_zoomed_grid=True)
                        
                        # Get current frame
                        current_maps = torch.from_numpy(np.stack((drone.grid.grid, drone.grid.position_grid),axis=0)).unsqueeze(0)
                        current_maps = current_maps.float().to(device)
                        #print(f"shape of the maps : {maps.shape}")

                        current_global_state = drone.process_state_before_network(drone.estimated_pose.position[0],
                            drone.estimated_pose.position[1],
                            drone.estimated_pose.orientation,
                            drone.estimated_pose.vitesse_X,
                            drone.estimated_pose.vitesse_Y,
                            drone.estimated_pose.vitesse_angulaire)
                        

                        # Update frame buffer
                        frame_buffers[drone].append(current_maps)
                        global_state_buffer[drone].append(current_global_state) # global_state_buffer[drone] = deque([tensor([x,y,...],tensor([..],))]
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
                
                if step % n_frame_skip == 0:
                    for drone in map_training.drones:
                        
                        # ON uptdate la map après l'action
                        drone.update_map_pose_speed() # conséquence de l'action

                        # calculate new state s'
                        next_maps = torch.from_numpy(np.stack((drone.grid.grid, drone.grid.position_grid),axis=0)).unsqueeze(0)
                        next_maps = next_maps.float().to(device)
                        next_global_state = drone.process_state_before_network(drone.estimated_pose.position[0],
                            drone.estimated_pose.position[1],
                            drone.estimated_pose.orientation,
                            drone.estimated_pose.vitesse_X,
                            drone.estimated_pose.vitesse_Y,
                            drone.estimated_pose.vitesse_angulaire).unsqueeze(0)
                        
                        
                        with torch.no_grad():
                            value_next = value_net(next_maps, next_global_state).squeeze()


                        found_wounded = drone.process_semantic_sensor()
                        done = found_wounded
                        min_dist = drone.process_lidar_sensor(drone.lidar())
                        is_collision = min_dist < 10
                        
                        reward = drone.compute_reward(is_collision, found_wounded, 1,actions_drones[drone])
                        
                        if done or (step == MAX_STEPS):
                            target = reward
                        else:
                            target = reward + GAMMA * value_next
                        
                        rewards.append(target)
                        
                        total_reward += reward
                        
                        if done:
                            print("found wounded !")

                else : 
                    for drone in map_training.drones:
                        drone.update_map_pose_speed()
                
                if step % UPDATE_VALUE_NET_PERIOD == 0 : 
                    #print(f"updating value at step {step}")
                    rewards_value = rewards[-UPDATE_VALUE_NET_PERIOD:]
                    states_map_value = states_map[-UPDATE_VALUE_NET_PERIOD:]
                    states_vector_value = states_vector[-UPDATE_VALUE_NET_PERIOD:]
                    actions_vector_value = actions[-UPDATE_VALUE_NET_PERIOD:]
                    dataset = DroneDataset(states_map_value, states_vector_value, actions_vector_value, rewards_value)
                    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE_VALUE, shuffle=True,generator = generator)
                    for batch in data_loader:
                        states_map_b, states_vector_b, actions_b, returns_b = batch
                        optimize_batch(states_map_b, states_vector_b, actions_b, returns_b, policy_net, value_net, optimizer_policy, optimizer_value, device,"value")
                            

        if any([drone.drone_health<=0 for drone in map_training.drones]):
            map_training = M1()
            playground = map_training.construct_playground(drone_type=MyDroneHulk)
            

        
        # Optimize the policy and value networks in batches
        returns = rewards
        rewards_per_episode.append(total_reward)

        dataset = DroneDataset(states_map, states_vector, actions, returns)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,generator = generator)
        for batch in data_loader:
            states_map_b, states_vector_b, actions_b, returns_b = batch
            optimize_batch(states_map_b, states_vector_b, actions_b, returns_b, policy_net, value_net, optimizer_policy, optimizer_value, device,"both")

        del states_map,states_vector, actions, rewards
        
        if episode % 10 == 1:
            print(f"Episode {episode}, Reward: {total_reward}, Mean Last 10 Rewards: {np.mean(rewards_per_episode[-10:])}")
            
        if np.mean(rewards_per_episode[-10:]) > 10000:
            print(f"Training solved in {episode} episodes!")
            break
    

    # After training ends, store the loss arrays in a CSV file
    losses_csv_path = os.path.join(folder_name, "losses.csv")
    with open(losses_csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Header
        csv_writer.writerow(["Step", "PolicyLoss", "ValueLoss", "EntropyLoss", "OutboundLoss", "WeightsPolicyLoss", "ExplorationLoss"])
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
    torch.save(policy_net.state_dict(), os.path.join(folder_name,"policy_net.pth"))
    torch.save(value_net.state_dict(), os.path.join(folder_name,"value_net.pth"))    
    print("Training complete. Files saved in:", folder_name)

if __name__ == "__main__":
    train()