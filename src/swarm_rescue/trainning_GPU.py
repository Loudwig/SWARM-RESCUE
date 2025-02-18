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
# from solutions.utils.NetworkPolicy import NetworkPolicy
# from solutions.utils.NetworkValue import NetworkValue
from solutions.my_drone_A2C_trainning import MyDroneHulk
from solutions.utils.ActorCriticNetwork import ActorCriticNetwork
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
generator = torch.Generator(device=device)

GAMMA = 0.99 # how much future reward are taken into account
LEARNING_RATE = 1e-4
ENTROPY_BETA = 1e-3
NB_EPISODES = 500
MAX_STEPS = 64*2 + 69 # multiple du batch size c'est mieux sinon des fois on a des batchs pas de la même taille.
BATCH_SIZE = 8 # prendre des puissance de 2
NUM_EPOCHS = 4
OUTBOUND_LOSS = True
ENTROPY_LOSS = True
WEIGHTS_LOSS_VALUE_NET = True
WEIGHTS_LOSS_POLICY_NET = True
CLIP_GRADIENTS_POLICY = True
CLIP_GRADIENTS_VALUE = True

class DroneDataset:
    def __init__(self, actions, returns, advantages, logprobs, values):
        # No gradients needed
        self.actions = torch.stack(actions).to(torch.float32).to(device)
        self.returns = torch.tensor(returns, dtype=torch.float32).to(device)
        self.advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        
        # Preserve gradients
        self.logprobs = torch.tensor(logprobs, dtype=torch.float32).clone().detach().requires_grad_(True).to(device)
        self.values = torch.tensor(values, dtype=torch.float32).clone().detach().requires_grad_(True).to(device)


    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.actions[idx], self.returns[idx], self.advantages[idx], self.logprobs[idx],self.values[idx]

PARAMS = {
    "learning_rate_policy" : LEARNING_RATE,
    "entropy_beta" : ENTROPY_BETA,
    "nb_episodes" : NB_EPISODES,
    "max_steps" : MAX_STEPS,
    "batch_size" : BATCH_SIZE,
    "outbound_loss" : OUTBOUND_LOSS,
    "entropy_loss" : ENTROPY_LOSS,
    "NUM_EPOCHS" : NUM_EPOCHS,
    "weights_loss_value_net" : WEIGHTS_LOSS_VALUE_NET,
    "weights_loss_policy_net" : WEIGHTS_LOSS_POLICY_NET,

} 

# Losses
LossValue = []
LossPolicy = []
LossEntropy = []
LossOutbound = []
LossWeightsValue = []
LossExploration = []
LossWeightsPolicy = []


def optimize_batch(actions_b, return_b, advantages_b,logprobs_b,values_b, optimizer,policy_net,value_net, device):
    """
    Optimize networks based on collected experience
    
    Args:
        mode (str): Update mode
            - "value": Update only value network
            - "policy": Update only policy network
    """



    # Move tensors to device and ensure proper dimensions
    action_batch = actions_b.to(device)
    return_batch = return_b.to(device)
    advantages_batch = advantages_b.to(device)
    logprob_batch = logprobs_b.to(device)
    values_batch = values_b.to(device)

    # Forward pass
    # means, log_stds = policy_net(states_map_batch, states_vector_batch)
    # values = value_net(states_map_batch, states_vector_batch).squeeze()

    # # Ensure proper dimensions for values
    # if values.dim() == 0:
    #     values = values.unsqueeze(0)
    if return_batch.dim() == 0:
        return_batch = return_batch.unsqueeze(0)

    # # Continuous action log probabilities with error checking
    # stds = torch.exp(log_stds.clamp(min=-20, max=2))  # Prevent numerical instability
    # log_probs_continuous = -0.5 * (((actions_batch - means) / (stds + 1e-8)) ** 2 + 2 * log_stds + math.log(2 * math.pi))
    # log_probs_continuous = log_probs_continuous.sum(dim=1)
    
    # # Safer tanh correction term
    # tanh_correction = torch.sum(torch.log(1 - actions_batch.pow(2) + 1e-6), dim=1)
    # log_probs = log_probs_continuous - tanh_correction

    # Compute advantages with proper normalization
    # advantages = returns_batch.detach() - values.detach()
    
    if advantages_batch.numel() > 1:
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

    # Policy loss with proper scaling
    policy_loss = -(logprob_batch * advantages_batch).mean() 

    ##penalty_weight = 1e-5 # 
    # PENELASIZE IF MEANS ARE OUTSIDE OF -1 ;1 
    ##outbound_loss = penalty_weight* torch.mean((torch.relu(torch.abs(means) - 1)))

    # Entropy regularization
    entropy_loss = -ENTROPY_BETA * logprob_batch.mean()
    
    # L2 regularization (weight decay)
    l2_lambda = 1e-6
    #l2_policy_loss = l2_lambda* sum(torch.sum(param ** 2) for param in policy_net.parameters())
    #l2_value_loss = l2_lambda* sum(torch.sum(param ** 2) for param in value_net.parameters())

    total_policy_loss = policy_loss
    value_loss = nn.functional.mse_loss(values_batch, return_batch)

    # if OUTBOUND_LOSS:
    #     total_policy_loss += outbound_loss
    if ENTROPY_LOSS:
        total_policy_loss += entropy_loss
    # if WEIGHTS_LOSS_POLICY_NET: 
    #     total_policy_loss += l2_policy_loss
    # if WEIGHTS_LOSS_VALUE_NET:
    #     total_policy_loss += l2_value_loss
    total_policy_loss += value_loss

    LossPolicy.append(total_policy_loss.item())
    LossValue.append(value_loss.item()) # just for logging
    LossEntropy.append(entropy_loss.item())
    LossOutbound.append(0)
    LossWeightsPolicy.append(0)
    LossExploration.append(policy_loss.item())
    LossWeightsValue.append(0)

    optimizer.zero_grad()
    total_policy_loss.backward()
    
    if CLIP_GRADIENTS_POLICY:
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
    if CLIP_GRADIENTS_VALUE:
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.5)
    
    optimizer.step()

def compute_reward(self,is_collision, found_wounded, time_penalty,action):
        forward = action["forward"]
        lateral = action["lateral"]
        rotation = action["rotation"]

        reward = 0
        reward += self.grid.exploration_score

        # Penalize collisions heavily
        if is_collision:
            reward -= 30
       
        # Penalize idling or lack of movement
        reward -= time_penalty

        # give penalty when action are initialy not between -0.9 and 0.9 
        if (abs(forward) < 0.5 and abs(lateral) < 0.5 and abs(rotation) < 0.5):
            #print("actions not saturated")
            pass
          
        return reward
   
def select_action(actor_critic_net, state_map, state_vector):
    # S'assurer que les entrées sont des tensors sur le bon device
    if not isinstance(state_map, torch.Tensor):
        state_map = torch.FloatTensor(state_map).to(device)
    if not isinstance(state_vector, torch.Tensor):
        state_vector = torch.FloatTensor(state_vector).to(device)
    state_vector = state_vector.squeeze().unsqueeze(0)
    
    # Appel du réseau : on récupère mu, log_sigma et aussi la valeur (que l'on peut ignorer ici)
    mu, log_sigma, _ = actor_critic_net(state_map, state_vector)
    
    # Calcul de l'action
    stds = torch.exp(log_sigma)
    sampled_continuous_actions = mu + torch.randn_like(mu) * stds
    continuous_actions = torch.tanh(sampled_continuous_actions)
    
    # Calcul du log_prob (avec correction tanh)
    log_probs_continuous = -0.5 * (
        ((sampled_continuous_actions - mu) / (stds + 1e-8)) ** 2 +
        2 * log_sigma + math.log(2 * math.pi)
    )
    log_probs_continuous = log_probs_continuous.sum(dim=1)
    log_probs_continuous -= torch.sum(torch.log(1 - continuous_actions ** 2 + 1e-6), dim=1)
    
    action = continuous_actions[0]
    if torch.isnan(action).any():
        print("NaN detected in action")
        return [0, 0, 0], None

    return action.detach(), log_probs_continuous

def train(n_frames_stack=1,n_frame_skip=1,grid_resolution = 8,map_channels = 2):
    
    PARAMS["map_channels"] = map_channels
    PARAMS["n_frames_stack"] = n_frames_stack
    PARAMS["n_frame_skip"] = n_frame_skip
    PARAMS["grid_resolution"] = grid_resolution
    print("Training bootstrap")
    print("Using device:", device)

    # Create a unique folder name based on hyperparameters and timestamp
    current_time = time.strftime("%Y%m%d-%H%M%S")
    folder_name = f"solutions/trained_models/run_{current_time}"
    os.makedirs(folder_name, exist_ok=True)

    # Save hyperparameters to a text file
    hyperparams_path = os.path.join(folder_name, "hyperparams.txt")
    with open(hyperparams_path, 'w') as f:
        # Write hyperparameters from the PARAMS DICT : 
        for key, value in PARAMS.items():
            f.write(f"{key}: {value}\n")

    map_training = M1()
    playground = map_training.construct_playground(drone_type=MyDroneHulk)

    h_dummy = int(map_training.drones[0].grid.size_area_world[0] / grid_resolution
                                   + 0.5)
    w_dummy  = int(map_training.drones[0].grid.size_area_world[1] / grid_resolution
                                   + 0.5)

    actor_critic_net = ActorCriticNetwork(h=h_dummy, w=w_dummy, frame_stack=n_frames_stack, map_channels=map_channels).to(device)
    optimizer = optim.Adam(actor_critic_net.parameters(), lr=LEARNING_RATE)
    actor_critic_net.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    
    # Initialize frame buffers
    frame_buffers = {drone: deque(maxlen=n_frames_stack) for drone in map_training.drones}
    global_state_buffer = { drone : deque(maxlen= n_frames_stack) for drone in map_training.drones}
    rewards_per_episode = []
    done = False
    DroneDestroyed = []
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

            dummy_maps = torch.zeros((1, map_channels,map_training.drones[0].grid.grid.shape[0], map_training.drones[0].grid.grid.shape[1]), device=device)
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
        values = []
        logprobs = []
        step = 0
        actions_drones = {drone: drone.process_actions([0, 0, 0]) for drone in map_training.drones}  # Initialize with neutral actions
        total_reward = 0
        done = False


        while not done and step < MAX_STEPS and all([drone.drone_health>0 for drone in map_training.drones]):
            step += 1
            # attendre 70 steps avant de commencer à apprendre (pour que les drones aient une carte un peu remplie) sinon il vont "explorer" 
            # meme si leur action sont mauvaises

            # WAIT BEFORE TRAINNING
            if step < 70:
                if step < 50 : 
                    i = random.random() # faire tourner le drone pour que le lidar prêne tout
                else : 
                    i = 0
                for drone in map_training.drones:
                    drone.timestep_count = step
                    #drone.showMaps(display_zoomed_position_grid=False, display_zoomed_grid=False)
                    actions_drones = {drone: drone.process_actions([0, 0, i]) for drone in map_training.drones}
                    drone.update_map_pose_speed()
                playground.step(actions_drones)
            
            # TRAINNING STARTS HERE
            else : 
                """
                Première étape : 
                - On récupère l'état s (map + global state)
                - On choisit une action a
                """   
             
                if step % n_frame_skip == 0: # frame skipping
                    for drone in map_training.drones:
                        drone.timestep_count = step
                        #drone.showMaps(display_zoomed_position_grid=True, display_zoomed_grid=True)
                        
                        # 2 map channels : grid + position grid
                        if map_channels ==2 : 
                            current_maps = torch.from_numpy(np.stack((drone.grid.grid, drone.grid.position_grid),axis=0)).unsqueeze(0)
                        # 1 map channel : grid
                        elif map_channels == 1 :
                            current_maps = torch.from_numpy(drone.grid.grid).unsqueeze(0).unsqueeze(0)
                        else : 
                            raise ValueError("map_channels must be 1 or 2")
                        
                        current_maps = current_maps.float().to(device)
                        current_global_state = drone.process_state_before_network(drone.estimated_pose.position[0],
                            drone.estimated_pose.position[1],
                            drone.estimated_pose.orientation,
                            drone.estimated_pose.vitesse_X,
                            drone.estimated_pose.vitesse_Y,
                            drone.estimated_pose.vitesse_angulaire).unsqueeze(0)
                        #with torch.no_grad():
                            # DEBUGGING SIZE 
                            #print(f"current maps shape : {current_maps.shape}")
                        value_current = value_net(current_maps, current_global_state).squeeze() 
                        
                        values.append(value_current)

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
                        action, logprob = select_action(policy_net, stacked_frames, stacked_global_states)
                        # action : tensor([1,-1,1])
                        actions_drones[drone] = drone.process_actions(action)
                        actions.append(action)
                        logprobs.append(logprob)
                
                """"
                Deuxième étape :
                - On applique l'action sur l'environnement
                """

                # Si on est dans une skip-frame cela va répéter la dernière action.
                # Si non cela fait la nouvelle action
                playground.step(actions_drones) 

                """
                Troisième étape :
                - On update l'environnement
                - On calcule la récompense r
                """
                if step % n_frame_skip == 0:
                    for drone in map_training.drones:
                        
                        # ON uptdate la map après l'action
                        drone.update_map_pose_speed() # conséquence de l'action

                        # # calculate new state s'
                        # # 2 map channels : grid + position grid
                        # if map_channels ==2 :
                        #     next_maps = torch.from_numpy(np.stack((drone.grid.grid, drone.grid.position_grid),axis=0)).unsqueeze(0)
                        # elif map_channels == 1 :
                        #     next_maps = torch.from_numpy(drone.grid.grid).unsqueeze(0).unsqueeze(0)
                        # next_maps = next_maps.float().to(device)
                        # next_global_state = drone.process_state_before_network(drone.estimated_pose.position[0],
                        #     drone.estimated_pose.position[1],
                        #     drone.estimated_pose.orientation,
                        #     drone.estimated_pose.vitesse_X,
                        #     drone.estimated_pose.vitesse_Y,
                        #     drone.estimated_pose.vitesse_angulaire).unsqueeze(0)
                        
                        
                        # with torch.no_grad():
                        #     value_next = value_net(next_maps, next_global_state).squeeze()


                        found_wounded = drone.process_semantic_sensor()
                        #done = found_wounded
                        min_dist = drone.process_lidar_sensor(drone.lidar())
                        is_collision = min_dist < 10
                        
                        reward = compute_reward(drone,is_collision, found_wounded, 1,actions_drones[drone])
                        
                        # if (step == MAX_STEPS):
                        #     # compute total exploration score
                        #     target = np.exp(5 * drone.grid.grid_score / (drone.grid.x_max_grid * drone.grid.y_max_grid))
                        #     print(f"Exploration score: {target}")
                        # else:
                        #     target = reward + GAMMA * value_next
                        
                        rewards.append(reward)
                        total_reward += reward
                        
                        if found_wounded:
                            print("found wounded !")

                else : 
                    print("Ne devrait pas passer ici")
                    for drone in map_training.drones:
                        drone.update_map_pose_speed()
                
                # if step % UPDATE_VALUE_NET_PERIOD == 0 : 
                #     # UPTDATE VALUE NETWORK
                    
        if any([drone.drone_health<=0 for drone in map_training.drones]):
            DroneDestroyed.append(1)
            map_training = M1()
            playground = map_training.construct_playground(drone_type=MyDroneHulk)
        else : 
            DroneDestroyed.append(0)
        
        with torch.no_grad():
            # Get final value estimate
            next_state_map = states_map[-1]  
            next_state_vector = states_vector[-1].squeeze().unsqueeze(0)
            next_value = value_net(next_state_map, next_state_vector).squeeze()

            # Initialize advantages array
            advantages = torch.zeros_like(torch.tensor(rewards)).to(device)
            lastgaelam = 0

            # Compute GAE
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - (t == len(rewards) - 1)  # Only terminal on last step
                    nextvalues = values[t + 1]
                    
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + GAMMA * 0.95 * nextnonterminal * lastgaelam

            # Compute returns
            returns = advantages + torch.tensor(values).to(device)
                    

        # Train for a number of epochs
        for epoch in range(NUM_EPOCHS):
            # Create new dataset and dataloader for each epoch
            dataset = DroneDataset(actions, returns, advantages,logprobs,values)
            data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
            
            for batch in data_loader:
                actions_b, return_b, advantages_b,logprobs_b,values_b = batch
                optimize_batch(actions_b, return_b, advantages_b,logprobs_b, values_b,optimizer,policy_net,value_net, device)

        returns = rewards
        rewards_per_episode.append(total_reward)

        del states_map, states_vector, actions, rewards
        
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
        csv_writer.writerow(["Step", "PolicyLoss", "ValueLoss", "EntropyLoss", "OutboundLoss", "WeightsPolicyLoss","WeightsValueLoss","ExplorationLoss"])
        # Rows
        for i in range(len(LossPolicy)):
            csv_writer.writerow([
                i,
                LossPolicy[i],
                LossValue[i],
                LossEntropy[i],
                LossOutbound[i],
                LossWeightsPolicy[i],
                LossWeightsValue[i],
                LossExploration[i],
            ])

    # idem with rewards per episode : 
    rewards_csv_path = os.path.join(folder_name, "rewards_per_episode.csv")
    with open(rewards_csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Header
        csv_writer.writerow(["", "TotalReward"])
        # Rows
        for i, reward in enumerate(rewards_per_episode):
            csv_writer.writerow([i, reward])
    drone_destroyed_csv_path = os.path.join(folder_name, "drone_destroyed.csv")
    with open(drone_destroyed_csv_path,"w",newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Step","DroneDestroyed"])
        for i, drone_destroyed in enumerate(DroneDestroyed):
            csv_writer.writerow([i,drone_destroyed])

    torch.save(policy_net.state_dict(), os.path.join(folder_name,"policy_net.pth"))
    torch.save(value_net.state_dict(), os.path.join(folder_name,"value_net.pth"))    
    print("Training complete. Files saved in:", folder_name)

if __name__ == "__main__":
    train()