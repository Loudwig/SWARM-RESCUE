from collections import deque
import math
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
from solutions.my_drone_A2C_trainning import MyDroneHulk

from torch.utils.data import Dataset, DataLoader

# Configuration et device
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
generator = torch.Generator(device=device)

GAMMA = 0.99             # Discount factor
LEARNING_RATE = 1e-4
ENTROPY_BETA = 1e-3
NB_EPISODES = 500
MAX_STEPS = 64*2 + 69    # en lien avec la taille du batch
BATCH_SIZE = 8
NUM_EPOCHS = 4
OUTBOUND_LOSS = True
ENTROPY_LOSS = True
WEIGHTS_LOSS_VALUE_NET = True
WEIGHTS_LOSS_POLICY_NET = True
CLIP_GRADIENTS_POLICY = True
CLIP_GRADIENTS_VALUE = True

# Listes pour le suivi des losses
LossPolicy = []
LossValue = []
LossEntropy = []
LossOutbound = []
LossWeightsValue = []
LossExploration = []
LossWeightsPolicy = []

# =============================================================================
# Définition du réseau Actor-Critic (partie CNN commune et deux têtes)
# =============================================================================
class ActorCriticNetwork(nn.Module):
    def __init__(self, 
                 map_channels=2, 
                 h=100, 
                 w=63, 
                 cnn_output_dim=64, 
                 global_state_dim=6, 
                 hidden_size=32, 
                 num_actions=3, 
                 frame_stack=1):
        super(ActorCriticNetwork, self).__init__()
        
        # Si plusieurs frames sont empilées, le nombre de canaux d'entrée est multiplié
        self.input_channels = map_channels * frame_stack

        # Partie CNN partagée
        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )
        
        # Calcul de la taille après le flatten
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, h, w)
            dummy_output = self.cnn(dummy_input)
            cnn_flatten_size = dummy_output.numel()
            
        self.fc_cnn = nn.Linear(cnn_flatten_size, cnn_output_dim)
        
        # Dimension commune : concaténation de la sortie CNN et de l'état global
        shared_dim = cnn_output_dim + (global_state_dim * frame_stack)
        
        # Tête de la politique
        self.policy_mlp = nn.Sequential(
            nn.Linear(shared_dim, hidden_size),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_size, num_actions)
        self.log_sigma_head = nn.Linear(hidden_size, num_actions)
        
        # Tête de la valeur
        self.value_mlp = nn.Sequential(
            nn.Linear(shared_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, map, global_state):
        x = self.cnn(map)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc_cnn(x))
        
        # Concaténation des features CNN et de l'état global
        x_combined = torch.cat((x, global_state), dim=1)
        
        # Branche politique
        policy_features = self.policy_mlp(x_combined)
        mu = self.mu_head(policy_features)
        log_sigma = self.log_sigma_head(policy_features)
        
        # Branche valeur
        value = self.value_mlp(x_combined)
        
        return mu, log_sigma, value

# =============================================================================
# Fonction de sélection d'action (utilise uniquement la branche policy)
# =============================================================================
def select_action(actor_critic_net, state_map, state_vector):
    # S'assurer que les entrées sont des tensors sur le device
    if not isinstance(state_map, torch.Tensor):
        state_map = torch.FloatTensor(state_map).to(device)
    if not isinstance(state_vector, torch.Tensor):
        state_vector = torch.FloatTensor(state_vector).to(device)
    state_vector = state_vector.squeeze().unsqueeze(0)
    
    mu, log_sigma, _ = actor_critic_net(state_map, state_vector)
    stds = torch.exp(log_sigma)
    sampled_actions = mu + torch.randn_like(mu) * stds
    continuous_actions = torch.tanh(sampled_actions)
    
    # Calcul du log-prob avec correction pour tanh
    log_probs = -0.5 * (((sampled_actions - mu)/(stds+1e-8))**2 + 2*log_sigma + math.log(2*math.pi))
    log_probs = log_probs.sum(dim=1)
    log_probs = log_probs - torch.sum(torch.log(1 - continuous_actions**2 + 1e-6), dim=1)
    
    action = continuous_actions[0]
    if torch.isnan(action).any():
        print("NaN detected in action")
        return [0, 0, 0], None
    return action.detach(), log_probs

# =============================================================================
# Dataset personnalisé qui inclut aussi les états (map et global_state)
# =============================================================================
class DroneDataset(Dataset):
    def __init__(self, states_map, states_vector, actions, returns, advantages, logprobs, values):
        # Ensure correct batch dimension
        self.states_map = torch.stack([
            s.squeeze(0) if isinstance(s, torch.Tensor) else torch.FloatTensor(s).squeeze(0)
            for s in states_map
        ]).to(device)  # [batch, channels, height, width]

        self.states_vector = torch.stack([
            sv.squeeze() if sv.dim() > 1 else sv 
            for sv in states_vector
        ]).to(device)  # [batch, feature_dim]
        
        # Other tensors
        self.actions = torch.stack(actions).to(torch.float32).to(device)
        self.returns = torch.tensor(returns, dtype=torch.float32).to(device)
        self.advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        self.logprobs = torch.tensor(logprobs, dtype=torch.float32).clone().detach().requires_grad_(True).to(device)
        self.values = torch.tensor(values, dtype=torch.float32).clone().detach().requires_grad_(True).to(device)
        
        # Debug prints
       # print(f"states_map shape: {self.states_map.shape}")
        #print(f"states_vector shape: {self.states_vector.shape}")
        
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, idx):
        return (
            self.states_map[idx],  # [channels, height, width]
            self.states_vector[idx],  # [feature_dim]
            self.actions[idx],
            self.returns[idx],
            self.advantages[idx],
            self.logprobs[idx],
            self.values[idx]
        )
# =============================================================================
# Fonction d'optimisation du batch (recalcule les sorties actuelles du réseau)
# =============================================================================
def optimize_batch(states_map_b, states_vector_b, actions_b, return_b, advantages_b, logprobs_b, values_b, optimizer, actor_critic_net, device):
    mu, log_sigma, values_pred = actor_critic_net(states_map_b, states_vector_b)
    # Ici, nous utilisons les logprobs stockés pour la loss de la politique.
    
    # DEBUG shapee values pred and return
    #print(f"values_pred shape: {values_pred.shape}")
    #print(f"return_b shape: {return_b.shape}")

    values_pred = values_pred.squeeze()

    policy_loss = -(logprobs_b * advantages_b).mean()
    entropy_loss = -ENTROPY_BETA * logprobs_b.mean()
    value_loss = nn.functional.mse_loss(values_pred, return_b)
    
    total_loss = policy_loss + entropy_loss + value_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic_net.parameters(), max_norm=0.5)
    optimizer.step()
    
    LossPolicy.append(total_loss.item())
    LossValue.append(value_loss.item())
    LossEntropy.append(entropy_loss.item())

# =============================================================================
# Fonction de calcul de la récompense
# =============================================================================
def compute_reward(drone, is_collision, found_wounded, time_penalty, action):
    forward = action["forward"]
    lateral = action["lateral"]
    rotation = action["rotation"]
    reward = 0
    reward += drone.grid.exploration_score
    if is_collision:
        reward -= 30
    reward -= time_penalty
    if abs(forward) < 0.5 and abs(lateral) < 0.5 and abs(rotation) < 0.5:
        pass
    return reward

# =============================================================================
# Fonction de training principale
# =============================================================================
def train(n_frames_stack=1, n_frame_skip=1, grid_resolution=8, map_channels=2):
    PARAMS = {
        "learning_rate_policy": LEARNING_RATE,
        "entropy_beta": ENTROPY_BETA,
        "nb_episodes": NB_EPISODES,
        "max_steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "outbound_loss": OUTBOUND_LOSS,
        "entropy_loss": ENTROPY_LOSS,
        "NUM_EPOCHS": NUM_EPOCHS,
        "weights_loss_value_net": WEIGHTS_LOSS_VALUE_NET,
        "weights_loss_policy_net": WEIGHTS_LOSS_POLICY_NET,
        "map_channels": map_channels,
        "n_frames_stack": n_frames_stack,
        "n_frame_skip": n_frame_skip,
        "grid_resolution": grid_resolution,
    }
    print("Training bootstrap")
    print("Using device:", device)
    
    # Création d'un dossier unique pour cette run
    current_time = time.strftime("%Y%m%d-%H%M%S")
    folder_name = f"solutions/trained_models/run_{current_time}"
    os.makedirs(folder_name, exist_ok=True)
    hyperparams_path = os.path.join(folder_name, "hyperparams.txt")
    with open(hyperparams_path, 'w') as f:
        for key, value in PARAMS.items():
            f.write(f"{key}: {value}\n")
    
    # Instanciation de l'environnement
    map_training = M1()
    playground = map_training.construct_playground(drone_type=MyDroneHulk)
    
    h_dummy = int(map_training.drones[0].grid.size_area_world[0] / grid_resolution + 0.5)
    w_dummy = int(map_training.drones[0].grid.size_area_world[1] / grid_resolution + 0.5)
    
    # Instanciation du réseau Actor-Critic et de l'optimiseur
    actor_critic_net = ActorCriticNetwork(h=h_dummy, w=w_dummy, frame_stack=n_frames_stack, map_channels=map_channels).to(device)
    optimizer = optim.Adam(actor_critic_net.parameters(), lr=LEARNING_RATE)
    actor_critic_net.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    
    # Initialisation des buffers de frames et d'états globaux pour chaque drone
    frame_buffers = {drone: deque(maxlen=n_frames_stack) for drone in map_training.drones}
    global_state_buffer = {drone: deque(maxlen=n_frames_stack) for drone in map_training.drones}
    
    rewards_per_episode = []
    DroneDestroyed = []
    
    for episode in range(NB_EPISODES):
        playground.reset()
        # Réinitialisation pour chaque drone
        for drone in map_training.drones:
            drone.grid = OccupancyGrid(size_area_world=drone.size_area,
                                        resolution=grid_resolution,
                                        lidar=drone.lidar())
            if drone in frame_buffers:
                frame_buffers[drone].clear()
            else:
                frame_buffers[drone] = deque(maxlen=n_frames_stack)
            dummy_maps = torch.zeros((1, map_channels, drone.grid.grid.shape[0], drone.grid.grid.shape[1]), device=device)
            for _ in range(n_frames_stack):
                frame_buffers[drone].append(dummy_maps)
            
            if drone in global_state_buffer:
                global_state_buffer[drone].clear()
            else:
                global_state_buffer[drone] = deque(maxlen=n_frames_stack)
            dummy_state = torch.zeros((6), device=device)
            for _ in range(n_frames_stack):
                global_state_buffer[drone].append(dummy_state)
                
        # Listes de collecte pour cet épisode
        states_map = []
        states_vector = []
        actions = []
        rewards = []
        values = []
        logprobs = []
        step = 0
        actions_drones = {drone: drone.process_actions([0, 0, 0]) for drone in map_training.drones}
        total_reward = 0
        done = False
        
        while not done and step < MAX_STEPS and all(drone.drone_health > 0 for drone in map_training.drones):
            step += 1
            # Période d'exploration avant de commencer l'apprentissage
            if step < 70:
                i = random.random() if step < 50 else 0
                for drone in map_training.drones:
                    drone.timestep_count = step
                    actions_drones = {drone: drone.process_actions([0, 0, i]) for drone in map_training.drones}
                    drone.update_map_pose_speed()
                playground.step(actions_drones)
            else:
                # Sélection d'action avec frame skip
                if step % n_frame_skip == 0:
                    for drone in map_training.drones:
                        drone.timestep_count = step
                        if map_channels == 2:
                            current_maps = torch.from_numpy(np.stack((drone.grid.grid, drone.grid.position_grid), axis=0)).unsqueeze(0)
                        elif map_channels == 1:
                            current_maps = torch.from_numpy(drone.grid.grid).unsqueeze(0).unsqueeze(0)
                        current_maps = current_maps.float().to(device)
                        current_global_state = drone.process_state_before_network(
                            drone.estimated_pose.position[0],
                            drone.estimated_pose.position[1],
                            drone.estimated_pose.orientation,
                            drone.estimated_pose.vitesse_X,
                            drone.estimated_pose.vitesse_Y,
                            drone.estimated_pose.vitesse_angulaire
                        ).unsqueeze(0)
                        
                        # Obtenir la valeur actuelle à partir du réseau
                        with torch.no_grad():
                            _, _, value_current = actor_critic_net(current_maps, current_global_state)
                        values.append(value_current)
                        
                        # Mise à jour des buffers
                        frame_buffers[drone].append(current_maps)
                        global_state_buffer[drone].append(current_global_state)
                        stacked_frames = torch.cat(list(frame_buffers[drone]), dim=1)
                        stacked_global_states = torch.cat(list(global_state_buffer[drone])).unsqueeze(0)
                        
                        states_map.append(stacked_frames)
                        states_vector.append(stacked_global_states)
                        
                        action, logprob = select_action(actor_critic_net, stacked_frames, stacked_global_states)
                        actions_drones[drone] = drone.process_actions(action)
                        actions.append(action)
                        logprobs.append(logprob)
                playground.step(actions_drones)
                
                if step % n_frame_skip == 0:
                    for drone in map_training.drones:
                        drone.update_map_pose_speed()
                        found_wounded = drone.process_semantic_sensor()
                        min_dist = drone.process_lidar_sensor(drone.lidar())
                        is_collision = min_dist < 10
                        reward = compute_reward(drone, is_collision, found_wounded, 1, actions_drones[drone])
                        rewards.append(reward)
                        total_reward += reward
                        if found_wounded:
                            print("found wounded!")
            
        if any(drone.drone_health <= 0 for drone in map_training.drones):
            DroneDestroyed.append(1)
            map_training = M1()
            playground = map_training.construct_playground(drone_type=MyDroneHulk)
        else:
            DroneDestroyed.append(0)
        
        # Bootstrap pour la dernière valeur
        with torch.no_grad():
            next_state_map = states_map[-1]
            next_state_vector = states_vector[-1].squeeze().unsqueeze(0)
            _, _, next_value = actor_critic_net(next_state_map, next_state_vector)
        
        # Calcul du GAE et des returns
        advantages = torch.zeros(len(rewards), device=device)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards)-1:
                nextnonterminal = 1.0 - done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0
                nextvalues = values[t+1]
            delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + GAMMA * 0.95 * nextnonterminal * lastgaelam
        returns = advantages + torch.tensor(values, device=device)
        
        # Optimisation sur plusieurs epochs
        for epoch in range(NUM_EPOCHS):
            dataset = DroneDataset(states_map, states_vector, actions, returns, advantages, logprobs, values)
            data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
            for batch in data_loader:
                states_map_b, states_vector_b, actions_b, return_b, advantages_b, logprobs_b, values_b = batch
                optimize_batch(states_map_b, states_vector_b, actions_b, return_b, advantages_b, logprobs_b, values_b, optimizer, actor_critic_net, device)
        
        rewards_per_episode.append(total_reward)
        del states_map, states_vector, actions, rewards
        
        if episode % 10 == 1:
            print(f"Episode {episode}, Reward: {total_reward}, Mean Last 10 Rewards: {np.mean(rewards_per_episode[-10:])}")
        if np.mean(rewards_per_episode[-10:]) > 10000:
            print(f"Training solved in {episode} episodes!")
            break
    
    # Sauvegarde des losses et des rewards
    losses_csv_path = os.path.join(folder_name, "losses.csv")
    with open(losses_csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Step", "PolicyLoss", "ValueLoss", "EntropyLoss", "OutboundLoss", "WeightsPolicyLoss", "WeightsValueLoss", "ExplorationLoss"])
        for i in range(len(LossPolicy)):
            csv_writer.writerow([
                i,
                LossPolicy[i],
                LossValue[i],
                LossEntropy[i],
                LossOutbound[i] if LossOutbound else 0,
                LossWeightsPolicy[i] if LossWeightsPolicy else 0,
                LossWeightsValue[i] if LossWeightsValue else 0,
                LossExploration[i] if LossExploration else 0,
            ])
    
    rewards_csv_path = os.path.join(folder_name, "rewards_per_episode.csv")
    with open(rewards_csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["", "TotalReward"])
        for i, reward in enumerate(rewards_per_episode):
            csv_writer.writerow([i, reward])
    
    drone_destroyed_csv_path = os.path.join(folder_name, "drone_destroyed.csv")
    with open(drone_destroyed_csv_path, "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Step", "DroneDestroyed"])
        for i, drone_destroyed in enumerate(DroneDestroyed):
            csv_writer.writerow([i, drone_destroyed])
    
    torch.save(actor_critic_net.state_dict(), os.path.join(folder_name, "actor_critic_net.pth"))
    print("Training complete. Files saved in:", folder_name)

# =============================================================================
# Point d'entrée principal
# =============================================================================
if __name__ == "__main__":
    train()
