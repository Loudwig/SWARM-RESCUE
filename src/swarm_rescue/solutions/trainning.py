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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
from solutions.my_drone_A2C import MyDroneHulk

GAMMA = 0.99
LEARNING_RATE = 1e-4
ENTROPY_BETA = 0.1
NB_EPISODES = 600
MAX_STEPS = 100


policy_net = NetworkPolicy()  
value_net = NetworkValue()
optimizer_policy = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
optimizer_value = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)
policy_net.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)


def compute_returns(rewards):
    returns = []
    g = 0
    for reward in reversed(rewards):
        g = reward + GAMMA * g
        returns.insert(0, g)
    return returns

def optimize_batch(states_maps,states_vectors, actions, returns, batch_size=8, entropy_beta=0.1):
    states_maps = np.array(states_maps, dtype=np.float32).squeeze()
    states_vectors = np.array(states_vectors, dtype=np.float32).squeeze()
    actions = np.array(actions, dtype=np.float32)
    returns = np.array(returns, dtype=np.float32)

    dataset_size = len(states_maps)
    for start_idx in range(0, dataset_size, batch_size):
        end_idx = min(start_idx + batch_size, dataset_size)

        states_map_batch = torch.tensor(states_maps[start_idx:end_idx], device=device)
        states_vector_batch = torch.tensor(states_vectors[start_idx:end_idx], device=device)
        actions_batch = torch.tensor(actions[start_idx:end_idx], device=device)
        returns_batch = torch.tensor(returns[start_idx:end_idx], device=device)

        # Forward pass
        means, log_stds = policy_net(states_map_batch, states_vector_batch)
        values = value_net(states_map_batch,states_vector_batch).squeeze()

        # # Separate continuous and discrete actions
        # continuous_actions_t = actions_batch[:, :3]
        # discrete_actions_t = actions_batch[:, 3].long()

        # Continuous action log probabilities
        stds = torch.exp(log_stds)
        log_probs_continuous = -0.5 * (((actions_batch - means) / (stds + 1e-8)) ** 2 + 2 * log_stds + math.log(2 * math.pi))
        log_probs_continuous = log_probs_continuous.sum(dim=1)

        log_probs_continuous = log_probs_continuous - torch.sum(torch.log(1 - actions_batch ** 2 + 1e-6), dim=1)

        # # Discrete action log probabilities
        # discrete_action_prob = torch.sigmoid(discrete_logits).squeeze()
        # log_probs_discrete = torch.log(discrete_action_prob + 1e-8) * discrete_actions_t + \
        #                      torch.log(1 - discrete_action_prob + 1e-8) * (1 - discrete_actions_t)

        # # Total log probabilities
        log_probs = log_probs_continuous 

        # Compute advantages
        advantages = returns_batch - values.detach()

        # Policy loss
        policy_loss = -(log_probs * advantages).mean()

        # Entropy regularization
        entropy_loss = -entropy_beta * (log_stds + 0.5 * math.log(2 * math.pi * math.e)).sum(dim=1).mean()
        total_policy_loss = policy_loss + entropy_loss

        # Value loss
        value_loss = nn.functional.mse_loss(values, returns_batch)

        # Backpropagation
        optimizer_policy.zero_grad()
        total_policy_loss.backward()
        optimizer_policy.step()

        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()



def train():
    print("Training...")
    map_training = M1()
    playground = map_training.construct_playground(drone_type=MyDroneHulk)
    rewards_per_episode = []
    for drone in map_training.drones:
        
        # On écrase les poids potentiellement loader ou bien on initialise les networks
        drone.policy_net = policy_net
        drone.value_net = value_net

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


                action, _ = drone.select_action(maps, global_state)
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
        optimize_batch(states_map,states_vector, actions, returns, batch_size=8)
        rewards_per_episode.append(total_reward)

        del states_map,states_vector, actions, rewards

        if episode % 5 == 1:
            print(f"Episode {episode}, Reward: {total_reward}, Mean Last 5 Rewards: {np.mean(rewards_per_episode[-5:])}")
            torch.save(policy_net.state_dict(), 'src/swarm_rescue/solutions/utils/trained_models/policy_net.pth')
            torch.save(value_net.state_dict(), 'src/swarm_rescue/solutions/utils/trained_models/value_net.pth')
            
        if np.mean(rewards_per_episode[-5:]) > 10000:
            print(f"Training solved in {episode} episodes!")
            break


if __name__ == "__main__":
    train()