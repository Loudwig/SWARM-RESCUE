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


class MyDroneHulk(DroneAbstract):
    class State(Enum):
        """
        All the states of the drone as a state machine
        """
        EXPLORING = 1
        FINISHED_EXPLORING = 2

    def __init__(self,identifier: Optional[int] = None,misc_data: Optional[MiscData] = None,**kwargs):
        super().__init__(identifier=identifier,misc_data=misc_data,**kwargs)
        
        
        
        self.policy_net = policy_net
        self.value_net = value_net
        
        
        
        # MAPING
        self.estimated_pose = Pose() # Fonctionne commant sans le GPS ?  erreur ou qu'est ce que cela retourne ? 
        resolution = 8 # pourquoi ?  Ok bon compromis entre précision et temps de calcul
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())
        

        self.display_map = True # Display the probability map during the simulation


        # POSTION 
        # deque remplit de 0 

        self.previous_position = deque(maxlen=1) 
        self.previous_position.append((0,0))  
        self.previous_orientation = deque(maxlen=1) 
        self.previous_orientation.append(0) 

        # Initialisation du state
        self.state  = self.State.EXPLORING
        self.previous_state = self.State.EXPLORING # Utile pour vérfier que c'est la première fois que l'on rentre dans un état

        # paramètres logs
        self.record_log = False
        self.log_file = "logs/log.txt"
        self.log_initialized = False
        self.flush_interval = 50  # Number of timesteps before flushing buffer
        self.timestep_count = 0  # Counter to track timesteps        
      
    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def control(self):
        print("Control")
        self.timestep_count +=1 
        self.update_map_pose_speed()
        maps = torch.tensor([self.grid.grid, self.grid.position_grid], dtype=torch.float32, device=device).unsqueeze(0)
        global_state = torch.tensor([self.estimated_pose.position[0], self.estimated_pose.position[1], self.estimated_pose.orientation, self.estimated_pose.vitesse_X, self.estimated_pose.vitesse_Y, self.estimated_pose.vitesse_angulaire], dtype=torch.float32, device=device).unsqueeze(0)
        action,_ = select_action(self.policy_net,maps,global_state)
        return process_actions(action)
    def process_semantic_sensor(self):
        semantic_values = self.semantic_values()
        found_wounded = False
        for data in semantic_values:
            if (data.entity_type ==
                    DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped):
                found_wounded = True
                
        return found_wounded
    
    def process_lidar_sensor(self,self_lidar):
        
        lidar_values = self_lidar.get_sensor_values()

        if lidar_values is None:
            return 0
        
        ray_angles = self_lidar.ray_angles
        size = self_lidar.resolution

        angle_nearest_obstacle = 0
        if size != 0:
            min_dist = min(lidar_values)

        return min_dist
    
    def compute_reward(self,is_collision, found_wounded, time_penalty,action):
        forward = action["forward"]
        lateral = action["lateral"]
        rotation = action["rotation"]

        
        
        
        
        reward = 0

        # Penalize collisions heavily


        if is_collision:
            reward -= 50


        if found_wounded:
            reward += 100

        reward += self.grid.exploration_score

        # Penalize idling or lack of movement
        reward -= time_penalty

        # give penalty when action are initialy not between -0.9 and 0.9 
        if (abs(forward) > 0.95 or abs(lateral) > 0.95 or abs(rotation) > 0.95):
            reward -= 5
        else : 
            print("actions in range")
        return reward


    def state_update(self,found_wounded):
        
        self.previous_state = self.state
        #print(f"Previous state : {self.previous_state}")
        if self.state is self.State.EXPLORING and (found_wounded):
            self.state = self.State.FINISHED_EXPLORING
        
        #print(f"State : {self.state}")
    
    def update_map_pose_speed(self):
        
        if self.timestep_count == 1: # first iteration
            # print("Starting control")
            start_x, start_y = self.measured_gps_position() # never none ? 
            # print(f"Initial position: {start_x}, {start_y}")
            self.grid.set_initial_cell(start_x, start_y)
        

        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                   self.measured_compass_angle(),self.odometer_values(),self.previous_position[-1],self.previous_orientation[-1],self.size_area)
        
        self.previous_position.append(self.estimated_pose.position)
        self.previous_orientation.append(self.estimated_pose.orientation)
        
        self.grid.update_grid(pose=self.estimated_pose)
        
    def showMaps(self,display_zoomed_position_grid = False,display_zoomed_grid = False):
        if  (self.timestep_count % 5 == 0):
            
            if display_zoomed_position_grid:
                self.grid.display(self.grid.zoomed_position_grid,
                                self.estimated_pose,
                                title="zoomed position grid")
            
            if display_zoomed_grid:
                self.grid.display(self.grid.zoomed_grid ,
                                    self.estimated_pose,
                                    title="zoomed occupancy grid")

    # Use this function only at one place in the control method. Not handled othewise.
    # params : variables_to_log : dict of variables to log with keys as variable names and values as variable values.
    def logging_variables(self, variables_to_log):
        """
        Buffers and logs variables to the log file when the buffer reaches the flush interval.

        :param variables_to_log: dict of variables to log with keys as variable names 
                                and values as variable values.
        """
        if not self.record_log:
            return

        # Initialize the log buffer if not already done
        if not hasattr(self, "log_buffer"):
            self.log_buffer = []

        # Append the current variables to the buffer
        log_entry = {"Timestep": self.timestep_count, **variables_to_log}
        self.log_buffer.append(log_entry)

        # Write the buffer to file when it reaches the flush interval
        if len(self.log_buffer) >= self.flush_interval:
            mode = "w" if not self.log_initialized else "a"
            with open(self.log_file, mode) as log_file:
                # Write the header if not initialized
                if not self.log_initialized:
                    headers = ",".join(log_entry.keys())
                    log_file.write(headers + "\n")
                    self.log_initialized = True

                # Write buffered entries
                for entry in self.log_buffer:
                    line = ",".join(map(str, entry.values()))
                    log_file.write(line + "\n")

            # Clear the buffer
            self.log_buffer.clear()
        
    def draw_top_layer(self):
        pass



GAMMA = 0.99
LEARNING_RATE = 1e-4
ENTROPY_BETA = 0.1
NB_EPISODES = 600
MAX_STEPS = 300


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

def process_actions(actions):
    return {
        "forward": actions[0],
        "lateral": actions[1],
        "rotation": actions[2],
        "grasper": 0  # 0 or 1 for grasping
    }

def select_action(policy, state_map,state_vector):
    state_map = torch.FloatTensor(state_map).to(device)
    state_vector = torch.FloatTensor(state_vector).to(device)
    means, log_stds = policy(state_map,state_vector)

    # Sample continuous actions
    stds = torch.exp(log_stds)
    sampled_continuous_actions = means + torch.randn_like(means) * stds

    # Clamp continuous actions to valid range
    # print(f"actions before tanh: {sampled_continuous_actions}")
    continuous_actions = torch.tanh(sampled_continuous_actions)
    # print(f"actions after tanh: {continuous_actions}")
    
    # Compute log probabilities for continuous actions
    # log prob with tanh squashing
    log_probs_continuous = -0.5 * (((sampled_continuous_actions - means) / (stds + 1e-8)) ** 2 + 2 * log_stds + math.log(2 * math.pi))
    log_probs_continuous = log_probs_continuous.sum(dim=1)

    log_probs_continuous = log_probs_continuous - torch.sum(torch.log(1 - continuous_actions ** 2 + 1e-6), dim=1)


    
    

    # Combine actions
    action = continuous_actions[0]

    log_prob = log_probs_continuous 
    return action.detach().cpu(), log_prob

def train():
    print("Training...")
    map_training = M1()
    playground = map_training.construct_playground(drone_type=MyDroneHulk)
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


                action, _ = select_action(drone.policy_net,maps, global_state)
                actions_drones[drone] = process_actions(action)
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
            print(f"Episode {episode}, Reward: {total_reward}, Mean Last 100 Rewards: {np.mean(rewards_per_episode[-5:])}")
            torch.save(policy_net.state_dict(), 'policy_net.pth')
            torch.save(value_net.state_dict(), 'value_net.pth')
            

        if np.mean(rewards_per_episode[-5:]) > 10000:
            print(f"Training solved in {episode} episodes!")
            break


if __name__ == "__main__":
    train()