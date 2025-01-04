"""
Le drone suit un path après avoir trouver le wounded person.
"""

from enum import Enum
from collections import deque
import math
from typing import Optional
import cv2
import numpy as np
import arcade
import torch

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

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         **kwargs)
        
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
        
        # increment the iteration counter
        self.timestep_count += 1
        
        # MAPPING is updating the pose also so donc forget to do it at the beginnig
        self.update_map_pose_speed()
        self.showMaps(True,True)
        
        # RECUPÈRATION INFORMATIONS SENSORS (LIDAR, SEMANTIC)
        min_dist = self.process_lidar_sensor(self.lidar())
        found_wounded = self.process_semantic_sensor()

        # paramètres responsables des transitions

        # TRANSITIONS OF THE STATE 
        self.state_update(found_wounded)
        print(f"EXPLORATION REWARD : {self.grid.exploration_score}")
        print(f"Vitesse (X,Y,0) : {self.estimated_pose.vitesse_X}{self.estimated_pose.vitesse_Y}{self.estimated_pose.vitesse_angulaire}")
        

        ##########
        # COMMANDS FOR EACH STATE
        ##########
        command_nothing = {"forward": 0.0,"lateral": 0.0,"rotation": 0.0,"grasper": 0}
        command_tout_droit = {"forward": 0.3,"lateral": 0.0,"rotation": 0.0,"grasper": 0}

        if  self.state is self.State.EXPLORING:
            return command_tout_droit
        
        else :
            return command_nothing

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
    
    def compute_reward(self,is_collision, found_wounded, time_penalty):
        reward = 0

        # Penalize collisions heavily
        if is_collision:
            reward -= 50


        if found_wounded:
            reward += 100

        reward += self.grid.exploration_score

        # Penalize idling or lack of movement
        reward -= time_penalty

        return reward


    def compute_returns(self,rewards):
        returns = []
        g = 0
        for reward in reversed(rewards):
            g = reward + GAMMA * g
            returns.insert(0, g)
        return returns

    def state_update(self,found_wounded):
        
        self.previous_state = self.state
        #print(f"Previous state : {self.previous_state}")
        if self.state is self.State.EXPLORING and (found_wounded):
            self.state = self.State.FINISHED_EXPLORING
        
        #print(f"State : {self.state}")
    
    def update_map_pose_speed(self):
        
        if self.timestep_count == 1: # first iteration
            print("Starting control")
            start_x, start_y = self.measured_gps_position() # never none ? 
            print(f"Initial position: {start_x}, {start_y}")
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
NB_EPISODES = 1200
MAX_STEPS = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



policy_net = NetworkPolicy()  
value_net = NetworkValue()
optimizer_policy = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
optimizer_value = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)
policy_net.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)


def optimize_batch(states, actions, returns, batch_size=8, entropy_beta=0.1):
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    returns = np.array(returns, dtype=np.float32)

    dataset_size = len(states)
    for start_idx in range(0, dataset_size, batch_size):
        end_idx = min(start_idx + batch_size, dataset_size)

        states_batch = torch.tensor(states[start_idx:end_idx], device=device)
        actions_batch = torch.tensor(actions[start_idx:end_idx], device=device)
        returns_batch = torch.tensor(returns[start_idx:end_idx], device=device)

        # Forward pass
        means, log_stds, discrete_logits = policy_net(states_batch)
        values = value_net(states_batch).squeeze()

        # Separate continuous and discrete actions
        continuous_actions_t = actions_batch[:, :3]
        discrete_actions_t = actions_batch[:, 3].long()

        # Continuous action log probabilities
        stds = torch.exp(log_stds)
        log_probs_continuous = -0.5 * (((continuous_actions_t - means) / (stds + 1e-8)) ** 2 + 2 * log_stds + math.log(2 * math.pi))
        log_probs_continuous = log_probs_continuous.sum(dim=1)

        # Discrete action log probabilities
        discrete_action_prob = torch.sigmoid(discrete_logits).squeeze()
        log_probs_discrete = torch.log(discrete_action_prob + 1e-8) * discrete_actions_t + \
                             torch.log(1 - discrete_action_prob + 1e-8) * (1 - discrete_actions_t)

        # Total log probabilities
        log_probs = log_probs_continuous + log_probs_discrete

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
    map_training = M1()
    playground = map_training.construct_playground(drone_type=MyDroneHulk)
    rewards_per_episode = []

    for episode in range(NB_EPISODES):
        
        playground.reset()
        #gui = GuiSR(playground=playground, the_map=map_training, draw_semantic_rays=True)
        done = False
        states, actions, rewards = [], [], []
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
                # maps is the occupation map and the position map as a tensor of shape (1, 2, h, w)
                maps = torch.tensor([drone.grid.grid, drone.grid.position_grid], dtype=torch.float32, device=device).unsqueeze(0)
                global_state = torch.tensor([drone.estimated_pose.position[0], drone.estimated_pose.position[1], drone.estimated_pose.orientation, drone.estimated_pose.vitesse_X, drone.estimated_pose.vitesse_Y, drone.estimated_pose.vitesse_angulaire], dtype=torch.float32, device=device).unsqueeze(0)
                
                state = torch.cat((maps, global_state), dim=1)
                states.append(state)


                action, _ = drone.policy_net.sample(maps, global_state)
                actions_drones[drone] = process_actions(action)
                actions.append(action)

            playground.step(actions_drones)

            for drone in map_training.drones:
                
                drone.update_map_pose_speed() # conséquence de l'action
                
                found_wounded = drone.process_semantic_sensor()
                min_dist = drone.process_lidar_sensor(drone.lidar())
                is_collision = min_dist < 10
                reward = drone.compute_reward(is_collision, found_wounded, 1)
                semantic_data = preprocess_semantic(drone.semantic_values())
                
                
                rewards.append(reward)
                total_reward += reward

                done = found_wounded
                if done:
                    print("found wounded !")

            
            
        if any([drone.drone_health<=0 for drone in map_training.drones]):
            map_training = MyMapIntermediate01()
            playground = map_training.construct_playground(drone_type=MyDrone)
        
        # Optimize the policy and value networks in batches
        returns = compute_returns(rewards)
        optimize_batch(states, actions, returns, batch_size=8)
        rewards_per_episode.append(total_reward)

        del states, actions, rewards

        if episode % 100 == 1:
            print(f"Episode {episode}, Reward: {total_reward}, Mean Last 100 Rewards: {np.mean(rewards_per_episode[-100:])}")
            print(map_training.drones[0].occupancy_grid.visited_zones)
            torch.save(policy_net.state_dict(), 'policy_net.pth')
            torch.save(value_net.state_dict(), 'value_net.pth')
            visualize_occupancy_grid(map_training.drones[0].occupancy_grid.grid, episode, step)
            #cv2.imshow("Occupancy Grid", map_training.drones[0].occupancy_grid.zoomed_grid)
            #cv2.waitKey(1)

        if np.mean(rewards_per_episode[-100:]) > 10000:
            print(f"Training solved in {episode} episodes!")
            break


if __name__ == "__main__":
    train()