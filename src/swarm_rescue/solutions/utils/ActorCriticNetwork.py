import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        
        # Si vous empilez plusieurs frames, le nombre de canaux d'entrée sera multiplié
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

        # Calculer la taille du flattening après le CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, h, w)
            dummy_output = self.cnn(dummy_input)
            cnn_flatten_size = dummy_output.numel()
        
        # Projection des features CNN vers une dimension souhaitée
        self.fc_cnn = nn.Linear(cnn_flatten_size, cnn_output_dim)

        # On combine la sortie CNN avec l'état global.
        # Ici, on considère que l'état global doit être empilé s'il y a plusieurs frames (frame_stack)
        shared_dim = cnn_output_dim + global_state_dim * frame_stack

        # Tête Policy
        self.policy_mlp = nn.Sequential(
            nn.Linear(shared_dim, hidden_size),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_size, num_actions)
        self.log_sigma_head = nn.Linear(hidden_size, num_actions)

        # Tête Value
        self.value_mlp = nn.Sequential(
            nn.Linear(shared_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, map, global_state):
        # Passage dans le CNN partagé
        x = self.cnn(map)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc_cnn(x))
        
        # Concaténer les features CNN avec l'état global
        x_combined = torch.cat((x, global_state), dim=1)
        
        # Branche policy
        policy_features = self.policy_mlp(x_combined)
        mu = self.mu_head(policy_features)
        log_sigma = self.log_sigma_head(policy_features)
        
        # Branche value
        value = self.value_mlp(x_combined)
        
        return mu, log_sigma, value
