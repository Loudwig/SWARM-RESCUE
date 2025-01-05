import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class NetworkPolicy(nn.Module):
    def __init__(self, map_channels= 2,h =100,w = 63,cnn_output_dim = 64,global_state_dim = 6,hidden_size = 32,num_actions = 3):
        super(NetworkPolicy, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(map_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # calculate the output size of the CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, map_channels, h, w)
            dummy_output = self.cnn(dummy_input)
            cnn_flatten_size = dummy_output.numel()

        # on projete ce flatten vers un 
        self.fc_cnn = nn.Linear(cnn_flatten_size, cnn_output_dim)

        # MLP final 
        mlp_input_dim = cnn_output_dim + global_state_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_size),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_size, num_actions)
        self.log_sigma_head = nn.Linear(hidden_size, num_actions)
    
    def forward(self, map, global_state):
        x = self.cnn(map)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_cnn(x))
        x = torch.cat((x, global_state), dim=1)
        x = self.mlp(x) 
        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x)
        return mu, log_sigma

    # sample an action from the policy with tanh squashing
    # correction du log_prob termes dans le jacobian

    def sample(self, maps, global_state):
        means, log_stds = self.forward(maps, global_state)
        
        stds = torch.exp(log_stds)
        sampled_continuous_actions = means + torch.randn_like(means) * stds

        # Clamp continuous actions to valid range
        continuous_actions = torch.clamp(sampled_continuous_actions, -1.0, 1.0)

        # Compute log probabilities for continuous actions
        log_probs_continuous = -0.5 * (((sampled_continuous_actions - means) / (stds + 1e-8)) ** 2 + 2 * log_stds + math.log(2 * math.pi))
        log_probs_continuous = log_probs_continuous.sum(dim=1)

        return continuous_actions, log_probs_continuous
