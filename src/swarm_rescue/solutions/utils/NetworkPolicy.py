import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class NetworkPolicy(nn.Module):
    def __init__(self, map_channels= 1,h =100,w = 63,cnn_output_dim = 64,global_state_dim = 6,hidden_size = 32,num_actions = 3,frame_stack=1):
        super(NetworkPolicy, self).__init__()
        
        self.input_channels = map_channels*frame_stack

        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2,stride=2),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2,stride=2),
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2,stride=2),
            nn.Tanh()
        )

        # calculate the output size of the CNN
        with torch.no_grad():
            print(f"test : {h,w}")
            dummy_input = torch.zeros(1, self.input_channels, h, w)
            dummy_output = self.cnn(dummy_input)
            cnn_flatten_size = dummy_output.numel()

        # on projete ce flatten vers un 
        self.fc_cnn = nn.Linear(cnn_flatten_size, cnn_output_dim)

        # MLP final 
        mlp_input_dim = cnn_output_dim + global_state_dim*frame_stack
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_size),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden_size, num_actions)
        self.log_sigma_head = nn.Linear(hidden_size, num_actions)
    
    def forward(self, map, global_state):
        # print(f"map shape: {map.shape}")
        x = self.cnn(map)
        # print(f"x shape aften cnn: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"x shape after view: {x.shape}")
        x = F.tanh(self.fc_cnn(x))
        # print(f"x shape after fc_cnn: {x.shape}")
        # print(f"global_state shape: {global_state.shape}")
        x = torch.cat((x, global_state), dim=1) # On trans
        x = self.mlp(x) 
        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x)
        return mu, log_sigma

    # sample an action from the policy with tanh squashing
    # correction du log_prob termes dans le jacobian

