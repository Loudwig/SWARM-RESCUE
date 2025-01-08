import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class NetworkValue(nn.Module):
    def __init__(self, map_channels= 2,h =100,w = 63,cnn_output_dim = 64,global_state_dim = 6,hidden_size = 32,num_actions = 3,frame_stack=1):
        super(NetworkValue, self).__init__()
        
        self.input_channels = map_channels * frame_stack
        
        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # calculate the output size of the CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, h, w)
            dummy_output = self.cnn(dummy_input)
            cnn_flatten_size = dummy_output.numel()

        # on projete ce flatten vers un 
        self.fc_cnn = nn.Linear(cnn_flatten_size, cnn_output_dim)

        # MLP final 
        mlp_input_dim = cnn_output_dim + global_state_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    
    def forward(self, map, global_state):
        x = self.cnn(map)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_cnn(x))
        x = torch.cat((x, global_state), dim=1)
        x = self.mlp(x) 
        return x

    
