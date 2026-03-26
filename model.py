import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal initialization: The 'secret sauce' for PPO. 
    It prevents gradients from vanishing/exploding better than standard uniform.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPO_Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        super(PPO_Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # The Neural Network for the Mean (mu)
        self.fc1 = layer_init(nn.Linear(state_size, fc1_units))
        self.fc2 = layer_init(nn.Linear(fc1_units, fc2_units))
        # The final layer uses a smaller std for initialization so actions start near 0
        self.fc_mean = layer_init(nn.Linear(fc2_units, action_size), std=0.01)
        
        # The Standard Deviation (sigma) is a standalone, learnable parameter!
        # We don't pass the state into it. It's just a set of 4 numbers that the network tunes over time.
        self.action_std = nn.Parameter(torch.zeros(1, action_size))

    def forward(self, state, action=None):
        """Outputs a probability distribution, samples an action, and returns its log probability."""
        
        # 1. Calculate the Mean (mu)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_mean = F.tanh(self.fc_mean(x)) # Squashed between -1 and 1
        
        # 2. Calculate the Standard Deviation (sigma)
        # We use softplus to ensure the standard deviation is always a positive number
        action_std = F.softplus(self.action_std)
        
        # 3. Create the Probability Distribution (The Bell Curve)
        dist = Normal(action_mean, action_std)
        
        # If we didn't pass in an action, we are interacting with the environment, so we sample one
        if action is None:
            action = dist.sample()
            
        # 4. Calculate the Log Probability of the action
        # We need this for the PPO ratio math later!
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        
        # 5. Calculate Entropy (how random the distribution is, used to encourage exploration)
        dist_entropy = dist.entropy().sum(dim=-1)
        
        return action, action_log_prob, dist_entropy


class PPO_Critic(nn.Module):
    def __init__(self, state_size, seed, fc1_units=256, fc2_units=128):
        super(PPO_Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # The Critic ONLY takes the state now. No action required!
        self.fc1 = layer_init(nn.Linear(state_size, fc1_units))
        self.fc2 = layer_init(nn.Linear(fc1_units, fc2_units))
        self.fc_value = layer_init(nn.Linear(fc2_units, 1), std=1.0)

    def forward(self, state):
        """Maps a state to a single V(s) value."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc_value(x)
