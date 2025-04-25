import torch 
import torch.nn as nn
import numpy as np

def init_weights(module, gain=np.sqrt(2)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.constant_(module.bias, 0.0)
    return module

class ActorProbabilistic(nn.Module): 
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int=256, log_std_min: float=-5, log_std_max: float=2, action_range: float=1.0): 
        super().__init__()
        
        self.net = nn.Sequential(*[
            init_weights(nn.Linear(input_dim, hidden_dim)), 
            nn.ReLU(inplace=True), 
            init_weights(nn.Linear(hidden_dim, hidden_dim)), 
            nn.ReLU(inplace=True),
            init_weights(nn.Linear(hidden_dim, 2 * output_dim))
        ])
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_range = action_range
                
    def forward(self, x: torch.Tensor):
        mean, log_std = self.net(x).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean,std

    def sample(self, x: torch.Tensor):
        mu, std = self.forward(x)
        dist = torch.distributions.Normal(mu, std)

        z = dist.rsample()
        action = torch.tanh(z) * self.action_range
        
        log_prob_z = dist.log_prob(z)
        log_prob = log_prob_z - torch.log(1 - torch.tanh(z).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob
    
    def load(self, weights: str): 
        self.load_state_dict(torch.load(weights))
        
    def save(self, path: str):
        torch.save(self.state_dict(), path)
        
class ActorDeterministic(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int=256, multiplier: float=2.0):
        super().__init__()
        
        self.net = nn.Sequential(*[
            init_weights(nn.Linear(input_dim, hidden_dim)), 
            nn.ReLU(inplace=True), 
            init_weights(nn.Linear(hidden_dim, hidden_dim)), 
            nn.ReLU(inplace=True), 
            init_weights(nn.Linear(hidden_dim , output_dim)), 
        ])
        
        self.multiplier = multiplier
        
    def forward(self, x: torch.Tensor):
        return torch.tanh(self.net(x)) * self.multiplier
    
    def load(self, weights: str): 
        self.load_state_dict(torch.load(weights))
        
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
class Critic(nn.Module): 
    def __init__(self, input_dim: int, hidden_dim: int=256): 
        super(Critic, self).__init__()
        self.net = nn.Sequential(*[
            init_weights(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
            init_weights(nn.Linear(hidden_dim, hidden_dim)),  
            nn.ReLU(),
            init_weights(nn.Linear(hidden_dim, 1))
        ])
        
    def forward(self, x):
        return self.net(x)
    
    def load(self, weights: str):
        self.load_state_dict(torch.load(weights))
        
    def save(self, path: str):
        torch.save(self.state_dict(), path)