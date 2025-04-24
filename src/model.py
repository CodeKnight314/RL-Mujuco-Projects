import torch 
import torch.nn as nn
import numpy as np

def init_weights(module, gain=np.sqrt(2)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.constant_(module.bias, 0.0)
    return module

class ActorProbabilistic(nn.Module): 
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int=256):
        super().__init__()
        
        self.net = nn.Sequential(*[
            init_weights(nn.Linear(input_dim, hidden_dim)), 
            nn.ReLU(inplace=True), 
            init_weights(nn.Linear(hidden_dim, hidden_dim)), 
            nn.ReLU(inplace=True),
        ])
        
        self.mean = init_weights(nn.Linear(hidden_dim, output_dim))
        
        self.log_std = nn.Parameter(torch.ones(output_dim) * -0.5)
        
    def forward(self, x: torch.Tensor):
        output = self.net(x)
        
        mean = self.mean(output)
        
        std = torch.exp(torch.clamp(self.log_std, -5, 2))
        return mean, std
    
    def sample(self, x: torch.Tensor):
        mean, std = self.forward(x)
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        
        log_probs = dist.log_prob(action).sum(dim=-1)
        
        return action, log_probs
    
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