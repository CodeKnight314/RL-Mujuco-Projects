from collections import deque
import random 
import torch

class ReplayBuffer: 
    def __init__(self, max: int = 10000):
        self.memory = deque(maxlen=max)
        
    def push(self, state, action, reward, next_state, done): 
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states]).detach()
        actions = torch.stack(actions).to(actions[0].device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in next_states])
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self): 
        return len(self.memory)
    