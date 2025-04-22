import gymnasium as gym
from ..model import ActorDeterministic, ActorProbabilistic, Critic
from ..replay import ReplayBuffer
import yaml
import torch
import os
from tqdm import tqdm

class PusherEnv:
    def __init__(self, config: str, weights: str, mode: str = "TD3"):
        with open(config, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.env = gym.make("Pusher-v5")
        
        self.obs = self.env.observation_space.shape[0]
        self.acs = self.env.action_space.shape[0]
        
        self.acs_max = self.config["ac_max"]
        self.acs_min = self.config["ac_min"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mode = mode
        
        if mode == "TD3": 
            self.actor = ActorDeterministic(self.obs, self.acs).to(self.device)
            self.target_actor = ActorDeterministic(self.obs, self.acs).to(self.device)
        elif mode == "SAC":
            self.actor = ActorProbabilistic(self.obs, self.acs).to(self.device)
            self.target_actor = ActorDeterministic(self.obs, self.acs).to(self.device)
            
        self.critic_1 = Critic(self.obs + self.acs).to(self.device)
        self.critic_2 = Critic(self.obs + self.acs).to(self.device)
        self.tc_1 = Critic(self.obs + self.acs).to(self.device)
        self.tc_2 = Critic(self.obs + self.acs).to(self.device)
        
        self.buffer = ReplayBuffer(self.config["memory"])
        
        if weights: 
            self.actor.load_state_dict(os.path.join(weights, "actor.pth"))
            self.critic_1.load_state_dict(os.path.join(weights, "critic_1.pth"))
            self.critic_2.load_state_dict(os.path.join(weights, "critic_2.pth"))
            
        self.update_target(True)
        
    def update_target(self, hard_update: bool = True, tau: float = 0.05):
        if hard_update: 
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.tc_1.load_state_dict(self.actor.state_dict())
            self.tc_2.load_state_dict(self.actor.state_dict())
        else: 
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for target_param, param in zip(self.tc_1.parameters(), self.critic_1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
            for target_param, param in zip(self.tc_2.parameters(), self.critic_2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
    def train(self, path: str):
        os.makedirs(path, exist_ok=True)    
        
        avg_reward = [] 
        avg_ac_loss = []
        avg_cr_loss = []
        
        for eps in tqdm(range(self.config["epsides"])):
            state, _ = self.env.reset()
            done = False 
            total_reward = 0.0
            
            while not done:
                if self.mode == "TD3": 
                    action = self.target_actor(torch.tensor(state).float().to(self.device))
                    noise = torch.normal(mean=0.0, std=self.config["td3_exploration"], out=action.shape)
                    action = (action + noise).clamp(self.acs_min, self.acs_max)
                else: 
                    action = self.target_actor.sample(torch.tensor(state).float().to(self.device))
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                self.buffer.push((state, action, reward, next_state, done))
                
                state = next_state
                
                if len(self.buffer) > self.config["batch_size"]:
                    states, actions, rewards, next_states, dones = self.buffer.sample(self.config["batch_size"])
    
    def test(self, path: str): 
        os.makedirs(path, exist_ok=True)
        pass
                
if __name__ == "__main__":
    pusher = PusherEnv("src/pusher/config.yaml", "huh")
    