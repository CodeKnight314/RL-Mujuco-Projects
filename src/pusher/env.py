import gymnasium as gym
from ..model import ActorDeterministic, ActorProbabilistic, Critic
from ..replay import ReplayBuffer
import yaml
import torch
import os
from tqdm import tqdm
from torch.nn import functional as F
import numpy as np 

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
            self.ac_opt = torch.optim.AdamW(self.actor.parameters(), self.config["actor_lr"])
            self.target_actor = ActorDeterministic(self.obs, self.acs).to(self.device)
        elif mode == "SAC":
            self.actor = ActorProbabilistic(self.obs, self.acs).to(self.device)
            self.ac_opt = torch.optim.AdamW(self.actor.parameters(), self.config["actor_lr"])
            self.target_actor = ActorProbabilistic(self.obs, self.acs).to(self.device)
            
        self.critic_1 = Critic(self.obs + self.acs).to(self.device)
        self.critic_2 = Critic(self.obs + self.acs).to(self.device)
        self.c1_opt = torch.optim.AdamW(self.critic_1.parameters(), self.config["critic_lr"])
        self.c2_opt = torch.optim.AdamW(self.critic_2.parameters(), self.config["critic_lr"])
        self.tc_1 = Critic(self.obs + self.acs).to(self.device)
        self.tc_2 = Critic(self.obs + self.acs).to(self.device)
        
        self.buffer = ReplayBuffer(self.config["memory"])
        
        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        
        if weights: 
            try:
                self.actor.load(os.path.join(weights, "actor.pth"))
                self.critic_1.load(os.path.join(weights, "critic_1.pth"))
                self.critic_2.load(os.path.join(weights, "critic_2.pth"))
                
            except Exception as e: 
                print(f"[ERROR] Could not load one or more of the provided weights. Training from scratch")
            
        self.update_target(True)
        
    def update_target(self, hard_update: bool = True, tau: float = 0.05):
        if hard_update: 
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.tc_1.load_state_dict(self.critic_1.state_dict())
            self.tc_2.load_state_dict(self.critic_2.state_dict())
        else: 
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            for target_param, param in zip(self.tc_1.parameters(), self.critic_1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
            for target_param, param in zip(self.tc_2.parameters(), self.critic_2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
    def td3_critic_update(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor):
        with torch.no_grad(): 
            noise = torch.normal(mean=0.0, std=self.config["td3_exploration"], out=actions.shape)
            noise = torch.clamp(noise, self.config["acs_min"], self.config["acs_max"])
            
            next_actions = (self.target_actor(states) + noise).clamp(self.config["acs_min"], self.config["acs_max"])
            
            state_action = torch.cat([next_states, next_actions], dim=-1)
            target = rewards + self.config["gamma"] * (1 - dones) * (torch.min(self.tc_1(state_action), self.tc_2(state_action)))
            
        current_q_c1 = self.critic_1(torch.cat([states, actions], dim=-1))
        current_q_c2 = self.critic_2(torch.cat([states, actions], dim=-1))
        
        q_c1_loss = F.mse_loss(current_q_c1, target)
        q_c2_loss = F.mse_loss(current_q_c2, target)
        total_loss = q_c1_loss + q_c2_loss
        
        self.c1_opt.zero_grad()
        self.c2_opt.zero_grad()
        total_loss.backward()
        self.c1_opt.step()
        self.c2_opt.step()
        
        return total_loss.item()
    
    def td3_actor_update(self, states: torch.Tensor):
        actor_loss = -self.critic_1(torch.cat([states, self.actor(states)], dim=-1)).mean()
        
        self.ac_opt.zero_grad()
        actor_loss.backward()
        self.ac_opt.step()
        
        return actor_loss.item()
    
    def sac_critic_update(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor):
        with torch.no_grad():
            next_actions, log_prob = self.target_actor.sample(states)
            target = rewards + self.config["gamma"] * (1 - dones) * (
                torch.min(
                    self.tc_1(torch.cat([next_states, next_actions], dim=-1)) + self.alpha * log_prob.unsqueeze(1),
                    self.tc_2(torch.cat([next_states, next_actions], dim=-1)) + self.alpha * log_prob.unsqueeze(1)
                )
            )
            
        c1_loss = F.mse_loss(self.critic_1(torch.cat([states, actions], dim=-1)), target)
        c2_loss = F.mse_loss(self.critic_2(torch.cat([states, actions], dim=-1)), target)
        total_loss = c1_loss + c2_loss 
        
        self.c1_opt.zero_grad()
        self.c2_opt.zero_grad()
        total_loss.backward()
        self.c1_opt.step()
        self.c2_opt.step()
        
        return total_loss.item()
    
    def sac_actor_update(self, states: torch.Tensor):
        actions, log_prob = self.actor.sample(states)
        
        Q1 = self.critic_1(torch.cat([states, actions], dim=-1))
        Q2 = self.critic_2(torch.cat([states, actions], dim=-1))
        Q = torch.min(Q1, Q2)
        
        actor_loss = (self.alpha * log_prob.unsqueeze(1) - Q).mean()
        self.ac_opt.zero_grad()
        actor_loss.backward() 
        self.ac_opt.step()
        
        alpha_loss = -(self.log_alpha * (log_prob + -self.acs).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return actor_loss.item()
                
    def train(self, path: str):
        os.makedirs(path, exist_ok=True)
            
        pbar = tqdm(range(self.config["episodes"]), desc="Episodes: ")
        avg_reward = [] 
        avg_ac_loss = []
        avg_cr_loss = []
        
        for eps in pbar:
            state, _ = self.env.reset()
            done = False 
            total_reward = 0.0
            steps = 0
            
            while not done:
                if self.mode == "TD3": 
                    action = self.target_actor(torch.tensor(state).float().to(self.device))
                    noise = torch.normal(mean=0.0, std=self.config["td3_exploration"], out=action.shape)
                    action = (action + noise).clamp(self.acs_min, self.acs_max)
                else: 
                    action = self.target_actor.sample(torch.tensor(state).float().to(self.device))
                
                next_state, reward, terminated, truncated, info = self.env.step(action.cpu().detach().numpy())
                done = terminated or truncated
                
                self.buffer.push((state, action, reward, next_state, done))
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if len(self.buffer) > self.config["batch_size"]:
                    states, actions, rewards, next_states, dones = self.buffer.sample(self.config["batch_size"])
                    
                    if self.mode == "TD3": 
                        critic_loss = self.td3_critic_update(states, actions, rewards, next_states, dones)
                        avg_cr_loss.append(critic_loss)
                        
                        if steps % self.config["actor_update_freq"] == 0: 
                            actor_loss = self.td3_actor_update(states)
                            avg_ac_loss.append(actor_loss)
                    else: 
                        critic_loss = self.sac_critic_update(states, actions, rewards, next_states, dones)
                        avg_cr_loss.append(critic_loss)
                        
                        actor_loss = self.sac_actor_update(states)
                        avg_ac_loss.append(actor_loss)
                        
                if steps % self.config["target_update_freq"] == 0: 
                    self.update_target(False)
            
            avg_reward.append(total_reward)
            eps_ac_loss = sum(avg_ac_loss) / len(avg_ac_loss)
            eps_cr_loss = sum(avg_cr_loss) / len(avg_cr_loss)
            eps_reward = sum(avg_reward) / len(avg_reward)
            pbar.set_postfix(reward=eps_reward, Actorloss=eps_ac_loss, Criticloss=eps_cr_loss)
            
            os.system('cls' if os.name == "nt" else 'clear')
            
        self.actor.save(os.path.join(path, "actor.pth"))
        self.critic_1.save(os.path.join(path, "critic_1.pth"))
        self.critic_2.save(os.path.join(path, "critic_2.pth"))
              
    def test(self, path: str): 
        os.makedirs(path, exist_ok=True)
        pass
                
if __name__ == "__main__":
    pusher = PusherEnv("src/pusher/config.yaml", "huh")
    