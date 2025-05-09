import gymnasium as gym
import cv2
from src.model import ActorDeterministic, ActorProbabilistic, Critic
from src.replay import ReplayBuffer
import yaml
import torch
import os
from tqdm import tqdm
from torch.nn import functional as F
import numpy as np

class BaseEnv:
    def __init__(self, config: str, weights: str, mode: str = "TD3", env_name: str = "Ant-v5"):
        with open(config, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.env = gym.make(env_name,
                            render_mode="rgb_array")
        
        self.obs = self.env.observation_space.shape[0]
        self.acs = self.env.action_space.shape[0]
        
        self.acs_max = self.config["ac_max"]
        self.acs_min = self.config["ac_min"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mode = mode
        
        if mode == "TD3": 
            self.actor = ActorDeterministic(self.obs, self.acs, hidden_dim=self.config["actor_hidden_dim"], multiplier=(self.acs_max - self.acs_min)/2.0).to(self.device)
            self.ac_opt = torch.optim.Adam(self.actor.parameters(), self.config["actor_lr"])
            self.target_actor = ActorDeterministic(self.obs, self.acs, hidden_dim=self.config["actor_hidden_dim"], multiplier=(self.acs_max - self.acs_min)/2.0).to(self.device)
        elif mode == "SAC":
            self.actor = ActorProbabilistic(self.obs, self.acs, hidden_dim=self.config["actor_hidden_dim"], action_range=(self.acs_max - self.acs_min)/2.0).to(self.device)
            self.ac_opt = torch.optim.Adam(self.actor.parameters(), self.config["actor_lr"])
            self.target_actor = ActorProbabilistic(self.obs, self.acs, hidden_dim=self.config["actor_hidden_dim"], action_range=(self.acs_max - self.acs_min)/2.0).to(self.device)
            
        self.critic_1 = Critic(self.obs + self.acs, hidden_dim=self.config["critic_hidden_dim"]).to(self.device)
        self.critic_2 = Critic(self.obs + self.acs, hidden_dim=self.config["critic_hidden_dim"]).to(self.device)
        self.c1_opt = torch.optim.Adam(self.critic_1.parameters(), self.config["critic_lr"])
        self.c2_opt = torch.optim.Adam(self.critic_2.parameters(), self.config["critic_lr"])
        self.tc_1 = Critic(self.obs + self.acs, hidden_dim=self.config["critic_hidden_dim"]).to(self.device)
        self.tc_2 = Critic(self.obs + self.acs, hidden_dim=self.config["critic_hidden_dim"]).to(self.device)
        
        self.buffer = ReplayBuffer(self.config["memory"])
        
        if self.mode == "TD3":
            self.td3_exploration = self.config["td3_exploration_start"]
            self.td3_exploration_min = self.config["td3_exploration_min"]
            self.td3_exploration_decay = self.config["td3_exploration_decay"]
            self.td3_noise_clip = self.config["td3_noise_clip"]
        else:
            self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
            self.target_entropy = -float(self.acs)
            
        if weights: 
            try:
                self.actor.load(os.path.join(weights, "actor.pth"))
                self.critic_1.load(os.path.join(weights, "critic_1.pth"))
                self.critic_2.load(os.path.join(weights, "critic_2.pth"))
                
            except Exception as e: 
                print(f"[ERROR] Could not load one or more of the provided weights. Training from scratch")
            
        self.update_target(True)
        
    def update_noise(self, episode: int):
        if self.mode == "TD3": 
            self.td3_exploration = max(self.td3_exploration_min, self.config["td3_exploration_start"] - (episode / self.td3_exploration_decay) * (self.config["td3_exploration_start"] - self.td3_exploration_min))
        else: 
            pass
        
    def warmup(self):
        while len(self.buffer) < self.config["batch_size"] * self.config["replay_buffer"]:
            state, _ = self.env.reset()
            done = False 
            total_reward = 0.0
            
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                action = torch.tensor(action, dtype=torch.float32).to(self.device)
                
                done = terminated or truncated
                                   
                self.buffer.push(
                    state, 
                    action, 
                    reward, 
                    next_state, 
                    done
                )
                
                state = next_state
                total_reward += reward
        
    def update_target(self, hard_update: bool = True, tau: float = 0.005):
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
            next_actions = self.target_actor(next_states.to(self.device))
            noise_scale = 0.2
            noise = (torch.randn_like(next_actions) * noise_scale).clamp(-0.5, 0.5)
            next_actions = torch.clamp(next_actions + noise, self.acs_min, self.acs_max)
            
            state_action = torch.cat([next_states, next_actions], dim=-1)
            target_q1 = self.tc_1(state_action)
            target_q2 = self.tc_2(state_action)
            Q = torch.min(target_q1, target_q2)
            
            target = rewards + self.config["gamma"] * (1 - dones) * Q
            target = target.detach()

        current_q_c1 = self.critic_1(torch.cat([states, actions], dim=-1))
        current_q_c2 = self.critic_2(torch.cat([states, actions], dim=-1))
        q1_mean = current_q_c1.mean().item()
        q2_mean = current_q_c2.mean().item()
        
        c1_loss = F.mse_loss(current_q_c1, target)
        c2_loss = F.mse_loss(current_q_c2, target)
        
        self.c1_opt.zero_grad()
        c1_loss.backward()
        self.c1_opt.step()
        
        self.c2_opt.zero_grad()
        c2_loss.backward()
        self.c2_opt.step()
        
        return c2_loss.item() + c1_loss.item(), q1_mean, q2_mean
    
    def td3_actor_update(self, states: torch.Tensor):
        actions = self.actor(states)
        actor_loss = -self.critic_1(torch.cat([states, actions], dim=-1)).mean()
        
        self.ac_opt.zero_grad()
        actor_loss.backward()
        self.ac_opt.step()
        
        return actor_loss.item()
    
    def sac_critic_update(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor):
        with torch.no_grad():
            next_actions, log_prob = self.actor.sample(next_states)
            target_q1 = self.tc_1(torch.cat([next_states, next_actions], dim=-1))
            target_q2 = self.tc_2(torch.cat([next_states, next_actions], dim=-1))
            min_q = torch.min(target_q1, target_q2)

            soft_q = min_q - self.alpha * log_prob
            target = rewards + self.config["gamma"] * (1 - dones) * soft_q
            
        current_q_c1 = self.critic_1(torch.cat([states, actions], dim=-1))
        current_q_c2 = self.critic_2(torch.cat([states, actions], dim=-1))
        q1_mean = current_q_c1.mean().item()
        q2_mean = current_q_c2.mean().item()
        
        c1_loss = F.smooth_l1_loss(current_q_c1, target)
        c2_loss = F.smooth_l1_loss(current_q_c2, target)
        
        self.c1_opt.zero_grad()
        c1_loss.backward()
        self.c1_opt.step()
        
        self.c2_opt.zero_grad()
        c2_loss.backward()
        self.c2_opt.step()
        
        return c2_loss.item() + c1_loss.item(), q1_mean, q2_mean
    
    def sac_actor_update(self, states: torch.Tensor):
        actions, log_prob = self.actor.sample(states)
        
        log_prob = torch.clamp(log_prob, -self.config["log_prob_clip"], self.config["log_prob_clip"])
        
        Q1 = self.critic_1(torch.cat([states, actions], dim=-1))
        Q2 = self.critic_2(torch.cat([states, actions], dim=-1))
        
        Q = torch.min(Q1, Q2)
        
        alpha = self.alpha.detach()
        actor_loss = (alpha * log_prob.unsqueeze(1) - Q).mean()
        
        self.ac_opt.zero_grad()
        actor_loss.backward()
        self.ac_opt.step()
        
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        return actor_loss.item()
                
    def train(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.warmup()
        print(f"[INFO] Warmup completed. Buffer size: {len(self.buffer)}")
            
        pbar = tqdm(range(self.config["episodes"]), desc="Episodes: ")
        
        from collections import deque
        avg_reward = deque(maxlen=10)
        avg_ac_loss = []
        avg_cr_loss = []
        avg_q1_value = []
        avg_q2_value = []
        
        max_avg_reward = -1e6
        
        for eps in pbar:
            state, _ = self.env.reset()
            done = False 
            total_reward = 0.0
            episode_reward_raw = 0.0
            steps = 0
            
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                
                if self.mode == "TD3": 
                    action = self.actor(state_tensor)
                    noise = (torch.randn_like(action) * self.td3_exploration).clamp(-self.td3_noise_clip, self.td3_noise_clip)
                    action = (action + noise).clamp(self.acs_min, self.acs_max)
                else: 
                    action, _ = self.actor.sample(state_tensor)
            
                next_state, reward, terminated, truncated, _ = self.env.step(action.cpu().detach().numpy())
                episode_reward_raw += reward
                
                done = terminated or truncated
                
                self.buffer.push(
                    state, 
                    action, 
                    reward, 
                    next_state, 
                    done
                )
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if len(self.buffer) > self.config["batch_size"]:
                    states, actions, rewards, next_states, dones = self.buffer.sample(self.config["batch_size"])
                    states = states.to(self.device).detach()
                    actions = actions.to(self.device).detach()
                    rewards = rewards.to(self.device).detach()
                    next_states = next_states.to(self.device).detach()
                    dones = dones.to(self.device).detach()
                    
                    if self.mode == "TD3": 
                        critic_loss, q1_mean, q2_mean = self.td3_critic_update(states, actions, rewards, next_states, dones)
                        avg_cr_loss.append(critic_loss)
                        avg_q1_value.append(q1_mean)
                        avg_q2_value.append(q2_mean)
                        
                        if steps % self.config["actor_update_freq"] == 0: 
                            actor_loss = self.td3_actor_update(states)
                            avg_ac_loss.append(actor_loss)
                    else: 
                        critic_loss, q1_mean, q2_mean = self.sac_critic_update(states, actions, rewards, next_states, dones)
                        avg_cr_loss.append(critic_loss)
                        avg_q1_value.append(q1_mean)
                        avg_q2_value.append(q2_mean)
                        
                        if steps % self.config["actor_update_freq"] == 0:
                            actor_loss = self.sac_actor_update(states)
                            avg_ac_loss.append(actor_loss)
                        
                    if steps % self.config["target_update_freq"] == 0: 
                        self.update_target(False)
            
            avg_reward.append(total_reward)
            
            eps_reward = sum(avg_reward) / len(avg_reward)
            eps_ac_loss = sum(avg_ac_loss) / len(avg_ac_loss) if len(avg_ac_loss) > 0 else 0
            eps_cr_loss = sum(avg_cr_loss) / len(avg_cr_loss) if len(avg_cr_loss) > 0 else 0
            eps_q1_value = avg_q1_value[-1] if len(avg_q1_value) > 0 else 0
            eps_q2_value = avg_q2_value[-1] if len(avg_q2_value) > 0 else 0
            
            if max_avg_reward < eps_reward:
                self.actor.save(os.path.join(path, "actor.pth"))
                self.critic_1.save(os.path.join(path, "critic_1.pth"))
                self.critic_2.save(os.path.join(path, "critic_2.pth"))
                
                max_avg_reward = eps_reward
            
            if self.mode == "TD3":
                self.update_noise(eps)
                pbar.set_postfix(
                    reward=f"{eps_reward:.4f}", 
                    raw=f"{episode_reward_raw:.1f}",
                    Actorloss=f"{eps_ac_loss:.4f}", 
                    Criticloss=f"{eps_cr_loss:.4f}", 
                    Q1=f"{eps_q1_value:.2f}", 
                    Q2=f"{eps_q2_value:.2f}",
                    noise=f"{self.td3_exploration:.4f}",
                )
            else: 
                pbar.set_postfix(
                    reward=f"{eps_reward:.4f}", 
                    raw=f"{episode_reward_raw:.1f}",
                    Actorloss=f"{eps_ac_loss:.4f}", 
                    Criticloss=f"{eps_cr_loss:.4f}", 
                    Q1=f"{eps_q1_value:.2f}", 
                    Q2=f"{eps_q2_value:.2f}",
                    alpha=f"{self.alpha:.4f}"
                )
            
        self.actor.save(os.path.join(path, "actor.pth"))
        self.critic_1.save(os.path.join(path, "critic_1.pth"))
        self.critic_2.save(os.path.join(path, "critic_2.pth"))
        
        return sum(avg_reward) / len(avg_reward)
              
    def test(self, path: str): 
        os.makedirs(path, exist_ok=True)
        video_path = os.path.join(path, "simulation.mp4")
        
        state, _ = self.env.reset()
        done = False
        total_reward = 0.0
        
        frame = self.env.render()
        height, width, _ = frame.shape
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        
        frames = []
        
        step_count = 0
        
        while not done:
            frame = self.env.render()
            frames.append(frame)
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame)
            
            with torch.no_grad():
                if self.mode == "TD3": 
                    action = self.actor(state_tensor)
                else: 
                    action, _ = self.actor.sample(state_tensor)
            
            next_state, reward, terminated, truncated, _ = self.env.step(action.cpu().detach().numpy())
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            step_count += 1
            
        video.release()
        print(f"MP4 video saved at {video_path}")
        print(f"Test completed with total reward: {total_reward}")
        self.env.close()
        return total_reward    