from src.base_env import BaseEnv
import gymnasium as gym
import os
from tqdm import tqdm 
import torch
import numpy as np
from collections import deque

class BaseMultiEnv(BaseEnv):
    def __init__(self, config: str, weights: str, mode: str = "TD3", env_name: str = "Ant-v5", num_envs: int = 8):
        super().__init__(config, weights, mode, env_name)
        
        if not isinstance(num_envs, int): 
            raise TypeError(f"[ERROR] number of environments given is not an integer")
        if num_envs <= 0: 
            raise ValueError(f"[ERROR] number of environments must be non-negative")
        
        self.envs = gym.vector.AsyncVectorEnv(
            [lambda: gym.make(env_name) for i in range(num_envs)], 
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
        )
        
        self.num_envs = num_envs
        
    def train(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.warmup()
        print(f"[INFO] Warmup completed. Buffer size: {len(self.buffer)}")
        
        global_step = 0
        
        avg_ac_loss = [] 
        avg_cr_loss = []
        avg_q1_value = []
        avg_q2_value = []
        
        episode_rewards = [0.0] * self.num_envs
        completed_episode_rewards = deque(maxlen=10)
        episode_count = 0 
        max_avg_reward = 1.0
        
        states, _ = self.envs.reset()
        episode_starts = np.zeros(self.num_envs, dtype=bool)

        pbar = tqdm(total=self.config["episodes"], desc="Episodes: ")
        while episode_count < self.config["episodes"]:            
            state_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
            
            if self.mode == "TD3":
                actions = self.actor(state_tensor)
                noise = (torch.randn_like(actions) * self.td3_exploration).clamp(-self.td3_noise_clip, self.td3_noise_clip)
                actions = (actions + noise).clamp(self.acs_min, self.acs_max)
            else: 
                actions, _ = self.actor.sample(state_tensor)
                
            next_states, rewards, terminateds, truncateds, _ = self.envs.step(actions.cpu().detach().numpy())
            
            for i in range(self.num_envs):
                if not episode_starts[i]:
                    self.buffer.push(
                        states[i], 
                        actions[i],
                        rewards[i],
                        next_states[i], 
                        terminateds[i] or truncateds[i]
                    )
                    
                    episode_rewards[i] += rewards[i]
                
                if terminateds[i] or truncateds[i]: 
                    completed_episode_rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0.0
                    episode_count += 1
                    pbar.update(1)
                
            if len(self.buffer) > self.config["batch_size"]:
                batch = self.buffer.sample(self.config["batch_size"])
                states_b, actions_b, rewards_b, next_states_b, dones_b = [tensor.to(self.device).detach() for tensor in batch]
                
                if self.mode == "TD3":
                    critic_loss, q1_mean, q2_mean = self.td3_critic_update(states_b, actions_b, rewards_b, next_states_b, dones_b)
                    avg_cr_loss.append(critic_loss)
                    avg_q1_value.append(q1_mean)
                    avg_q2_value.append(q2_mean)
                    
                    if global_step % self.config["actor_update_freq"] == 0: 
                        actor_loss = self.td3_actor_update(states_b)
                        avg_ac_loss.append(actor_loss)
                else: 
                    critic_loss, q1_mean, q2_mean = self.sac_critic_update(states_b, actions_b, rewards_b, next_states_b, dones_b)
                    avg_cr_loss.append(critic_loss)
                    avg_q1_value.append(q1_mean)
                    avg_q2_value.append(q2_mean)
                    
                    if global_step % self.config["actor_update_freq"] == 0:
                        actor_loss = self.sac_actor_update(states_b)
                        avg_ac_loss.append(actor_loss)
                    
                if global_step % self.config["target_update_freq"] == 0: 
                    self.update_target(False)

            episode_starts = np.logical_or(terminateds, truncateds)
            
            eps_reward = sum(completed_episode_rewards) / len(completed_episode_rewards) if len(completed_episode_rewards) > 0 else 0
            eps_last = completed_episode_rewards[-1] if len(completed_episode_rewards) > 0 else 0.0
            eps_ac_loss = sum(avg_ac_loss) / len(avg_ac_loss) if len(avg_ac_loss) > 0 else 0
            eps_cr_loss = sum(avg_cr_loss) / len(avg_cr_loss) if len(avg_cr_loss) > 0 else 0
            eps_q1_value = avg_q1_value[-1] if len(avg_q1_value) > 0 else 0
            eps_q2_value = avg_q2_value[-1] if len(avg_q2_value) > 0 else 0
            
            self.update_noise(episode_count)
            
            denom = max_avg_reward if max_avg_reward != 0 else 1.0
            percentage_increase = (eps_reward - max_avg_reward) / denom * 100            
            if percentage_increase > 5:
                self.actor.save(os.path.join(path, "actor.pth"))
                self.critic_1.save(os.path.join(path, "critic_1.pth"))
                self.critic_2.save(os.path.join(path, "critic_2.pth"))
                
                max_avg_reward = eps_reward
            
            if self.mode == "TD3":
                self.update_noise(episode_count)
                pbar.set_postfix(
                    reward=f"{eps_reward:.4f}", 
                    raw=f"{eps_last:.1f}",
                    Actorloss=f"{eps_ac_loss:.4f}", 
                    Criticloss=f"{eps_cr_loss:.4f}", 
                    Q1=f"{eps_q1_value:.2f}", 
                    Q2=f"{eps_q2_value:.2f}",
                    noise=f"{self.td3_exploration:.4f}",
                )
            else: 
                pbar.set_postfix(
                    reward=f"{eps_reward:.4f}", 
                    raw=f"{eps_last:.1f}",
                    Actorloss=f"{eps_ac_loss:.4f}", 
                    Criticloss=f"{eps_cr_loss:.4f}", 
                    Q1=f"{eps_q1_value:.2f}", 
                    Q2=f"{eps_q2_value:.2f}",
                    alpha=f"{self.alpha:.4f}"
                )

            states = next_states
            global_step += 1
            
        self.actor.save(os.path.join(path, "actor.pth"))
        self.critic_1.save(os.path.join(path, "critic_1.pth"))
        self.critic_2.save(os.path.join(path, "critic_2.pth"))
        
        return sum(completed_episode_rewards) / len(completed_episode_rewards)

    def test(self, path: str):
       return super().test(path)