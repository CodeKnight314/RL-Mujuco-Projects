from src.base_env import BaseEnv
import gymnasium as gym
import os
from tqdm import tqdm 
import torch
import numpy as np

class BaseMultiEnv(BaseEnv):
    def __init__(self, config: str, weights: str, mode: str = "TD3", env_name: str = "Ant-v5", num_envs: int = 8):
        super().__init__(config, weights, mode, env_name)
        
        if isinstance(num_envs, int): 
            raise TypeError(f"[ERROR] number of environments given is not an integer")
        if num_envs <= 0: 
            raise ValueError(f"[ERROR] number of environments must be non-negative")
        
        def make_env(env_name: str):
            env = gym.make(env_name)
            return env
        
        self.envs = gym.vector.AsyncVectorEnv(
            [make_env(env_name)for i in range(num_envs)],
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
        )
        
        self.num_envs = num_envs
        
    def train(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.warmup()
        print(f"[INFO] Warmup completed. Buffer size: {len(self.buffer)}")

        pbar = tqdm(range(self.config["episodes"]), desc="Episodes: ")
        avg_reward = []
        avg_ac_loss = [] 
        avg_cr_loss = []
        avg_q1_value = []
        avg_q2_value = []
        
        for eps in pbar:
            states, _ = self.envs.reset()
            dones = {i: False for i in range(self.num_envs)}
            total_reward = [0.0 for i in range(self.num_envs)]
            episode_reward_raw = 0.0
            steps = 0
            
            while not all(dones.values()):
                state_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
                
                if self.mode == "TD3":
                    actions = self.actor(state_tensor)
                    noise = (torch.randn_like(actions) * self.td3_exploration).clamp(-self.td3_noise_clip, self.td3_noise_clip)
                    actions = (actions + noise).clamp(self.acs_min, self.acs_max)
                else: 
                    actions, _ = self.actor.sample(state_tensor)
                    

                next_states, rewards, terminateds, truncateds, _ = self.envs.step(actions.cpu().detach().numpy())
                
                for i in range(self.num_envs):
                    if not dones[i]:
                        self.buffer.push(
                            states[i], 
                            actions[i],
                            rewards[i],
                            next_states[i], 
                            dones[i]
                        )
                    
                    total_reward[i] += rewards[i]
                
                states = next_states
                steps += 1
                
                dones = np.logical_or(terminateds, truncateds)

                if len(self.buffer) > self.config["batch_size"]:
                    batch = self.buffer.sample(self.config["batch_size"])
                    states_b, actions_b, rewards_b, next_states_b, dones_b = [tensor.to(self.device) for tensor in batch]
                    
                    if self.mode == "TD3":
                        critic_loss, q1_mean, q2_mean = self.td3_critic_update(states_b, actions_b, rewards_b, next_states_b, dones_b)
                        avg_cr_loss.append(critic_loss)
                        avg_q1_value.append(q1_mean)
                        avg_q2_value.append(q2_mean)
                        
                        if steps % self.config["actor_update_freq"] == 0: 
                            actor_loss = self.td3_actor_update(states_b)
                            avg_ac_loss.append(actor_loss)
                    else: 
                        critic_loss, q1_mean, q2_mean = self.sac_critic_update(states_b, actions_b, rewards_b, next_states_b, dones_b)
                        avg_cr_loss.append(critic_loss)
                        avg_q1_value.append(q1_mean)
                        avg_q2_value.append(q2_mean)
                        
                        if steps % self.config["actor_update_freq"] == 0:
                            actor_loss = self.sac_actor_update(states_b)
                            avg_ac_loss.append(actor_loss)
                        
                    if steps % self.config["target_update_freq"] == 0: 
                        self.update_target(False)
                        
                avg_reward.append(total_reward)
            
            eps_reward = sum(avg_reward) / len(avg_reward)
            eps_ac_loss = sum(avg_ac_loss) / len(avg_ac_loss) if len(avg_ac_loss) > 0 else 0
            eps_cr_loss = sum(avg_cr_loss) / len(avg_cr_loss) if len(avg_cr_loss) > 0 else 0
            eps_q1_value = avg_q1_value[-1] if len(avg_q1_value) > 0 else 0
            eps_q2_value = avg_q2_value[-1] if len(avg_q2_value) > 0 else 0
            
            self.update_noise(eps)
            
            pbar.set_postfix(
                reward=f"{eps_reward:.4f}", 
                raw=f"{episode_reward_raw:.4f}",
                Actorloss=f"{eps_ac_loss:.4f}", 
                Criticloss=f"{eps_cr_loss:.4f}", 
                Q1=f"{eps_q1_value:.2f}", 
                Q2=f"{eps_q2_value:.2f}",
                noise=f"{self.td3_exploration:.4f}",
            )
            
        self.actor.save(os.path.join(path, "actor.pth"))
        self.critic_1.save(os.path.join(path, "critic_1.pth"))
        self.critic_2.save(os.path.join(path, "critic_2.pth"))
        
        return sum(avg_reward) / len(avg_reward)

    def test(self, path: str):
       return super().test(path)