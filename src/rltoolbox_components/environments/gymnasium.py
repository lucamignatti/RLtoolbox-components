import gymnasium as gym
import numpy as np
from rltoolbox import RLComponent
from typing import Dict
import torch

class GymnasiumEnvironment(RLComponent):
    def __init__(self, config: Dict):
        self.env = gym.make(config['env_name'])
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()

    def episode_reset(self, context: Dict):
        obs, info = self.env.reset()
        context['state'] = obs
        context['info'] = info

    def environment_step(self, context: Dict):
        action = context['action']
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        obs, reward, terminated, truncated, info = self.env.step(action)
        context['next_state'] = obs
        context['reward'] = reward
        context['done'] = terminated or truncated
        context['info'] = info
