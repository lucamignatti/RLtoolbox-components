from rltoolbox import RLComponent
from typing import Dict
import torch
import torch.nn as nn
import numpy
from stable_baselines3 import PPO

class PPO(RLComponent):
    def __init__(self, config: Dict):
        pass

    def train(self, model: nn.Module):
        pass
    
    def episode_end(self, context: Dict):
        pass