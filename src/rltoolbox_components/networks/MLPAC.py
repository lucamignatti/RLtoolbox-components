from typing import Dict
import torch
import torch.nn as nn
from rltoolbox import RLComponent
import numpy as np

class MLPAC(nn.Module, RLComponent):
    def __init__(self, config: Dict):
        super(MLPAC, self).__init__()

        input_dim = config['input_dim']
        output_dim = config['output_dim']
        hidden_dims = config['hidden_dims']

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.network = nn.Sequential(*layers)

        self.actor = nn.Linear(hidden_dims[-1], output_dim)
        self.critic = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        x = self.network(x)
        return self.actor(x), self.critic(x)

    def action_selection(self, context: Dict):
        x = self.network(torch.tensor(context["state"]))
        action = self.actor(x).detach().numpy()
        action = np.argmax(action)
        context['action'] = action

    def forward_actor(self, x):
        x = self.network(x)
        return self.actor(x)

    def forward_critic(self, x):
        x = self.network(x)
        return self.critic(x)
