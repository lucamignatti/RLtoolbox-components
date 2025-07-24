from typing import Dict
import torch
import torch.nn as nn
from rltoolbox import RLComponent
import numpy as np

class MLPAC(nn.Module, RLComponent):
    def __init__(self, config: Dict):
        super(MLPAC, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_dim = config['input_dim']
        output_dim = config['output_dim']
        hidden_dims = config['hidden_dims']

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.network = nn.Sequential(*layers).to(self.device)

        self.actor = nn.Linear(hidden_dims[-1], output_dim).to(self.device)
        self.critic = nn.Linear(hidden_dims[-1], 1).to(self.device)

    def forward(self, x):
        x.to(self.device)
        x = self.network(x)
        logits = self.actor(x)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_probs = distribution.log_prob(action)
        return action, log_probs, self.critic(x)

    def action_selection(self, context: Dict):
        with torch.no_grad():
            x = self.network(torch.tensor(context["state"], dtype=torch.float32).to(self.device))
            logits = self.actor(x).detach()
            value = self.critic(x).detach()

        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_probs = distribution.log_prob(action)

        context['log_probs'] = log_probs.cpu().numpy()
        context['state_value'] = value.cpu().numpy()
        context['action'] = action.cpu().numpy()

    def forward_actor(self, x):
        x.to(self.device)
        x = self.network(x)
        return self.actor(x)

    def forward_critic(self, x):
        x.to(self.device)
        x = self.network(x)
        return self.critic(x)

    def evaluate_actions(self, states, actions):
        states.to(self.device)
        actions.to(self.device)
        features = self.network(states)
        logits = self.actor(features)
        distribution = torch.distributions.Categorical(logits=logits)
        new_log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        new_values = self.critic(features)

        return new_log_probs, new_values, entropy
