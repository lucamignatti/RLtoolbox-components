from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from rltoolbox import RLComponent
import numpy as np

class MLPADC(nn.Module, RLComponent):
    """
    MLP Actor-Distributional Critic network for DPPO.
    Features a standard actor and a categorical distributional critic (C51-style).
    """
    def __init__(self, config: Dict):
        super(MLPADC, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_dim = config['input_dim']
        actor_output_dim = config['output_dim']
        hidden_dims = config['hidden_dims']

        # Distributional critic parameters
        self.v_min = config.get('v_min', -10.0)
        self.v_max = config.get('v_max', 10.0)
        self.num_atoms = config.get('num_atoms', 51)
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=self.device)
        self.advantage_calculation_method = config.get('advantage_calculation_method', 'quantile_sampling')

        # Shared feature network
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.shared_network = nn.Sequential(*layers).to(self.device)

        # Actor head (policy network)
        self.actor = nn.Linear(hidden_dims[-1], actor_output_dim).to(self.device)

        # Distributional critic head (outputs logits for categorical distribution)
        self.critic = nn.Linear(hidden_dims[-1], self.num_atoms).to(self.device)

    def forward(self, x):
        """Standard forward pass for action selection during rollout."""
        x = x.to(self.device)
        features = self.shared_network(x)

        # Actor output
        logits = self.actor(features)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_probs = distribution.log_prob(action)

        # Critic output (sample value based on method)
        critic_logits = self.critic(features)
        sampled_value = self._sample_value_from_distribution(critic_logits)

        return action, log_probs, sampled_value

    def _sample_value_from_distribution(self, critic_logits):
        """Sample value from categorical distribution based on configured method."""
        with torch.no_grad():
            if self.advantage_calculation_method == "quantile_sampling":
                # Sample from the categorical distribution
                dist = torch.distributions.Categorical(logits=critic_logits)
                sampled_indices = dist.sample()  # Shape [batch_size]
                support_device = self.support.to(critic_logits.device)
                return support_device[sampled_indices]  # Shape [batch_size]
            else:  # mean
                # Calculate the expected value (mean) from the distribution
                probs = F.softmax(critic_logits, dim=-1)
                support_device = self.support.to(probs.device)
                return torch.sum(probs * support_device.unsqueeze(0), dim=-1)  # Shape [batch_size]

    def action_selection(self, context: Dict):
        """Hook method for action selection during rollout."""
        with torch.no_grad():
            state_tensor = torch.tensor(context["state"], dtype=torch.float32, device=self.device)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)

            features = self.shared_network(state_tensor)

            # Actor forward pass
            logits = self.actor(features)
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            log_probs = distribution.log_prob(action)

            # Critic forward pass (sample value for GAE)
            critic_logits = self.critic(features)
            sampled_value = self._sample_value_from_distribution(critic_logits)

        context['log_probs'] = log_probs.squeeze(0) if log_probs.dim() > 0 else log_probs
        context['state_value'] = sampled_value.squeeze(0) if sampled_value.dim() > 0 else sampled_value
        context['action'] = action.squeeze(0) if action.dim() > 0 else action

    def forward_actor(self, x):
        """Forward pass through actor only."""
        x = x.to(self.device)
        features = self.shared_network(x)
        return self.actor(features)

    def forward_critic(self, x):
        """Forward pass through critic only (returns logits for categorical distribution)."""
        x = x.to(self.device)
        features = self.shared_network(x)
        return self.critic(features)

    def forward_features(self, x):
        """Forward pass to get shared features."""
        x = x.to(self.device)
        return self.shared_network(x)

    def get_mean_value(self, critic_logits):
        """Get mean value from categorical distribution (for logging/debugging)."""
        with torch.no_grad():
            probs = F.softmax(critic_logits, dim=-1)
            support_device = self.support.to(probs.device)
            return (probs * support_device).sum(dim=-1)

    def get_distribution_stats(self, critic_logits):
        """Get statistics from the value distribution (for monitoring)."""
        with torch.no_grad():
            probs = F.softmax(critic_logits, dim=-1)
            support_device = self.support.to(probs.device)

            # Mean
            mean_value = (probs * support_device).sum(dim=-1)

            # Variance
            mean_squared = (probs * support_device.pow(2)).sum(dim=-1)
            variance = mean_squared - mean_value.pow(2)

            # Entropy
            log_probs = F.log_softmax(critic_logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)

            return {
                'mean': mean_value,
                'variance': variance,
                'entropy': entropy
            }

    def evaluate_actions(self, states, actions):
        """Evaluate actions for PPO update (returns logits for distributional critic)."""
        states = states.to(self.device)
        actions = actions.to(self.device)

        # Get shared features
        features = self.shared_network(states)

        # Actor evaluation
        actor_logits = self.actor(features)
        distribution = torch.distributions.Categorical(logits=actor_logits)
        new_log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # Critic evaluation (return raw logits for distributional loss)
        critic_logits = self.critic(features)

        return new_log_probs, critic_logits, entropy

    def get_action_probabilities(self, states):
        """Get action probabilities from actor."""
        states = states.to(self.device)
        features = self.shared_network(states)
        logits = self.actor(features)
        return F.softmax(logits, dim=-1)

    def get_value_distribution(self, states):
        """Get value distribution probabilities from critic."""
        states = states.to(self.device)
        features = self.shared_network(states)
        critic_logits = self.critic(features)
        return F.softmax(critic_logits, dim=-1)
