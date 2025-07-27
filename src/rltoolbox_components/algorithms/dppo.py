from rltoolbox import RLComponent
from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import math

class DistributionalRolloutBuffer:
    def __init__(self, gamma, gae_lambda, device: torch.device = None, update_frequency: int = 1,
                 v_min: float = -10.0, v_max: float = 10.0, num_atoms: int = 51,
                 advantage_calculation_method: str = "quantile_sampling"):
        self.device = device
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []  # Stores sampled values for GAE
        self.dones = []
        self.returns = []
        self.advantages = []

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.update_frequency = update_frequency

        # Distributional parameters
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.support = torch.linspace(v_min, v_max, num_atoms, device=device)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.advantage_calculation_method = advantage_calculation_method

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.dones.clear()
        self.returns = []
        self.advantages = []

    def add(self, state: torch.Tensor, action: torch.Tensor, logprob: torch.Tensor,
            reward: float, state_value: torch.Tensor, done: bool):
        if self.device is None:
            self.device = state.device

        self.states.append(state.detach() if isinstance(state, torch.Tensor) else state)
        self.actions.append(action.detach() if isinstance(action, torch.Tensor) else action)
        self.logprobs.append(logprob.detach() if isinstance(logprob, torch.Tensor) else logprob)
        self.rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
        self.state_values.append(state_value.detach() if isinstance(state_value, torch.Tensor) else state_value)  # This should be the sampled value
        self.dones.append(torch.tensor(done, dtype=torch.bool, device=self.device))

    def get(self, last_critic_logits):
        states_tensor = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions_tensor = torch.stack(self.actions)
        logprobs_tensor = torch.stack(self.logprobs)
        rewards_tensor = torch.stack(self.rewards)
        state_values_tensor = torch.stack(self.state_values)
        dones_tensor = torch.stack(self.dones)

        # Calculate last value using the specified method
        with torch.no_grad():
            if self.advantage_calculation_method == "quantile_sampling":
                dist = torch.distributions.Categorical(logits=last_critic_logits)
                sampled_idx = dist.sample()
                last_value = self.support[sampled_idx].item()
            else:  # mean
                probs = F.softmax(last_critic_logits, dim=-1)
                last_value = (probs * self.support).sum().item()

        self.compute_returns_and_advantages(last_value, state_values_tensor)

        return {
            "actions": actions_tensor,
            "states": states_tensor,
            "logprobs": logprobs_tensor,
            "rewards": rewards_tensor,
            "state_values": state_values_tensor,
            "dones": dones_tensor,
            "returns": self.returns,
            "advantages": self.advantages
        }

    def __len__(self):
        return len(self.states)

    def compute_returns_and_advantages(self, last_value, state_values_tensor):
        rewards = torch.stack(self.rewards)
        dones = torch.stack(self.dones)
        state_values = state_values_tensor.squeeze(-1)

        advantages = torch.zeros_like(rewards, dtype=torch.float32)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_values = last_value
            else:
                next_non_terminal = 1.0 - dones[t+1].float()
                next_values = state_values[t+1]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - state_values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

        self.returns = advantages + state_values
        self.advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-8)

    def is_full(self):
        return len(self.states) >= self.update_frequency


class DPPO(RLComponent):
    def __init__(self, config: Dict):
        super().__init__(config)

        self._config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Basic PPO parameters
        self.lr = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_ratio = config.get("clip_ratio", 0.2)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.num_epochs = config.get("num_epochs", 10)
        self.batch_size = config.get("batch_size", 64)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)

        # Distributional critic parameters
        self.v_min = config.get("v_min", -10.0)
        self.v_max = config.get("v_max", 10.0)
        self.num_atoms = config.get("num_atoms", 51)
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=self._device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        # DPPO specific parameters
        self.use_adaptive_epsilon = config.get("use_adaptive_epsilon", True)
        self.adaptive_epsilon_beta = config.get("adaptive_epsilon_beta", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.05)
        self.epsilon_max = config.get("epsilon_max", 0.3)

        self.use_confidence_weighting = config.get("use_confidence_weighting", True)
        self.confidence_weight_type = config.get("confidence_weight_type", "entropy")  # "entropy" or "variance"
        self.confidence_weight_delta = config.get("confidence_weight_delta", 1e-6)
        self.normalize_confidence_weights = config.get("normalize_confidence_weights", False)

        self.advantage_calculation_method = config.get("advantage_calculation_method", "quantile_sampling")

        # Reward normalization (SimbaV2 style)
        self.reward_norm_G_max = config.get("reward_norm_G_max", 10.0)
        self.running_G = 0.0
        self.running_G_mean = 0.0
        self.running_G_var = 1.0
        self.running_G_max = 0.0
        self.running_G_count = 0
        self.reward_norm_epsilon = 1e-8

        # Initialize buffer
        self.rollout_buffer = DistributionalRolloutBuffer(
            device=self._device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            update_frequency=config.get("update_frequency", 1),
            v_min=self.v_min,
            v_max=self.v_max,
            num_atoms=self.num_atoms,
            advantage_calculation_method=self.advantage_calculation_method
        )

        self.actor_critic = config.get("actor_critic", None)
        if self.actor_critic is None:
            raise ValueError("Actor-Critic model must be provided in the configuration.")

        self.mseloss = nn.MSELoss()
        self.initialized = False
        self.optimizer = None

        # Metrics tracking
        self.current_episode_rewards = []
        self.episode_returns = deque(maxlen=100)

    def normalize_reward(self, reward, done):
        """SimbaV2 reward normalization."""
        gamma = self.gamma
        eps = self.reward_norm_epsilon
        G_max = self.reward_norm_G_max

        is_tensor = isinstance(reward, torch.Tensor)
        device = reward.device if is_tensor else self._device

        reward_np = reward.detach().cpu().numpy() if is_tensor else np.array(reward)
        done_np = done.detach().cpu().numpy() if is_tensor else np.array(done)

        was_scalar = reward_np.ndim == 0
        if was_scalar:
            reward_np, done_np = reward_np.reshape(1), done_np.reshape(1)

        normed_rewards = np.empty_like(reward_np)

        for i in range(len(reward_np)):
            r, d = reward_np[i], done_np[i]

            if d:
                self.running_G = 0.0
                self.running_G_max = 0.0

            self.running_G = gamma * self.running_G + r
            self.running_G_count += 1

            delta = self.running_G - self.running_G_mean
            self.running_G_mean += delta / self.running_G_count
            delta2 = self.running_G - self.running_G_mean
            self.running_G_var += delta * delta2

            self.running_G_max = max(self.running_G_max, abs(self.running_G))

            current_variance = self.running_G_var / self.running_G_count if self.running_G_count > 1 else 1.0
            std = np.sqrt(max(0.0, current_variance) + eps)

            denom_max_term = self.running_G_max / G_max if G_max > 0 else 0.0
            denom = max(std, denom_max_term, eps)

            normed_rewards[i] = r / denom

        if was_scalar:
            normed_rewards = normed_rewards.item()

        return torch.tensor(normed_rewards, dtype=torch.float32, device=device) if is_tensor else normed_rewards

    def _compute_confidence_weight(self, predicted_probs):
        """Compute confidence weight (inverse uncertainty)."""
        if not self.use_confidence_weighting:
            return torch.ones(predicted_probs.shape[0], device=self._device)

        if self.confidence_weight_type == "entropy":
            eps = 1e-8
            log_probs = torch.log(predicted_probs + eps)
            entropy = -(predicted_probs * log_probs).sum(dim=1)
            raw_weights = 1.0 / (entropy + self.confidence_weight_delta)
        elif self.confidence_weight_type == "variance":
            support = self.support.to(predicted_probs.device)
            expected_value = (predicted_probs * support).sum(dim=1)
            expected_value_squared = (predicted_probs * support.pow(2)).sum(dim=1)
            variance = F.relu(expected_value_squared - expected_value.pow(2))
            raw_weights = 1.0 / (variance + self.confidence_weight_delta)
        else:
            return torch.ones(predicted_probs.shape[0], device=self._device)

        if self.normalize_confidence_weights:
            weights = raw_weights / (raw_weights.mean() + 1e-8)
        else:
            weights = raw_weights

        return weights.detach()

    def sample_value_from_distribution(self, critic_logits):
        """Sample value from categorical distribution based on method."""
        with torch.no_grad():
            if self.advantage_calculation_method == "quantile_sampling":
                dist = torch.distributions.Categorical(logits=critic_logits)
                sampled_indices = dist.sample()
                support_device = self.support.to(critic_logits.device)
                return support_device[sampled_indices]
            else:  # mean
                probs = F.softmax(critic_logits, dim=-1)
                support_device = self.support.to(probs.device)
                return torch.sum(probs * support_device.unsqueeze(0), dim=-1)

    def train(self, model: nn.Module, last_state):
        if not self.initialized:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            self.initialized = True

        # Get last critic logits for final value calculation
        with torch.no_grad():
            if isinstance(last_state, torch.Tensor):
                last_state_tensor = last_state.detach().clone()
            else:
                last_state_tensor = torch.tensor(last_state, dtype=torch.float32, device=self._device)
            last_critic_logits = model.forward_critic(last_state_tensor)
        buffer = self.rollout_buffer.get(last_critic_logits)

        states = buffer["states"]
        actions = buffer["actions"]
        old_logprobs = buffer["logprobs"]
        returns = buffer["returns"]
        advantages = buffer["advantages"]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0

        for epoch in range(self.num_epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Get current predictions
                new_logprobs, critic_logits, entropy = model.evaluate_actions(batch_states, batch_actions)

                # Convert critic logits to probabilities
                predicted_probs = F.softmax(critic_logits, dim=1)
                predicted_log_probs = F.log_softmax(critic_logits, dim=1)

                # Calculate variance for adaptive epsilon
                if self.use_adaptive_epsilon:
                    support_device = self.support.to(predicted_probs.device)
                    expected_value = (predicted_probs * support_device).sum(dim=1)
                    expected_value_squared = (predicted_probs * support_device.pow(2)).sum(dim=1)
                    batch_variance = F.relu(expected_value_squared - expected_value.pow(2))

                    # Adaptive epsilon: higher variance -> lower epsilon
                    epsilon_t = self.clip_ratio / (1.0 + self.adaptive_epsilon_beta * batch_variance)
                    epsilon_t = torch.clamp(epsilon_t, self.epsilon_min, self.epsilon_max)
                else:
                    epsilon_t = torch.tensor(self.clip_ratio, device=self._device)

                # Calculate confidence weights
                confidence_weights = self._compute_confidence_weight(predicted_probs)

                # PPO policy loss with adaptive clipping and confidence weighting
                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - epsilon_t, 1.0 + epsilon_t) * batch_advantages
                ppo_objectives = torch.min(surr1, surr2)
                weighted_objectives = ppo_objectives * confidence_weights
                policy_loss = -weighted_objectives.mean()

                # Distributional critic loss (KL divergence)
                with torch.no_grad():
                    # Project scalar returns to categorical distribution
                    clamped_returns = batch_returns.clamp(min=self.v_min, max=self.v_max)
                    b = (clamped_returns - self.v_min) / self.delta_z
                    lower_bound_idx = b.floor().long().clamp(0, self.num_atoms - 1)
                    upper_bound_idx = b.ceil().long().clamp(0, self.num_atoms - 1)

                    upper_prob = b - b.floor()
                    lower_prob = 1.0 - upper_prob

                    target_p = torch.zeros_like(predicted_log_probs)
                    target_p.scatter_add_(1, lower_bound_idx.unsqueeze(1), lower_prob.unsqueeze(1))
                    target_p.scatter_add_(1, upper_bound_idx.unsqueeze(1), upper_prob.unsqueeze(1))
                    target_p_stable = target_p + 1e-8

                # KL(target || predicted)
                kl_div = (target_p_stable * (torch.log(target_p_stable) - predicted_log_probs)).sum(dim=1)
                value_loss = kl_div.mean()

                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()

                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                num_updates += 1

        self.rollout_buffer.clear()

        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates

        return avg_policy_loss, avg_value_loss, avg_entropy_loss

    def episode_end(self, context: Dict):
        model = context["components"][self.actor_critic]
        last_state = context["next_state"]

        # Track episode returns
        if "reward" in context:
            self.current_episode_rewards.append(context["reward"])
        if context.get("done", False):
            episode_return = sum(self.current_episode_rewards)
            self.episode_returns.append(episode_return)
            self.current_episode_rewards = []

        if self._should_update():
            policy_loss, value_loss, entropy_loss = self.train(model, last_state)

            if 'policy_loss' not in context['metrics']:
                context['metrics']['policy_loss'] = []
            context['metrics']['policy_loss'].append(policy_loss)

            if 'value_loss' not in context['metrics']:
                context['metrics']['value_loss'] = []
            context['metrics']['value_loss'].append(value_loss)

            if 'entropy_loss' not in context['metrics']:
                context['metrics']['entropy_loss'] = []
            context['metrics']['entropy_loss'].append(entropy_loss)

            if len(self.episode_returns) > 0:
                if 'mean_return' not in context['metrics']:
                    context['metrics']['mean_return'] = []
                context['metrics']['mean_return'].append(np.mean(self.episode_returns))

    def _should_update(self):
        return self.rollout_buffer.is_full()

    def experience_storage(self, context: Dict):
        # Normalize reward
        normalized_reward = self.normalize_reward(context["reward"], context["done"])

        self.rollout_buffer.add(
            state=context["state"],
            action=context["action"],
            logprob=context["log_probs"],
            reward=normalized_reward,
            state_value=context["state_value"],  # This should be the sampled value
            done=context["done"]
        )
