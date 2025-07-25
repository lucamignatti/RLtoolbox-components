from rltoolbox import RLComponent
from typing import Dict
import torch
import torch.nn as nn
import numpy as np

class RolloutBuffer:
    def __init__(self, gamma, gae_lambda, device: torch.device = None, update_frequency: int = 1):
        self.device = device
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []
        self.returns = []
        self.advantages = []

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.update_frequency = update_frequency

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.dones.clear()
        self.returns = []
        self.advantages = []

    def add(self, state: torch.Tensor, action: torch.Tensor, logprob: torch.Tensor, reward: float, state_value: torch.Tensor, done: bool):
        if self.device is None:
            self.device = state.device

        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(torch.tensor(reward, device=self.device))
        self.state_values.append(state_value)
        self.dones.append(torch.tensor(done, device=self.device))

    def get(self, last_value):
        states_tensor = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions_tensor = torch.stack(self.actions)
        logprobs_tensor = torch.stack(self.logprobs)
        rewards_tensor = torch.stack(self.rewards)
        state_values_tensor = torch.stack(self.state_values)
        dones_tensor = torch.stack(self.dones)

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
        last_value = last_value.detach().item()
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



class PPO(RLComponent):
    def __init__(self, config: Dict):
        super().__init__(config)

        self._config = config

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.lr = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_ratio = config.get("clip_ratio", 0.2)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.num_epochs = config.get("num_epochs", 10)
        self.batch_size = config.get("batch_size", 64)

        self.rollout_buffer = RolloutBuffer(device =self._device, gamma=self.gamma, gae_lambda=self.gae_lambda, update_frequency=config.get("update_frequency", 1))

        self.actor_critic = config.get("actor_critic", None)

        if self.actor_critic is None:
            raise ValueError("Actor-Critic model must be provided in the configuration.")

        self.mseloss = nn.MSELoss()

        self.initialized = False
        self.optimizer = None

    def train(self, model: nn.Module, last_state):

        if not self.initialized:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            self.initialized = True

        last_value = model.forward_critic(torch.tensor(last_state, dtype=torch.float32, device= self._device))
        buffer = self.rollout_buffer.get(last_value)

        states = buffer["states"]
        actions = buffer["actions"]
        logprobs = buffer["logprobs"]
        returns = buffer["returns"]
        advantages = buffer["advantages"]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_updates = 0

        for _ in range(self.num_epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]

                batch_logprobs = logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                new_logprobs, new_values, entropy = model.evaluate_actions(batch_states, batch_actions)

                ratio = torch.exp(new_logprobs - batch_logprobs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = self.mseloss(new_values.squeeze(-1), batch_returns)

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_updates += 1

        self.rollout_buffer.clear()

        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates

        return avg_policy_loss, avg_value_loss

    def episode_end(self, context: Dict):
        model = context["components"][self.actor_critic]
        last_state = context["next_state"]
        if self._should_update():
            policy_loss, value_loss = self.train(model, last_state)
            if 'policy_loss' not in context['metrics']:
                context['metrics']['policy_loss'] = []
            context['metrics']['policy_loss'].append(policy_loss)
            if 'value_loss' not in context['metrics']:
                context['metrics']['value_loss'] = []
            context['metrics']['value_loss'].append(value_loss)

    def _should_update(self):
        return self.rollout_buffer.is_full()

    def experience_storage(self, context: Dict):
        self.rollout_buffer.add(
            state=context["state"],
            action=context["action"],
            logprob=context["log_probs"],
            reward=context["reward"],
            state_value=context["state_value"],
            done=context["done"]
        )
