from rltoolbox import RLComponent
from typing import Dict
import torch
import torch.nn as nn
import numpy as np

class RolloutBuffer:
    def __init__(self, gamma, device: torch.device = None, update_frequency: int = 1):
        self.device = device
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []
        self.returns = []
        self.advantages = []

        self.gamma = gamma
        self.update_frequency = update_frequency

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.dones.clear()
        self.returns.clear()
        self.advantages.clear()

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
        states_tensor = torch.stack(self.states)
        actions_tensor = torch.stack(self.actions)
        logprobs_tensor = torch.stack(self.logprobs)
        rewards_tensor = torch.stack(self.rewards)
        state_values_tensor = torch.stack(self.state_values)
        dones_tensor = torch.stack(self.dones)

        self.compute_returns_and_advantages(last_value)

        return {
            "actions": actions_tensor.cpu().numpy(),
            "states": states_tensor.cpu().numpy(),
            "logprobs": logprobs_tensor.cpu().numpy(),
            "rewards": rewards_tensor.cpu().numpy(),
            "state_values": state_values_tensor.cpu().numpy(),
            "dones": dones_tensor.cpu().numpy(),
            "returns": np.array(self.returns),
            "advantages": np.array(self.advantages)
        }

    def __len__(self):
        return len(self.states)

    def compute_returns_and_advantages(self, last_value):

        next_return = 0.0 if self.dones[-1] else last_value.detach().cpu().numpy()

        buffer_size = len(self.rewards)

        self.returns = []
        self.advantages = []

        for i in reversed(range(buffer_size)):
            self.returns.append((self.rewards[i] + self.gamma * next_return * (1 - int(self.dones[i]))).cpu().numpy())

            self.advantages.append((self.returns[buffer_size-i-1] - self.state_values[i].cpu().numpy()))

            next_return = self.returns[buffer_size-i-1]

        self.advantages = (self.advantages - np.mean(self.advantages)) / (np.std(self.advantages) + 1e-8)

        self.advantages = self.advantages.tolist()

        self.returns = self.returns[::-1]
        self.advantages = self.advantages[::-1]
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

        self.rollout_buffer = RolloutBuffer(device =self._device, gamma=self.gamma, update_frequency=config.get("update_frequency", 1))

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

        states = torch.tensor(buffer["states"], dtype=torch.float32, device=self._device)
        actions = torch.tensor(buffer["actions"], dtype=torch.float32, device=self._device)
        logprobs = torch.tensor(buffer["logprobs"], dtype=torch.float32, device = self._device)
        returns = torch.tensor(buffer["returns"], dtype=torch.float32, device=self._device)
        advantages = torch.tensor(buffer["advantages"], dtype=torch.float32, device=self._device)

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
        self.rollout_buffer.clear()

    def episode_end(self, context: Dict):
        model = context["components"][self.actor_critic]
        last_state = context["next_state"]
        if self._should_update():
            self.train(model, last_state)

    def _should_update(self):
        return self.rollout_buffer.is_full()

    def experience_storage(self, context: Dict):
        self.rollout_buffer.add(
            state=torch.tensor(context["state"], dtype=torch.float32, device=self._device),
            action=torch.tensor(context["action"], dtype=torch.float32, device=self._device),
            logprob=torch.tensor(context["log_probs"], dtype=torch.float32, device=self._device),
            reward=context["reward"],
            state_value=torch.tensor(context["state_value"], dtype=torch.float32, device=self._device),
            done=context["done"]
        )
