from rltoolbox import RLComponent
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

class GraphLogger(RLComponent):
    def __init__(self, config: Dict):
        super().__init__(config)

    def moving_average(self, data, window_size=100):
        if len(data) < window_size:
            return np.array(data)
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def training_end(self, context: Dict):
        metrics = context["metrics"]
        episode_rewards = metrics["episode_rewards"]
        episode_lengths = metrics["episode_lengths"]
        losses = metrics.get("losses", [])
        policy_losses = metrics.get("policy_loss", [])
        value_losses = metrics.get("value_loss", [])

        episodes = np.arange(1, len(episode_rewards) + 1)

        # Smoothing
        window = 100  # You can adjust this window size
        rewards_smooth = self.moving_average(episode_rewards, window)
        lengths_smooth = self.moving_average(episode_lengths, window)
        smooth_episodes = np.arange(window, len(episode_rewards) + 1)

        n_plots = 2
        if losses or policy_losses or value_losses:
            n_plots += 1

        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4*n_plots))
        if n_plots == 1:
            axes = [axes]

        # Plot smoothed episode rewards
        axes[0].plot(smooth_episodes, rewards_smooth, 'b-', label=f'Moving Avg (window={window})')
        axes[0].set_title('Episode Rewards (Smoothed)')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].grid(True)
        axes[0].legend()

        # Plot smoothed episode lengths
        axes[1].plot(smooth_episodes, lengths_smooth, 'g-', label=f'Moving Avg (window={window})')
        axes[1].set_title('Episode Lengths (Smoothed)')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Length')
        axes[1].grid(True)
        axes[1].legend()

        # Plot losses if available
        if n_plots > 2:
            loss_axes = axes[2]
            if losses:
                loss_episodes = list(range(1, len(losses) + 1))
                loss_axes.plot(loss_episodes, losses, 'r-', label='Total Loss')
            if policy_losses:
                policy_loss_episodes = list(range(1, len(policy_losses) + 1))
                loss_axes.plot(policy_loss_episodes, policy_losses, 'b-', label='Policy Loss')
            if value_losses:
                value_loss_episodes = list(range(1, len(value_losses) + 1))
                loss_axes.plot(value_loss_episodes, value_losses, 'g-', label='Value Loss')
            loss_axes.set_title('Training Losses')
            loss_axes.set_xlabel('Update Step')
            loss_axes.set_ylabel('Loss')
            loss_axes.legend()
            loss_axes.grid(True)

        plt.tight_layout()
        plt.show()
