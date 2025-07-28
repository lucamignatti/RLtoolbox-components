from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.api.config.reward_function import RewardFunction
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rlgym.rocket_league import common_values
import numpy as np
from rltoolbox import RLComponent
from typing import Dict
import torch

class BallProximityReward(RewardFunction):
    """Reward function that gives reward based on proximity to the ball."""

    def __init__(self, max_distance=5000):
        """
        Initialize ball proximity reward.

        Args:
            max_distance: Maximum distance for reward calculation (default: 5000)
        """
        super().__init__()
        self.max_distance = max_distance

    def reset(self, agents, initial_state, shared_info):
        """Reset the reward function."""
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        """Calculate reward based on distance to ball for each agent."""
        rewards = {}

        # Get ball position (already a numpy array)
        ball_pos = state.ball.position

        for agent in agents:
            # Get player data for this agent
            player = state.cars[agent]

            # Get player position (already a numpy array)
            player_pos = player.physics.position

            # Calculate distance
            distance = np.linalg.norm(player_pos - ball_pos)

            # Normalize distance and invert (closer = higher reward)
            normalized_distance = min(distance / self.max_distance, 1.0)
            reward = 1.0 - normalized_distance

            rewards[agent] = reward

        return rewards

class RLGymEnvironment(RLComponent):
    def __init__(self, config: Dict):
        """
        Initialize RLGym environment with configuration.

        Args:
            config: Dictionary containing environment configuration
                - team_size: Number of players per team (default: 1)
                - spawn_opponents: Whether to spawn opponents (default: True)
                - action_repeat: Number of times to repeat each action (default: 8)
                - no_touch_timeout_seconds: Timeout if ball not touched (default: 30)
                - game_timeout_seconds: Maximum game length (default: 300)
                - goal_reward: Reward for scoring a goal (default: 10)
                - touch_reward: Reward for touching the ball (default: 0.1)
                - ball_proximity_reward: Reward multiplier for being close to ball (default: 0.01)
        """
        super().__init__(config)

        # Extract configuration parameters
        self.team_size = config.get('team_size', 1)
        self.spawn_opponents = config.get('spawn_opponents', True)
        self.action_repeat = config.get('action_repeat', 8)
        self.no_touch_timeout_seconds = config.get('no_touch_timeout_seconds', 30)
        self.game_timeout_seconds = config.get('game_timeout_seconds', 300)
        self.goal_reward = config.get('goal_reward', 10)
        self.touch_reward = config.get('touch_reward', 0.1)
        self.ball_proximity_reward = config.get('ball_proximity_reward', 0.01)

        # Build the environment
        self.env = self._build_rlgym_environment()

        # Get spaces from the first agent since RLGym requires agent parameter
        obs_dict = self.env.reset()
        agents = list(obs_dict.keys())
        first_agent = agents[0]

        # Store the observation and action spaces
        obs_space_info = self.env.observation_space(first_agent)
        action_space_info = self.env.action_space(first_agent)

        # Create simplified spaces for single agent interface
        if obs_space_info[0] == 'real':
            self.observation_space = ('real', obs_space_info[1])
        else:
            self.observation_space = obs_space_info

        if action_space_info[0] == 'discrete':
            self.action_space = ('discrete', action_space_info[1])
        else:
            self.action_space = action_space_info

        # Reset again to ensure clean state
        self._last_obs = self.env.reset()

    def _build_rlgym_environment(self):
        """Build and configure the RLGym environment."""

        # Team configuration
        blue_team_size = self.team_size
        orange_team_size = self.team_size if self.spawn_opponents else 0

        # Action parser with repetition
        action_parser = RepeatAction(LookupTableAction(), repeats=self.action_repeat)

        # Termination condition (game ends on goal)
        termination_condition = GoalCondition()

        # Truncation conditions (timeout conditions)
        truncation_condition = AnyCondition(
            NoTouchTimeoutCondition(timeout_seconds=self.no_touch_timeout_seconds),
            TimeoutCondition(timeout_seconds=self.game_timeout_seconds)
        )

        # Reward function (goals + ball touches + ball proximity)
        reward_fn = CombinedReward(
            (GoalReward(), self.goal_reward),
            (TouchReward(), self.touch_reward),
            (BallProximityReward(), self.ball_proximity_reward)
        )

        # Observation builder with normalized values
        obs_builder = DefaultObs(
            zero_padding=None,
            pos_coef=np.asarray([
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z
            ]),
            ang_coef=1 / np.pi,
            lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
            ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
            boost_coef=1 / 100.0,
        )

        # State mutator for team sizes and kickoffs
        state_mutator = MutatorSequence(
            FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
            KickoffMutator()
        )

        # Create RLGym environment
        rlgym_env = RLGym(
            state_mutator=state_mutator,
            obs_builder=obs_builder,
            action_parser=action_parser,
            reward_fn=reward_fn,
            termination_cond=termination_condition,
            truncation_cond=truncation_condition,
            transition_engine=RocketSimEngine()
        )

        # Return RLGym environment directly
        return rlgym_env

    def reset(self):
        """Reset the environment and return initial observation."""
        obs_dict = self.env.reset()
        # Store obs for step method
        self._last_obs = obs_dict

        # Always use first agent's observation for single agent interface
        agents = list(obs_dict.keys())
        obs = obs_dict[agents[0]]
        return obs, {}

    def step(self, action):
        """Take a step in the environment."""
        # RLGym expects actions for each agent
        obs_dict = getattr(self, '_last_obs', None)
        if obs_dict is None:
            # If we don't have last obs, reset first
            obs_dict = self.env.reset()

        # Create action dict for all agents
        agents = list(obs_dict.keys())
        if isinstance(action, dict):
            action_dict = action
        else:
            # Single action for first agent, others get default action
            # Convert to numpy array if it's an integer
            if isinstance(action, (int, np.integer)):
                action = np.array([action])
            elif not isinstance(action, np.ndarray):
                action = np.array([action])
            else:
                # Ensure action is 1D
                action = action.reshape(-1)
                if len(action) == 1:
                    action = action
                else:
                    # If it's a multi-dimensional action, take the first element
                    action = np.array([action[0]])

            action_dict = {agents[0]: action}
            for agent in agents[1:]:
                action_dict[agent] = np.array([0])

        result = self.env.step(action_dict)
        obs_dict, reward_dict, done_dict, info_dict = result

        # Store obs for next step
        self._last_obs = obs_dict

        # Use first agent's values for single agent interface
        first_agent = agents[0]
        obs = obs_dict[first_agent]
        reward = reward_dict[first_agent]
        done = done_dict[first_agent]
        info = info_dict[first_agent] if isinstance(info_dict, dict) else {}

        # Convert done to terminated/truncated format
        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info

    def close(self):
        """Close the environment."""
        self.env.close()

    def episode_reset(self, context: Dict):
        """Reset for a new episode and update context."""
        obs_dict = self.env.reset()
        # Store obs for step method
        self._last_obs = obs_dict

        # Use first agent's observation for single agent interface
        agents = list(obs_dict.keys())
        obs = obs_dict[agents[0]]
        context['state'] = obs
        context['info'] = {}

    def environment_step(self, context: Dict):
        """Take an environment step and update context."""
        action = context['action']

        # Convert torch tensor to numpy if needed
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        # Take step in environment using our wrapper
        obs, reward, terminated, truncated, info = self.step(action)

        # Update context
        context['next_state'] = obs
        context['reward'] = reward
        context['done'] = terminated or truncated
        context['info'] = info

    def get_action_space(self):
        """Get the action space of the environment."""
        return self.action_space

    def get_observation_space(self):
        """Get the observation space of the environment."""
        return self.observation_space
