from rltoolbox import RLComponent
from typing import Dict

class ConsoleLogger(RLComponent):
    def __init__(self, config: Dict):
        super().__init__(config)

    def episode_end(self, context: Dict):
        episode = context["training"]["episode"]
        length = context["training"]["episode_length"]
        reward = context["training"]["episode_reward"]
        print(f"Episode {episode:4d} | Length: {length:4d} | Reward: {reward:7.2f}")
