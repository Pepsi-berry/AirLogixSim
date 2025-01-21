from typing import Optional

import gymnasium as gym


class SingleAgentEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0, False, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
