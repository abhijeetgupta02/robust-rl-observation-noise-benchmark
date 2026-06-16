from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from rl_noise_benchmark.noise.gaussian import GaussianObservationNoise


class NoisyCartPoleEnv(gym.Wrapper):
    """CartPole environment with configurable Gaussian observation noise.

    The action/termination/reward dynamics are unchanged; only the observation
    vector returned to the agent is perturbed.
    """

    def __init__(
        self,
        env: Optional[gym.Env] = None,
        sigma: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        if env is None:
            env = gym.make("CartPole-v1")
        super().__init__(env)
        self._rng = np.random.default_rng(seed)
        self._noise = GaussianObservationNoise(sigma=sigma)

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        noisy_obs = self._noise.apply(obs, rng=self._rng)
        return noisy_obs, info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        noisy_obs = self._noise.apply(obs, rng=self._rng)
        return noisy_obs, reward, terminated, truncated, info
