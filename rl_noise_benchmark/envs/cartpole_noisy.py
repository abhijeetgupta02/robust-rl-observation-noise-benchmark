"""CartPole environment with additive observation noise.

Wraps Gymnasium's ``CartPole-v1`` and injects noise into every observation
returned by ``reset`` and ``step``. The underlying dynamics and reward are
unchanged -- only what the agent *observes* is corrupted, which is the setting
this benchmark targets.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from ..noise.gaussian import GaussianObservationNoise


class NoisyObservationWrapper(gym.ObservationWrapper):
    """Add a noise process to every observation of ``env``.

    The observation space is inherited from the base env. Noise can push values
    outside the nominal bounds; that is intentional and not clipped, since the
    point of the benchmark is to study agents under corrupted sensing.
    """

    def __init__(self, env: gym.Env, noise: GaussianObservationNoise) -> None:
        super().__init__(env)
        self.noise = noise

    def observation(self, observation: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.noise(np.asarray(observation))

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.floating], dict[str, Any]]:
        # Keep noise reproducible alongside the env: if a seed is given, derive
        # a distinct-but-deterministic noise seed so noise and dynamics streams
        # do not share state yet both stay reproducible.
        if seed is not None:
            self.noise.reset(seed=seed + 1)
        return super().reset(seed=seed, options=options)


def make_noisy_cartpole(
    sigma: float = 0.1,
    *,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Build a ``CartPole-v1`` env wrapped with Gaussian observation noise.

    Parameters
    ----------
    sigma:
        Standard deviation of the Gaussian observation noise.
    seed:
        Seed for the noise process (the env itself is seeded on ``reset``).
    max_episode_steps:
        Optional override of the default 500-step CartPole horizon.
    render_mode:
        Passed through to ``gym.make`` (e.g. ``"human"``); ``None`` for headless.
    """
    base = gym.make(
        "CartPole-v1",
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )
    noise = GaussianObservationNoise(sigma=sigma, seed=seed)
    return NoisyObservationWrapper(base, noise)


def make_cartpole(
    *,
    max_episode_steps: int | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Build a clean (noise-free) ``CartPole-v1`` env, used for training."""
    return gym.make(
        "CartPole-v1",
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )
