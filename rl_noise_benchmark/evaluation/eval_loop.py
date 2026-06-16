"""Evaluation routine: run a (greedy) agent for N episodes and summarize return.

Everything here is computed from real rollouts. The agent is queried with a
``predict``/``act`` style interface; any object exposing ``act(obs, epsilon=0)``
works, which includes :class:`rl_noise_benchmark.agents.DQNAgent`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import gymnasium as gym
import numpy as np


class Policy(Protocol):
    """Anything that can pick a greedy action for an observation."""

    def act(self, obs, epsilon: float = ...) -> int:
        ...


@dataclass(frozen=True)
class EvalResult:
    """Aggregated evaluation statistics over a set of episodes."""

    n_episodes: int
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    returns: list[float]

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(
    agent: Policy,
    env: gym.Env,
    n_episodes: int = 20,
    *,
    seed: int | None = None,
) -> EvalResult:
    """Run ``n_episodes`` greedy episodes and return aggregate statistics.

    A distinct (deterministic) seed is used per episode when ``seed`` is given,
    so the evaluation is reproducible without every episode being identical.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")

    returns: list[float] = []
    for ep in range(n_episodes):
        reset_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=reset_seed)
        obs = np.asarray(obs, dtype=np.float32)
        done = False
        ep_return = 0.0
        while not done:
            action = agent.act(obs, epsilon=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = np.asarray(obs, dtype=np.float32)
            ep_return += float(reward)
            done = bool(terminated or truncated)
        returns.append(ep_return)

    arr = np.asarray(returns, dtype=np.float64)
    return EvalResult(
        n_episodes=n_episodes,
        mean_return=float(arr.mean()),
        std_return=float(arr.std()),
        min_return=float(arr.min()),
        max_return=float(arr.max()),
        returns=returns,
    )
