from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import gymnasium as gym
import numpy as np

from rl_noise_benchmark.envs import NoisyCartPoleEnv
from rl_noise_benchmark.agents import DQNCartPoleAgent, DQNConfig


@dataclass
class EvalResult:
    returns: List[float]

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.returns)) if self.returns else 0.0

    @property
    def std_return(self) -> float:
        return float(np.std(self.returns)) if self.returns else 0.0


def train_dqn_cartpole(
    episodes: int = 500,
    max_steps: int = 500,
    seed: int = 0,
) -> Tuple[DQNCartPoleAgent, gym.Env]:
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNCartPoleAgent(obs_dim, n_actions, DQNConfig())

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        for t in range(max_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay.add(obs, action, reward, next_obs, done)
            loss = agent.optimize()
            if (ep * max_steps + t) % agent.cfg.target_update == 0:
                agent.update_target()
            ep_return += reward
            obs = next_obs
            if done:
                break
        # No logging here; this is intentionally minimal.
    return agent, env


def evaluate_agent(
    agent: DQNCartPoleAgent,
    episodes: int = 20,
    max_steps: int = 500,
    noise_sigma: float = 0.05,
    seed: int = 0,
) -> EvalResult:
    base_env = gym.make("CartPole-v1")
    base_env.reset(seed=seed)
    env = NoisyCartPoleEnv(base_env, sigma=noise_sigma, seed=seed)

    returns: List[float] = []
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        for _ in range(max_steps):
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            if terminated or truncated:
                break
        returns.append(ep_return)
    return EvalResult(returns=returns)
