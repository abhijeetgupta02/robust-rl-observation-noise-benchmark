from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 10_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 500
    target_update: int = 100


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_obs[idxs],
            self.dones[idxs],
        )


class DQNCartPoleAgent:
    """Simple DQN agent for CartPole.

    This is intentionally minimal and CPU-friendly. It is not tuned for
    leaderboard performance, but it should be able to solve CartPole-v1 in
    reasonable time on a laptop when trained without noise.
    """

    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig) -> None:
        self.device = torch.device("cpu")
        self.cfg = cfg
        self.q_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_size, obs_dim)
        self.steps_done = 0

    def select_action(self, obs: np.ndarray) -> int:
        eps = self._epsilon_by_step(self.steps_done)
        self.steps_done += 1
        if np.random.rand() < eps:
            return np.random.randint(0, self.n_actions)
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        return int(torch.argmax(q_values, dim=1).item())

    @property
    def n_actions(self) -> int:
        return self.q_net.net[-1].out_features

    def _epsilon_by_step(self, step: int) -> float:
        cfg = self.cfg
        return cfg.eps_end + (cfg.eps_start - cfg.eps_end) * np.exp(
            -1.0 * step / cfg.eps_decay
        )

    def optimize(self) -> float:
        if self.replay.size < self.cfg.batch_size:
            return 0.0
        (
            obs,
            actions,
            rewards,
            next_obs,
            dones,
        ) = self.replay.sample(self.cfg.batch_size)

        obs_t = torch.from_numpy(obs).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        next_obs_t = torch.from_numpy(next_obs).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)

        q_values = self.q_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_obs_t).max(1).values
            target = rewards_t + self.cfg.gamma * (1.0 - dones_t) * next_q

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())
