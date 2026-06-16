"""A small DQN baseline for CartPole.

Deliberately minimal but real: an MLP Q-network, a uniform replay buffer, an
epsilon-greedy behaviour policy, and a periodically synced target network.
Trains to solve vanilla CartPole within a few minutes on CPU. No part of this
module fabricates results -- returns come from actual environment interaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray


@dataclass
class DQNConfig:
    """Hyperparameters for the DQN agent and its training loop."""

    hidden_sizes: Sequence[int] = (128, 128)
    gamma: float = 0.99
    lr: float = 1e-3
    buffer_size: int = 50_000
    batch_size: int = 64
    learning_starts: int = 1_000
    train_freq: int = 1
    target_update_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    max_grad_norm: float = 10.0
    seed: int = 42


class QNetwork(nn.Module):
    """MLP mapping observations to per-action Q-values."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last, size))
            layers.append(nn.ReLU())
            last = size
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ReplayBuffer:
    """Fixed-size uniform replay buffer backed by NumPy arrays."""

    capacity: int
    obs_dim: int
    rng: np.random.Generator
    _size: int = field(default=0, init=False)
    _idx: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: NDArray[np.floating],
        action: int,
        reward: float,
        next_obs: NDArray[np.floating],
        done: bool,
    ) -> None:
        i = self._idx
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self._idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = self.rng.integers(0, self._size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "actions": torch.as_tensor(self.actions[idx]),
            "rewards": torch.as_tensor(self.rewards[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "dones": torch.as_tensor(self.dones[idx]),
        }


class DQNAgent:
    """DQN agent: network, target network, replay buffer, and update rule."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: DQNConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config or DQNConfig()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device(device)

        self.rng = np.random.default_rng(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.q_net = QNetwork(obs_dim, n_actions, self.config.hidden_sizes).to(
            self.device
        )
        self.target_net = QNetwork(
            obs_dim, n_actions, self.config.hidden_sizes
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.config.lr)
        self._train_steps = 0

    def epsilon(self, step: int) -> float:
        """Linearly annealed exploration rate."""
        cfg = self.config
        frac = min(1.0, step / max(1, cfg.epsilon_decay_steps))
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    def act(self, obs: NDArray[np.floating], epsilon: float = 0.0) -> int:
        """Epsilon-greedy action; ``epsilon=0`` is the greedy (eval) policy."""
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))
        with torch.no_grad():
            obs_t = torch.as_tensor(
                np.asarray(obs, dtype=np.float32), device=self.device
            ).unsqueeze(0)
            q_values = self.q_net(obs_t)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self, batch: dict[str, torch.Tensor]) -> float:
        """One gradient step on the TD(0) Q-learning loss; returns loss value."""
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["dones"].to(self.device)

        q = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_obs).max(dim=1).values
            target = rewards + self.config.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "hidden_sizes": list(self.config.hidden_sizes),
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> "DQNAgent":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = DQNConfig(hidden_sizes=tuple(ckpt["hidden_sizes"]))
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], config=config, device=device)
        agent.q_net.load_state_dict(ckpt["q_net"])
        agent.target_net.load_state_dict(ckpt["q_net"])
        return agent


def _greedy_eval_return(agent: "DQNAgent", env: gym.Env, n_episodes: int) -> float:
    """Mean greedy-policy return over ``n_episodes`` (used for checkpointing)."""
    total = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        done = False
        while not done:
            action = agent.act(obs, epsilon=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = np.asarray(obs, dtype=np.float32)
            total += float(reward)
            done = bool(terminated or truncated)
    return total / n_episodes


def train(
    env: gym.Env,
    config: DQNConfig,
    total_timesteps: int,
    *,
    log_interval: int = 10,
    eval_interval: int = 2_000,
    eval_episodes: int = 5,
    verbose: bool = True,
) -> tuple[DQNAgent, list[float]]:
    """Train a :class:`DQNAgent` on ``env`` for ``total_timesteps`` steps.

    Vanilla DQN on CartPole is known to be unstable late in training (it can
    peak and then collapse). To make the *saved* baseline reliable, the loop
    periodically runs a short greedy evaluation on a fresh copy of ``env`` and
    keeps the best-performing weights. These evaluations are real rollouts; no
    returns are fabricated.

    Returns the trained agent (loaded with the best checkpoint) and the list of
    completed-episode (behaviour-policy) returns.
    """
    import copy

    obs_dim = int(np.prod(env.observation_space.shape))
    assert isinstance(env.action_space, gym.spaces.Discrete)
    n_actions = int(env.action_space.n)

    agent = DQNAgent(obs_dim, n_actions, config=config)
    buffer = ReplayBuffer(config.buffer_size, obs_dim, rng=agent.rng)

    # Separate env for periodic greedy evaluation / checkpoint selection.
    eval_env = copy.deepcopy(env) if eval_interval > 0 else None
    best_eval = -float("inf")
    best_state = copy.deepcopy(agent.q_net.state_dict())

    episode_returns: list[float] = []
    obs, _ = env.reset(seed=config.seed)
    obs = np.asarray(obs, dtype=np.float32)
    ep_return = 0.0
    ep_count = 0

    for step in range(1, total_timesteps + 1):
        eps = agent.epsilon(step)
        action = agent.act(obs, epsilon=eps)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        done = bool(terminated or truncated)
        # Only 'terminated' is a true terminal for bootstrapping; truncation is
        # a time limit and should still bootstrap from next_obs.
        buffer.add(obs, action, float(reward), next_obs, bool(terminated))

        obs = next_obs
        ep_return += float(reward)

        if (
            len(buffer) >= config.learning_starts
            and step % config.train_freq == 0
        ):
            agent.update(buffer.sample(config.batch_size))

        if done:
            episode_returns.append(ep_return)
            ep_count += 1
            if verbose and ep_count % log_interval == 0:
                recent = episode_returns[-log_interval:]
                print(
                    f"step={step:>6}  episodes={ep_count:>4}  "
                    f"eps={eps:.3f}  return(mean last {log_interval})="
                    f"{np.mean(recent):.1f}"
                )
            obs, _ = env.reset()
            obs = np.asarray(obs, dtype=np.float32)
            ep_return = 0.0

        # Periodic greedy evaluation -> keep the best checkpoint.
        if (
            eval_env is not None
            and len(buffer) >= config.learning_starts
            and step % eval_interval == 0
        ):
            mean_ret = _greedy_eval_return(agent, eval_env, eval_episodes)
            if mean_ret > best_eval:
                best_eval = mean_ret
                best_state = copy.deepcopy(agent.q_net.state_dict())
            if verbose:
                print(
                    f"step={step:>6}  [greedy eval] mean_return={mean_ret:.1f}  "
                    f"best={best_eval:.1f}"
                )

    if eval_env is not None:
        eval_env.close()
        # Restore the best-performing weights for the returned/saved agent.
        agent.q_net.load_state_dict(best_state)
        agent.target_net.load_state_dict(best_state)

    return agent, episode_returns
