#!/usr/bin/env python3
"""Train the DQN baseline on clean CartPole-v1 and save the model.

Usage:
    python scripts/train_cartpole.py --config configs/cartpole_gaussian.yml
    python scripts/train_cartpole.py --config configs/cartpole_gaussian.yml \
        --total-timesteps 5000   # quick smoke run

Training is done on the noise-free env; observation noise is applied only at
evaluation time (see scripts/eval_cartpole_noisy.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Allow running directly from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_noise_benchmark.agents.dqn_cartpole import DQNConfig, train  # noqa: E402
from rl_noise_benchmark.envs.cartpole_noisy import make_cartpole  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_dqn_config(cfg: dict) -> DQNConfig:
    agent = cfg.get("agent", {})
    training = cfg.get("training", {})
    return DQNConfig(
        hidden_sizes=tuple(agent.get("hidden_sizes", (128, 128))),
        gamma=agent.get("gamma", 0.99),
        lr=agent.get("lr", 1e-3),
        buffer_size=agent.get("buffer_size", 50_000),
        batch_size=agent.get("batch_size", 64),
        learning_starts=agent.get("learning_starts", 1_000),
        train_freq=agent.get("train_freq", 1),
        target_update_interval=agent.get("target_update_interval", 500),
        epsilon_start=agent.get("epsilon_start", 1.0),
        epsilon_end=agent.get("epsilon_end", 0.05),
        epsilon_decay_steps=agent.get("epsilon_decay_steps", 10_000),
        max_grad_norm=agent.get("max_grad_norm", 10.0),
        seed=training.get("seed", 42),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="YAML config path.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override training.total_timesteps (useful for smoke tests).",
    )
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    env_cfg = cfg.get("env", {})
    training_cfg = cfg.get("training", {})
    output_cfg = cfg.get("output", {})

    total_timesteps = args.total_timesteps or training_cfg.get("total_timesteps", 50_000)
    dqn_config = build_dqn_config(cfg)

    env = make_cartpole(max_episode_steps=env_cfg.get("max_episode_steps"))
    print(
        f"Training DQN on {env_cfg.get('id', 'CartPole-v1')} (clean) "
        f"for {total_timesteps} timesteps, seed={dqn_config.seed} ..."
    )
    agent, episode_returns = train(env, dqn_config, total_timesteps=total_timesteps)
    env.close()

    model_dir = REPO_ROOT / output_cfg.get("model_dir", "results/baselines/dqn_cartpole")
    model_path = model_dir / output_cfg.get("model_name", "dqn_cartpole.pt")
    agent.save(model_path)

    n_eps = len(episode_returns)
    last = episode_returns[-min(20, n_eps):] if n_eps else []
    tail_mean = sum(last) / len(last) if last else float("nan")
    print(
        f"Done. Episodes completed: {n_eps}. "
        f"Mean train return (last {len(last)}): {tail_mean:.1f}"
    )
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
