#!/usr/bin/env python3
"""Evaluate a trained DQN agent on CartPole under Gaussian observation noise.

Usage:
    python scripts/eval_cartpole_noisy.py --config configs/cartpole_gaussian.yml
    python scripts/eval_cartpole_noisy.py --config configs/cartpole_gaussian.yml \
        --sigma 0.2 --n-episodes 50

Prints a summary and writes a JSON results file under results/baselines/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Allow running directly from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_noise_benchmark import __version__  # noqa: E402
from rl_noise_benchmark.agents.dqn_cartpole import DQNAgent  # noqa: E402
from rl_noise_benchmark.envs.cartpole_noisy import make_noisy_cartpole  # noqa: E402
from rl_noise_benchmark.evaluation.eval_loop import evaluate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="YAML config path.")
    parser.add_argument(
        "--sigma", type=float, default=None, help="Override evaluation noise sigma."
    )
    parser.add_argument(
        "--n-episodes", type=int, default=None, help="Override number of eval episodes."
    )
    parser.add_argument(
        "--model", type=Path, default=None, help="Override path to the saved model."
    )
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    env_cfg = cfg.get("env", {})
    noise_cfg = cfg.get("noise", {})
    eval_cfg = cfg.get("eval", {})
    output_cfg = cfg.get("output", {})

    # Resolve eval sigma: CLI > eval.sigma > noise.sigma.
    sigma = args.sigma
    if sigma is None:
        sigma = eval_cfg.get("sigma")
    if sigma is None:
        sigma = noise_cfg.get("sigma", 0.1)

    n_episodes = args.n_episodes or eval_cfg.get("n_episodes", 20)
    eval_seed = eval_cfg.get("seed", 123)

    model_dir = REPO_ROOT / output_cfg.get("model_dir", "results/baselines/dqn_cartpole")
    model_path = args.model or (model_dir / output_cfg.get("model_name", "dqn_cartpole.pt"))
    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run scripts/train_cartpole.py first."
        )

    agent = DQNAgent.load(model_path, device="cpu")
    env = make_noisy_cartpole(
        sigma=sigma,
        seed=eval_seed,
        max_episode_steps=env_cfg.get("max_episode_steps"),
    )

    print(
        f"Evaluating {model_path.name} on "
        f"{env_cfg.get('id', 'CartPole-v1')} + Gaussian(sigma={sigma}) "
        f"for {n_episodes} episodes (seed={eval_seed}) ..."
    )
    result = evaluate(agent, env, n_episodes=n_episodes, seed=eval_seed)
    env.close()

    print(
        f"Return: mean={result.mean_return:.1f} +/- {result.std_return:.1f} "
        f"(min={result.min_return:.0f}, max={result.max_return:.0f}, "
        f"n={result.n_episodes})"
    )

    record = {
        "benchmark_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "agent": cfg.get("agent", {}).get("type", "dqn"),
        "env": env_cfg.get("id", "CartPole-v1"),
        "noise": {"type": noise_cfg.get("type", "gaussian"), "sigma": sigma},
        "n_episodes": n_episodes,
        "eval_seed": eval_seed,
        "model_path": str(model_path),
        "results": result.as_dict(),
    }

    out_dir = REPO_ROOT / output_cfg.get("model_dir", "results/baselines/dqn_cartpole")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_gaussian_sigma{sigma}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
