#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import yaml

from rl_noise_benchmark.evaluation import evaluate_agent, train_dqn_cartpole


def main() -> None:
    cfg_path = Path("configs/cartpole_gaussian.yml")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging_dir = Path(cfg["logging"]["output_dir"])
    logging_dir.mkdir(parents=True, exist_ok=True)

    agent, _ = train_dqn_cartpole(
        episodes=cfg["agent"]["train_episodes"],
        max_steps=cfg["agent"]["max_steps"],
        seed=cfg["agent"]["seed"],
    )

    result = evaluate_agent(
        agent,
        episodes=cfg["agent"]["eval_episodes"],
        max_steps=cfg["agent"]["max_steps"],
        noise_sigma=cfg["noise"]["sigma"],
        seed=cfg["agent"]["seed"],
    )

    summary = {
        "mean_return": result.mean_return,
        "std_return": result.std_return,
        "episodes": cfg["agent"]["eval_episodes"],
        "noise_sigma": cfg["noise"]["sigma"],
    }

    import json

    with (logging_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Evaluation under noise completed.")
    print(summary)


if __name__ == "__main__":  # pragma: no cover
    main()
