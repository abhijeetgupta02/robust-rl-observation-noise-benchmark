#!/usr/bin/env python3
"""Plot mean evaluation return vs. observation-noise sigma.

Reads the per-sigma evaluation JSONs written by ``scripts/eval_cartpole_noisy.py``
(``eval_gaussian_sigma*.json``) and produces a single figure of mean return with
std error bars as a function of the Gaussian noise level sigma.

Usage:
    # after training + evaluating at a few sigmas:
    python scripts/eval_cartpole_noisy.py --config configs/cartpole_gaussian.yml --sigma 0.0 --n-episodes 10
    python scripts/eval_cartpole_noisy.py --config configs/cartpole_gaussian.yml --sigma 0.1 --n-episodes 10
    python scripts/eval_cartpole_noisy.py --config configs/cartpole_gaussian.yml --sigma 0.2 --n-episodes 10
    python scripts/eval_cartpole_noisy.py --config configs/cartpole_gaussian.yml --sigma 0.3 --n-episodes 10

    python scripts/plot_return_vs_sigma.py

Every value in the plot comes from real rollouts on your machine. Nothing is
hard-coded -- if there are no eval JSONs, the script tells you to run the
evaluation first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "baselines" / "dqn_cartpole"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "cartpole_return_vs_sigma.png"


def load_points(results_dir: Path) -> list[dict]:
    """Read every eval_gaussian_sigma*.json and return points sorted by sigma."""
    points: list[dict] = []
    for path in sorted(results_dir.glob("eval_gaussian_sigma*.json")):
        with path.open("r", encoding="utf-8") as fh:
            record = json.load(fh)
        results = record["results"]
        points.append(
            {
                "sigma": float(record["noise"]["sigma"]),
                "mean": float(results["mean_return"]),
                "std": float(results["std_return"]),
                "n_episodes": int(results["n_episodes"]),
                "agent": record.get("agent", "agent"),
                "env": record.get("env", "env"),
            }
        )
    points.sort(key=lambda p: p["sigma"])
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing eval_gaussian_sigma*.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path.",
    )
    args = parser.parse_args()

    points = load_points(args.results_dir)
    if not points:
        sys.exit(
            f"No eval_gaussian_sigma*.json found in {args.results_dir}.\n"
            "Run scripts/eval_cartpole_noisy.py at a few sigmas first."
        )

    sigmas = [p["sigma"] for p in points]
    means = [p["mean"] for p in points]
    stds = [p["std"] for p in points]
    agent = points[0]["agent"].upper()
    env = points[0]["env"]
    n_eps = points[0]["n_episodes"]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.errorbar(
        sigmas,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
        linewidth=1.8,
        label=f"{agent} (mean ± std, n={n_eps} eps/point)",
    )
    ax.set_xlabel("Observation-noise std-dev  σ")
    ax.set_ylabel("Episodic return")
    ax.set_title(f"{agent} on {env}: return vs. Gaussian observation noise")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output} from {len(points)} sigma point(s): {sigmas}")


if __name__ == "__main__":
    main()
