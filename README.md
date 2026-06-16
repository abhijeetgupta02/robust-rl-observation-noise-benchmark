# Robust RL under Observation Noise Benchmark

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-experimental-orange)](https://github.com/abhijeetgupta02/robust-rl-observation-noise-benchmark)
[![Maintainer](https://img.shields.io/badge/maintainer-Abhijeet%20Gupta-0e75b6)](https://github.com/abhijeetgupta02)

Benchmark suite for **robust reinforcement learning under observation noise**, with environments, noise processes, and baseline agents.

> Status: ✨ Planning / scaffolding — initial environments and baselines to be added.

## Why you might care

- You want to **stress-test RL agents** when sensors are noisy, delayed, or missing.
- You need a **standardized CartPole noise benchmark** for papers, not one-off scripts.
- You care about **seeded, reproducible runs** with clear configs and JSON logs.

## Goals

- Provide a **standardized benchmark** for RL under observation noise.
- Include **noise processes** that reflect realistic sensor imperfections.
- Ship **baseline agents** (PPO, SAC, etc.) with clear training/eval scripts.
- Offer a **leaderboard-style README** where others can add their scores.

## Benchmark Design (planned)

- A small set of base environments (e.g., classic control, simple continuous control).
- Configurable noise types, such as:
  - Gaussian observation noise
  - Missing observations / random masking
  - Latency / delayed observations
- Standard train/eval splits and evaluation protocol.

## Repository Layout (planned)

```text
robust-rl-observation-noise-benchmark/
  rl_noise_benchmark/
    envs/             # Environment wrappers with noise injection
    noise/            # Noise process definitions
    agents/           # Baseline agents (PPO, SAC, etc.)
    evaluation/       # Evaluation loops and metrics
  configs/
    ppo_gaussian.yml
    sac_masking.yml
  scripts/
    train.py
    evaluate.py
  results/
    baselines/        # Reference runs for the paper / docs
  tests/
  README.md
  pyproject.toml
```

## Leaderboard (planned)

Once baselines are in place, a simple markdown table like:

```markdown
| Agent | Env + Noise         | Return (mean ± std) | Seeds | Paper / Repo |
|-------|---------------------|---------------------|-------|--------------|
| PPO   | CartPole + Gaussian | TBD                 | TBD   | -            |
```

## Getting Started

Planned steps:

1. Choose the initial environment set and noise processes.
2. Implement environment wrappers and noise modules.
3. Add PPO baseline and reference configs.
4. Release initial baselines and populate the leaderboard.

If you are interested in using or extending this benchmark, feel free to open an issue with:

- Environment / noise combinations you care about.
- Agent types you want baseline comparisons for.
- Metrics that are most relevant for your use case.

---

## Quickstart (minimal first slice)

A first **runnable** vertical slice is implemented:

- **Environment:** `CartPole-v1` (Gymnasium) with an observation-noise wrapper.
- **Noise:** additive Gaussian observation noise, `obs_noisy = obs + N(0, σ²)`,
  reproducible given a seed.
- **Agent:** a small PyTorch **DQN** baseline.
- **Eval:** episodic evaluation under noise reporting return mean ± std.

> ⚠️ This is a *starting point*, not a finished benchmark. It currently contains
> **one** environment, **one** noise process, and **one** agent. No leaderboard
> numbers are claimed, and none are committed to the repo — see
> [No results are fabricated](#no-results-are-fabricated) below.

### Install

Requires Python 3.10+ and runs on a **CPU-only** laptop.

```bash
pip install -e .          # installs the package + deps (gymnasium, torch, numpy, pyyaml)
pip install -e ".[dev]"   # optional: adds pytest
```

### 1. Train the baseline (clean CartPole)

Training uses the **noise-free** environment; noise is applied only at
evaluation time. The loop periodically runs a short greedy evaluation and keeps
the best checkpoint (vanilla DQN can otherwise peak and then collapse).

```bash
python scripts/train_cartpole.py --config configs/cartpole_gaussian.yml
# quick smoke run:
python scripts/train_cartpole.py --config configs/cartpole_gaussian.yml --total-timesteps 3000
```

Weights are saved to `results/baselines/dqn_cartpole/dqn_cartpole.pt`. On a
typical laptop CPU the default run takes only a couple of minutes.

### 2. Evaluate under observation noise

```bash
python scripts/eval_cartpole_noisy.py --config configs/cartpole_gaussian.yml
# sweep the noise level:
python scripts/eval_cartpole_noisy.py --config configs/cartpole_gaussian.yml --sigma 0.2 --n-episodes 50
```

This prints `mean ± std` return and writes a JSON summary (agent, env, noise
level, seed, and per-episode returns) under `results/baselines/dqn_cartpole/`.

### Run the tests

```bash
pytest
```

The tests check observation shape/dtype from the noisy wrapper and that the
Gaussian noise is reproducible under a fixed seed (no training required).

## No results are fabricated

- The README contains **no performance numbers**. Any returns you see come from
  running the scripts yourself on your own machine.
- Trained weights (`*.pt`) and eval JSONs are **git-ignored**; they are
  reproducible artifacts, not committed claims.
- The leaderboard below is an **empty template** with `TBD` placeholders. Rows
  will be filled only after real, seeded runs.

## Leaderboard (template — values are placeholders)

| Agent | Env + Noise         | Return (mean ± std) | Seeds | Paper / Repo |
|-------|---------------------|---------------------|-------|--------------|
| DQN   | CartPole + Gaussian | TBD                 | TBD   | -            |
| PPO   | CartPole + Gaussian | TBD                 | TBD   | -            |

## Implemented layout

```text
rl_noise_benchmark/
  envs/cartpole_noisy.py   # CartPole wrapper that injects observation noise
  noise/gaussian.py        # reproducible additive Gaussian noise process
  agents/dqn_cartpole.py   # small DQN baseline (network, replay, training loop)
  evaluation/eval_loop.py  # greedy evaluation -> return mean/std
configs/cartpole_gaussian.yml
scripts/train_cartpole.py
scripts/eval_cartpole_noisy.py
results/baselines/         # where trained models + eval JSONs land (git-ignored)
tests/test_env_and_noise.py
```

## Contributing

This benchmark is designed to grow. Good first contributions:

- **Add a noise process** — implement it in `rl_noise_benchmark/noise/`
  (e.g. observation masking/dropout, delayed/latent observations, salt-and-pepper)
  following the `GaussianObservationNoise` interface (`__call__(obs) -> obs`,
  reproducible under a seed).
- **Add an environment wrapper** — under `rl_noise_benchmark/envs/`, wrapping any
  Gymnasium env and applying a noise process to observations.
- **Add an agent** — under `rl_noise_benchmark/agents/` (e.g. PPO, SAC, or a
  robustness-specific method), exposing an `act(obs, epsilon=0) -> action` method
  so it plugs into `evaluation/eval_loop.py`.
- **Contribute a baseline run** — see `results/baselines/README.md` for the
  expected config + seed + JSON format.

Please keep new code typed, CPU-runnable, and free of fabricated metrics.
