# Robust RL under Observation Noise Benchmark

Benchmark suite for **robust reinforcement learning under observation noise**, with environments, noise processes, and baseline agents.

> Status: ✨ Planning / scaffolding — initial environments and baselines to be added.

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
