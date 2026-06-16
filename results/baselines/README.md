# Baseline runs

This directory holds reference runs produced by the scripts in `scripts/`.
Nothing here is checked in as a "result" until it comes from a real run on your
machine — there are **no precomputed numbers committed to the repo**.

## Layout

```text
results/baselines/
  <agent>_<env>/
    <model>.pt                       # trained weights (git-ignored by default)
    eval_gaussian_sigma<σ>.json      # evaluation summary written by the eval script
```

For the shipped example that is:

```text
results/baselines/dqn_cartpole/
  dqn_cartpole.pt
  eval_gaussian_sigma0.1.json
```

## How to produce a baseline

```bash
# 1. Train on clean CartPole and save weights here.
python scripts/train_cartpole.py --config configs/cartpole_gaussian.yml

# 2. Evaluate under Gaussian observation noise; writes a JSON summary here.
python scripts/eval_cartpole_noisy.py --config configs/cartpole_gaussian.yml
```

## What the eval JSON contains

Each `eval_*.json` records the agent, env, noise setting, seed, episode count,
and the per-episode returns plus their mean/std/min/max — all computed from
actual rollouts. Example schema (values depend on your run):

```json
{
  "agent": "dqn",
  "env": "CartPole-v1",
  "noise": {"type": "gaussian", "sigma": 0.1},
  "n_episodes": 20,
  "eval_seed": 123,
  "results": {"mean_return": "...", "std_return": "...", "returns": ["..."]}
}
```

## Contributing a baseline

If you want to add a baseline for the (future) leaderboard, include:

- the exact config used,
- the random seed(s),
- the JSON output(s) from the eval script,
- and the library versions (`pip freeze` or `pyproject.toml` lock).

Please do not hand-edit the metric values in the JSON files — report what the
scripts actually produced.
