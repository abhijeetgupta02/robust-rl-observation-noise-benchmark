"""rl_noise_benchmark: robust RL under observation noise (minimal first slice).

This package currently ships a single vertical slice:

- one environment wrapper (:mod:`rl_noise_benchmark.envs`) that injects
  observation noise into Gymnasium's ``CartPole-v1``,
- one noise process (:mod:`rl_noise_benchmark.noise`) -- additive Gaussian,
- one baseline agent (:mod:`rl_noise_benchmark.agents`) -- a small DQN,
- and an evaluation loop (:mod:`rl_noise_benchmark.evaluation`).

It is intentionally small and is meant as a starting point for a larger
benchmark, not a finished one. No baseline numbers are claimed here.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
