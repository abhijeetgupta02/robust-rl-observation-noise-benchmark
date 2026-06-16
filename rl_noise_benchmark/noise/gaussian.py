"""Additive Gaussian observation noise.

The noise process is deterministic given a fixed seed: it owns a NumPy
``Generator`` so repeated runs with the same seed produce identical noise
sequences. ``sigma`` may be a scalar (applied to every dimension) or an
array broadcastable to the observation shape (per-dimension noise).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def add_gaussian_noise(
    obs: NDArray[np.floating],
    sigma: float | Sequence[float] | NDArray[np.floating],
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Return ``obs + N(0, sigma^2)`` sampled from ``rng``.

    The returned array preserves the dtype of ``obs``. With ``sigma == 0`` a
    copy of ``obs`` is returned (no draw is taken from ``rng``).
    """
    obs = np.asarray(obs)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    if np.all(sigma_arr == 0):
        return obs.copy()
    noise = rng.normal(loc=0.0, scale=1.0, size=obs.shape) * sigma_arr
    return (obs + noise).astype(obs.dtype, copy=False)


class GaussianObservationNoise:
    """Stateful Gaussian observation-noise process.

    Parameters
    ----------
    sigma:
        Standard deviation of the noise. Scalar or array broadcastable to the
        observation shape. Must be non-negative.
    seed:
        Seed for the internal ``Generator``. ``None`` draws fresh entropy
        (non-reproducible); pass an int for reproducible noise.
    """

    def __init__(
        self,
        sigma: float | Sequence[float] | NDArray[np.floating],
        seed: int | None = None,
    ) -> None:
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        if np.any(sigma_arr < 0):
            raise ValueError(f"sigma must be non-negative, got {sigma!r}")
        self.sigma = sigma
        self._seed = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        """Reset the internal generator, optionally re-seeding it."""
        if seed is not None:
            self._seed = seed
        self.rng = np.random.default_rng(self._seed)

    def __call__(self, obs: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply noise to a single observation."""
        return add_gaussian_noise(obs, self.sigma, self.rng)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"GaussianObservationNoise(sigma={self.sigma!r}, seed={self._seed!r})"
