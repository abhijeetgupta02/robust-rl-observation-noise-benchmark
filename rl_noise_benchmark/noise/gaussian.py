from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianObservationNoise:
    """Additive Gaussian noise for continuous observations.

    Parameters
    ----------
    sigma:
        Standard deviation of the noise applied independently to each
        observation dimension.
    """

    sigma: float = 0.05

    def apply(self, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        noise = rng.normal(loc=0.0, scale=self.sigma, size=obs.shape).astype(
            np.float32
        )
        return obs + noise
