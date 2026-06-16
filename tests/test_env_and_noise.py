"""Tests for the noise process and the noisy CartPole wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from rl_noise_benchmark.envs.cartpole_noisy import make_cartpole, make_noisy_cartpole
from rl_noise_benchmark.noise.gaussian import (
    GaussianObservationNoise,
    add_gaussian_noise,
)


def test_noisy_env_obs_shape_and_dtype():
    env = make_noisy_cartpole(sigma=0.1, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, _ = env.step(action)
    assert obs2.shape == env.observation_space.shape
    assert obs2.dtype == np.float32
    assert np.isscalar(reward) or np.ndim(reward) == 0
    env.close()


def test_noise_is_reproducible_with_fixed_seed():
    obs = np.zeros(4, dtype=np.float32)
    n1 = GaussianObservationNoise(sigma=0.5, seed=123)
    n2 = GaussianObservationNoise(sigma=0.5, seed=123)
    seq1 = [n1(obs) for _ in range(5)]
    seq2 = [n2(obs) for _ in range(5)]
    for a, b in zip(seq1, seq2):
        np.testing.assert_array_equal(a, b)


def test_noise_differs_across_seeds():
    obs = np.zeros(4, dtype=np.float32)
    a = GaussianObservationNoise(sigma=0.5, seed=1)(obs)
    b = GaussianObservationNoise(sigma=0.5, seed=2)(obs)
    assert not np.allclose(a, b)


def test_noisy_env_reproducible_across_resets():
    env1 = make_noisy_cartpole(sigma=0.2, seed=7)
    env2 = make_noisy_cartpole(sigma=0.2, seed=7)
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)
    # Same action sequence -> identical noisy observations.
    for action in [0, 1, 0, 1, 1]:
        o1, *_ = env1.step(action)
        o2, *_ = env2.step(action)
        np.testing.assert_array_equal(o1, o2)
    env1.close()
    env2.close()


def test_zero_sigma_is_noiseless():
    obs = np.array([1.0, -2.0, 3.0, 0.5], dtype=np.float32)
    noise = GaussianObservationNoise(sigma=0.0, seed=0)
    np.testing.assert_array_equal(noise(obs), obs)


def test_noise_actually_perturbs_observation():
    clean_env = make_cartpole()
    noisy_env = make_noisy_cartpole(sigma=0.5, seed=3)
    clean_obs, _ = clean_env.reset(seed=10)
    noisy_obs, _ = noisy_env.reset(seed=10)
    # Underlying state is the same seed; observations should differ due to noise.
    assert not np.allclose(clean_obs, noisy_obs)
    clean_env.close()
    noisy_env.close()


def test_add_gaussian_noise_preserves_dtype():
    obs = np.zeros(4, dtype=np.float32)
    rng = np.random.default_rng(0)
    out = add_gaussian_noise(obs, 0.1, rng)
    assert out.dtype == np.float32
    assert out.shape == obs.shape


def test_negative_sigma_rejected():
    with pytest.raises(ValueError):
        GaussianObservationNoise(sigma=-1.0)
