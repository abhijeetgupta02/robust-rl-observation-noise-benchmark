import numpy as np

from rl_noise_benchmark.envs import NoisyCartPoleEnv


def test_noisy_env_observation_shape():
    env = NoisyCartPoleEnv(sigma=0.1, seed=0)
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape

    obs2, _, _, _, _ = env.step(env.action_space.sample())
    assert isinstance(obs2, np.ndarray)
    assert obs2.shape == env.observation_space.shape
