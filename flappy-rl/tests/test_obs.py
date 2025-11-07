from flappy.env import FlappyEnv


def test_default_observation_contains():
    env = FlappyEnv()
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    assert env.observation_space.shape[0] == 7


def test_rich_observation_contains():
    env = FlappyEnv(use_rays=True, n_rays=5, wind=True, energy=True)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    assert env.observation_space.shape[0] == 11
