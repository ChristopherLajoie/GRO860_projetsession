from flappy.env import FlappyEnv


def test_reward_improves_when_distance_shrinks():
    env = FlappyEnv()
    closer = env._compute_reward(dist=10.0, prev_dist=20.0, flap_used=0.0, pipe_cross=False, crash=False)
    farther = env._compute_reward(dist=30.0, prev_dist=20.0, flap_used=0.0, pipe_cross=False, crash=False)
    assert closer > farther
