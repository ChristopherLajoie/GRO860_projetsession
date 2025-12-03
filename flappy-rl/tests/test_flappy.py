
import pytest
import numpy as np
from flappy.env import FlappyEnv
from flappy.physics import G, MAX_VY, PIPE_VX
from flappy.pid import PIDAgent

# --- Physics Tests ---
def test_gravity():
    """Test that the bird accelerates downwards due to gravity."""
    env = FlappyEnv(seed=42)
    env.reset()
    obs, _, _, _, _ = env.step(0)
    state = env._state
    assert np.isclose(state["bird_vy"], G), f"Expected vy={G}, got {state['bird_vy']}"

def test_wind_effect():
    """Test that wind affects the effective pipe speed."""
    env = FlappyEnv(wind=True, seed=42)
    env.reset()
    
    env._state["wind"] = 0.0
    s0 = env.step(0)
    x_no_wind = env._state["x_pipe"]
    
    env.reset()
    env._state["wind"] = 0.5 
    s1 = env.step(0)
    x_wind = env._state["x_pipe"]
    
    assert x_wind != x_no_wind, "Wind should affect pipe movement"

# --- PID Tests ---
def test_pid_reaction():
    """Test that the PID controller flaps when well below the target."""
    env = FlappyEnv(seed=42)
    obs, _ = env.reset()
    env._state["bird_y"] = 500 
    env._state["bird_vy"] = 5.0 
    obs = env._get_obs(env._state)
    
    agent = PIDAgent(kp=10, kd=0, target_offset=0)
    action, _ = agent.predict(obs)
    assert action == 1, "PID should flap when falling below the gap"

def test_pid_idle():
    """Test that PID does nothing when well above the target."""
    env = FlappyEnv(seed=42)
    obs, _ = env.reset()
    env._state["bird_y"] = 100 
    env._state["bird_vy"] = -5.0 
    obs = env._get_obs(env._state)
    
    agent = PIDAgent(kp=10, kd=0, target_offset=0)
    action, _ = agent.predict(obs)
    assert action == 0, "PID should not flap when rising above the gap"

# --- Reproducibility Tests ---
def test_determinism():
    """Test that the environment is deterministic given a seed."""
    seed = 12345
    env1 = FlappyEnv(seed=seed, wind=True, moving_pipes=True)
    obs1, _ = env1.reset()
    actions = [0, 1, 0, 0, 1, 1, 0]
    traj1 = []
    for a in actions:
        o, r, term, trunc, _ = env1.step(a)
        traj1.append((o, r))
        
    env2 = FlappyEnv(seed=seed, wind=True, moving_pipes=True)
    obs2, _ = env2.reset()
    traj2 = []
    for a in actions:
        o, r, term, trunc, _ = env2.step(a)
        traj2.append((o, r))
        
    assert np.allclose(obs1, obs2), "Initial observations match"
    for (o1, r1), (o2, r2) in zip(traj1, traj2):
        assert np.allclose(o1, o2), "Observations should match"
        assert r1 == r2, "Rewards should match"

def test_different_seeds():
    """Test that different seeds produce different initial states."""
    env1 = FlappyEnv(seed=1, wind=True)
    env2 = FlappyEnv(seed=2, wind=True)
    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    w1 = env1._state["wind"]
    w2 = env2._state["wind"]
    assert w1 != w2, "Different seeds should produce different wind values"

# --- Reward Tests ---
def test_crash_penalty():
    """Test that crashing results in a negative reward."""
    env = FlappyEnv(seed=42)
    env.reset()
    env._state["bird_y"] = -10.0 # Clearly out of bounds
    obs, reward, term, trunc, info = env.step(0)
    assert term, "Bird should be dead"
    assert reward < -1.0, f"Crash reward should be significantly negative, got {reward}"

def test_pass_reward():
    """Test that passing a pipe results in a positive reward."""
    env = FlappyEnv(seed=42)
    env.reset()
    env._state["x_pipe"] = 21.0
    env._state["bird_y"] = env._state["gap_center_y"] 
    obs, reward, term, trunc, info = env.step(0)
    assert info["passed"], "Should have passed pipe"
    assert reward > 0.5, f"Pass reward should be positive, got {reward}"

# --- Curriculum Tests ---
def test_difficulty_update():
    """Test that applying settings actually changes the environment physics."""
    env = FlappyEnv(seed=42, wind=False, moving_pipes=False)
    env.reset() # Initialize state
    assert not env.wind
    
    env.apply_settings(wind=True)
    assert env.wind, "Wind should be enabled"
    
    target_speed = -20.0
    env.apply_settings(pipe_speed=target_speed)
    assert env._cfg.pipe_vx == target_speed
    
    env.step(0)
    assert env._state["pipe_vx"] == target_speed

def test_gap_range_update():
    """Test that we can shrink the gap size dynamically."""
    env = FlappyEnv(seed=42)
    env.reset() # Initialize state
    new_range = (50.0, 50.0)
    env.apply_settings(gap_height_range=new_range)
    assert env._cfg.gap_height_range == new_range
    
    env._state["x_pipe"] = -100 
    env.step(0)
    current_gap = env._state["gap_height"]
    assert current_gap == 50.0
