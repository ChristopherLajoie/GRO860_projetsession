
import argparse
import numpy as np
from flappy.env import FlappyEnv
from flappy.pid import PIDAgent
from tqdm import tqdm

def evaluate_pid(kp, kd, offset, episodes=10, seed=42):
    env = FlappyEnv(
        use_rays=False,
        wind=True,
        moving_pipes=True,
        pipe_speed_cap=-12.0, # Match the hard evaluation settings
        seed=seed
    )
    env.apply_settings(entry_speed=-4.0, entry_transition_steps=600)
    
    total_score = 0
    
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        agent = PIDAgent(kp=kp, kd=kd, target_offset=offset)
        
        while not done:
            # PID doesn't need stacked frames if we handle it in predict, 
            # but env returns stacked by default if wrapped? 
            # Here we use raw env, so obs is 1D array.
            action, _ = agent.predict(obs)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            
        total_score += info.get("pipes", 0)
        
    env.close()
    return total_score / episodes

def tune():
    print("Starting PID Tuning...")
    best_score = -1
    best_params = None
    
    # Random Search
    n_trials = 50
    
    for i in range(n_trials):
        kp = np.random.uniform(0.0, 25.0)
        kd = np.random.uniform(0.0, 15.0)
        offset = np.random.uniform(-0.2, 0.2)
        
        score = evaluate_pid(kp, kd, offset, episodes=5)
        
        if score > best_score:
            best_score = score
            best_params = (kp, kd, offset)
            print(f"New Best: Score={score:.2f} | Kp={kp:.2f}, Kd={kd:.2f}, Offset={offset:.2f}")
            
    print("\n--- Tuning Complete ---")
    print(f"Best Score: {best_score:.2f}")
    print(f"Best Params: Kp={best_params[0]:.2f}, Kd={best_params[1]:.2f}, Offset={best_params[2]:.2f}")

if __name__ == "__main__":
    tune()
