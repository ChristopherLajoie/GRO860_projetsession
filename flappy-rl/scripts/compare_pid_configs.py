
from flappy.env import FlappyEnv
from flappy.pid import PIDAgent
import numpy as np

def eval_config(name, kp, kd, offset, episodes=20):
    env = FlappyEnv(
        use_rays=False,
        wind=True,
        moving_pipes=True,
        pipe_speed_cap=-12.0,
        seed=42
    )
    env.apply_settings(entry_speed=-4.0, entry_transition_steps=600)
    
    scores = []
    for i in range(episodes):
        obs, _ = env.reset(seed=42+i)
        done = False
        agent = PIDAgent(kp=kp, kd=kd, target_offset=offset)
        while not done:
            action, _ = agent.predict(obs)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        scores.append(info.get("pipes", 0))
    
    env.close()
    mean_score = np.mean(scores)
    print(f"Config {name}: Mean Pipes = {mean_score:.2f} (Kp={kp}, Kd={kd}, Off={offset})")
    return mean_score

if __name__ == "__main__":
    print("Comparing PID Configurations...")
    # Old Defaults
    eval_config("Old Defaults", kp=10.0, kd=4.0, offset=0.0)
    # New Tuned
    eval_config("New Tuned", kp=17.43, kd=4.96, offset=0.12)
