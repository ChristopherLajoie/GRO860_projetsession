# --- Fichier train_parking.py ---

import gymnasium as gym
import numpy as np
from parking import ParkingEnv
import pygame 

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv # Optional, for parallel training

def human_render_loop(agent, env, transpose=False, print_step=False):
    obs, info = env.reset()
    clock = pygame.time.Clock()
    fps = env.metadata["render_fps"]

    term = trunc = quit_loop = False
    step = 0
    total_reward = 0

    while not (term or trunc or quit_loop):
        # Vérifier si l'utilisateur veut quitter
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_loop = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                quit_loop = True

        action = agent(obs)
        obs, reward, term, trunc, info = env.step(action)
        
        env.render() # L'appel à render() dessine l'écran

        step += 1
        total_reward += reward

        if print_step:
            print(f"Step {step}, Action {action}, Info {info}")

        clock.tick(fps)
    
    env.close()
    print(f"Boucle terminée. Récompense totale : {total_reward}")

# --- Wrappers (copiés/adaptés de gym.py) ---
class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        unwrapped_env = env.unwrapped

        # [car_x, car_y, car_theta, car_v, target_x, target_y]
        # Set bounds based on world size
        world_w = unwrapped_env.world_width / 2
        world_h = unwrapped_env.world_height / 2
        max_v = unwrapped_env.MAX_VELOCITY

        low = np.array([-world_w, -world_h, -np.pi, -max_v/2, -world_w, -world_h], dtype=np.float32)
        high = np.array([world_w, world_h, np.pi, max_v, world_w, world_h], dtype=np.float32)

        self.obs_low = low
        self.obs_high = high
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    def observation(self, obs):
        # ... (Normalization formula remains the same, just works on the 5-element array now)
        range_ = self.obs_high - self.obs_low
        range_[range_ == 0] = 1e-6
        return -1.0 + 2.0 * (obs - self.obs_low) / range_
    
class NormalizeAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Agent outputs actions in [-1, 1] range
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32)

    def action(self, act):
        # Denormalize action from [-1, 1] to env's action space bounds
        low = self.env.action_space.low
        high = self.env.action_space.high
        # Formula: low + (high - low) * (act + 1) / 2
        return low + (high - low) * (act + 1.0) / 2.0

# --- Main Training Logic ---
if __name__ == "__main__":
    
    # 1. Register the environment
    #    The string format is "Namespace:EnvName-vVersion"
    gym.register(
        id="Parking-v0",
        entry_point="__main__:ParkingEnv", # Assumes ParkingEnv is in the same file
        max_episode_steps=1500 # Limit episode length (important!)
    )
    
    print("Environnement 'Parking-v0' enregistré.")

    # 2. Create and wrap the environment
    #    make_vec_env creates multiple instances for parallel training (faster)
    #    n_envs=4 means 4 parallel environments
    #    Adjust n_envs based on your CPU cores
    env_id = "Parking-v0"
    n_envs = 4 
    
    # Define a function to create a single wrapped environment
    def make_env():
        env = gym.make(env_id)
        env = NormalizeObservation(env)
        env = NormalizeAction(env)
        return env

    # Create the vectorized environment
    # Use SubprocVecEnv for true parallelism if desired, otherwise default DummyVecEnv is fine
    vec_env = make_vec_env(make_env, n_envs=n_envs) #, vec_env_cls=SubprocVecEnv) 
    
    print(f"{n_envs} environnements vectorisés et enveloppés créés.")

    # 3. Instantiate the PPO agent
    #    "MlpPolicy" is a standard neural network policy
    #    verbose=1 prints training progress
    #model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_parking_tensorboard/")
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=1e-4, tensorboard_log="./ppo_parking_tensorboard/")
    
    print("Agent PPO créé.")

    # 4. Train the agent
    #    total_timesteps is crucial. Start small (e.g., 10k) to test, 
    #    but you'll likely need 1M+ for good results.
    total_timesteps = 2_000_000 # Start with 100k, increase later
    print(f"Début de l'entraînement pour {total_timesteps} timesteps...")
    
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    print("Entraînement terminé.")

    # 5. Save the trained model
    model_save_path = "./ppo_parking_model"
    model.save(model_save_path)
    print(f"Modèle sauvegardé à : {model_save_path}.zip")

    vec_env.close()