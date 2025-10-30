# --- Fichier test_agent.py ---

import gymnasium as gym
import numpy as np
from parking import ParkingEnv
from stable_baselines3 import PPO
import pygame # Needed for human_render_loop

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

# --- Main Testing Logic ---
if __name__ == "__main__":
    
    # 1. Register the environment (must match training script)
    gym.register(
        id="Parking-v0",
        entry_point="__main__:ParkingEnv", 
        max_episode_steps=1500
    )

    # 2. Create a SINGLE wrapped environment for testing
    #    IMPORTANT: Render mode is "human" now!
    env = gym.make("Parking-v0", render_mode="human")
    env = NormalizeObservation(env)
    env = NormalizeAction(env)

    # 3. Load the trained model
    model_load_path = "./ppo_parking_model.zip" 
    try:
        model = PPO.load(model_load_path, env=env)
        print(f"Modèle chargé depuis : {model_load_path}")
    except FileNotFoundError:
        print(f"ERREUR: Fichier modèle non trouvé à {model_load_path}")
        print("Assurez-vous d'avoir lancé train_parking.py d'abord.")
        env.close()
        exit()

    # 4. Define the agent function (uses the model's prediction)
    def agent(obs):
        # IMPORTANT: SB3 returns action and hidden states. We only need the action.
        # deterministic=True means the agent always chooses the best action it knows.
        # Set deterministic=False to see more exploration (useful during training).
        action, _states = model.predict(obs, deterministic=True)
        return action

    # 5. Run the human_render_loop with the AGENT driving
    print("Lancement de l'évaluation de l'agent. Appuyez sur 'Q' pour quitter.")
    human_render_loop(agent, env, print_step=False) # print_step=False for cleaner output