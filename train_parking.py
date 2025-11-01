import gymnasium as gym
import numpy as np
from parking import ParkingEnv
from performance_tracker import PerformanceTracker
import pygame 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        unwrapped_env = env.unwrapped

        world_w = unwrapped_env.world_width / 2
        world_h = unwrapped_env.world_height / 2
        max_v = unwrapped_env.MAX_VELOCITY
        
        low = np.array(
            [-world_w, -world_h, -np.pi, -max_v/2,      # car (4)
             -world_w, -world_h,                         # target (2)
             -1.0, -1.0,                                 # target_dir (2)
             -np.pi,                                     # angle_error (1)
             -world_w, -world_h, -world_w, -world_h,    # 4 obstacles positions (8)
             -world_w, -world_h, -world_w, -world_h,
             0.0, 0.0, 0.0, 0.0,                        # distances (4)
             0.0, 0.0, 0.0, 0.0],                       # danger scores (4)
            dtype=np.float32
        )
        
        # Max distance = diagonale du monde
        max_dist = np.sqrt((2*world_w)**2 + (2*world_h)**2)
        
        high = np.array(
            [world_w, world_h, np.pi, max_v,
             world_w, world_h,
             1.0, 1.0,
             np.pi,
             world_w, world_h, world_w, world_h,
             world_w, world_h, world_w, world_h,
             max_dist, max_dist, max_dist, max_dist,    # distances max
             1.0, 1.0, 1.0, 1.0],                        # danger scores max (exp(0) = 1)
            dtype=np.float32
        )

        self.obs_low = low
        self.obs_high = high
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(25,), dtype=np.float32)

    def observation(self, obs):
        # Normaliser entre -1 et 1
        range_ = self.obs_high - self.obs_low
        range_[range_ == 0] = 1e-6
        return -1.0 + 2.0 * (obs - self.obs_low) / range_
    
class NormalizeAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32)

    def action(self, act):
        low = self.env.action_space.low
        high = self.env.action_space.high
        return low + (high - low) * (act + 1.0) / 2.0

# --- Callback pour intégrer le Performance Tracker ---
class PerformanceCallback(BaseCallback):
    def __init__(self, tracker, verbose=0):
        super(PerformanceCallback, self).__init__(verbose)
        self.tracker = tracker
        self.episode_rewards = []
        self.episode_data = {}
    
    def _on_step(self) -> bool:
        # Pour chaque environnement dans le vec_env
        for idx in range(self.training_env.num_envs):
            # Si pas encore initialisé pour cet env
            if idx not in self.episode_data:
                self.episode_data[idx] = {
                    'tracker_initialized': False,
                    'episode_reward': 0
                }
            
            # Récupérer les infos du step
            info = self.locals.get('infos', [{}] * self.training_env.num_envs)[idx]
            reward = self.locals['rewards'][idx]
            done = self.locals['dones'][idx]
            
            self.episode_data[idx]['episode_reward'] += reward
            
            # Si premier step de l'épisode
            if not self.episode_data[idx]['tracker_initialized']:
                self.tracker.reset_episode()
                self.episode_data[idx]['tracker_initialized'] = True
            
            # Enregistrer le step
            if 'state' in info:
                self.tracker.step(
                    state=info['state'],
                    reward=reward,
                    terminated=done,
                    truncated=False,
                    info=info
                )
            
            # Si épisode terminé
            if done:
                self.tracker.end_episode()
                self.episode_data[idx]['tracker_initialized'] = False
                self.episode_data[idx]['episode_reward'] = 0
        
        return True

# --- Main Training Logic ---
if __name__ == "__main__":
    
    # 1. Register the environment
    gym.register(
        id="Parking-v0",
        entry_point="parking:ParkingEnv",
        max_episode_steps=3000
    )
    print("Environnement 'Parking-v0' enregistré.")

    # 2. Initialiser le tracker de performance
    tracker = PerformanceTracker(log_file="parking_performance.txt", batch_size=100)
    print("Tracker de performance initialisé.")

    # 3. Create and wrap the environment
    env_id = "Parking-v0"
    n_envs = 8  # Environnements parallèles
    
    def make_env():
        env = gym.make(env_id)
        env = NormalizeObservation(env)
        env = NormalizeAction(env)
        return env

    vec_env = make_vec_env(make_env, n_envs=n_envs)
    print(f"{n_envs} environnements vectorisés créés.")

    # 4. Instantiate PPO agent avec hyperparamètres optimisés
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1, 
        learning_rate=5e-4,          # Augmenté (était 3e-4)
        n_steps=4096,                # Augmenté (était 2048)
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,               # Augmenté pour plus d'exploration (était 0.01)
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./ppo_parking_tensorboard/"
    )

    # 5. Create callback
    callback = PerformanceCallback(tracker=tracker, verbose=1)

    # 6. Train the agent
    total_timesteps = 2_000_000  # 2M timesteps
    print(f"Début de l'entraînement pour {total_timesteps:,} timesteps...")
    print(f"Les performances seront loggées dans 'parking_performance.txt'")
    print(f"Un batch = 100 épisodes\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur.")
    
    print("\nEntraînement terminé.")
    
    # 7. Save the trained model
    model_save_path = "./ppo_parking_model_improved"
    model.save(model_save_path)
    print(f"Modèle sauvegardé à : {model_save_path}.zip")
    
    # 8. Afficher le résumé final
    summary = tracker.get_summary()
    print("\n" + "="*80)
    print("RÉSUMÉ FINAL")
    print("="*80)
    print(f"Total d'épisodes: {summary['total_episodes']}")
    print(f"Taux de succès récent (100 derniers): {summary['recent_success_rate']:.1f}%")
    print(f"Voir 'parking_performance.txt' pour le détail complet")
    print("="*80)

    vec_env.close()