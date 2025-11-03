"""
Script d'entraînement PPO pour l'environnement Parking-v0 (LIDAR).
MODIFIÉ pour:
 - utiliser VecNormalize pour normalisation des observations (et optionnellement des rewards)
 - ajouter un EvalCallback (évaluations régulières)
 - conserver votre PerformanceCallback et CheckpointCallback
"""

import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from parking import ParkingEnv

# -- register env --
gym.register(
    id="Parking-v0",
    entry_point="parking:ParkingEnv",
    max_episode_steps=2000
)
print("Environnement 'Parking-v0' (Lidar) enregistré.")

# -- env factory --
def make_env(rank, seed=0):
    def _init():
        env = gym.make("Parking-v0")
        # Don't wrap with Monitor here - VecMonitor will handle it
        return env
    return _init

if __name__ == "__main__":
    # configuration (tweakable)
    NUM_CPU = 8
    TOTAL_TIMESTEPS = 5_000_000
    MODEL_NAME = "ppo_parking_lidar"
    LOG_DIR = "./logs/"
    MODEL_DIR = "./models/"
    EVAL_FREQ = 50_000           # evaluate every this many environment steps
    EVAL_EPISODES = 50

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Create vectorized training envs
    env_fns = [make_env(i) for i in range(NUM_CPU)]
    vec_env = SubprocVecEnv(env_fns)
    # Wrap with VecMonitor so that episode rewards/lengths are tracked in the vectorized env
    vec_env = VecMonitor(vec_env)

    # IMPORTANT: Use VecNormalize for observation normalization.
    # We normalize observations (norm_obs=True). We keep norm_reward=False by default.
    # Note: When saving/loading models you must also save/load the VecNormalize wrapper.
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    print(f"{NUM_CPU} environnements vectorisés créés et normalisés (VecNormalize).")

    # PPO hyperparameters — tuned for stability with multiple envs (feel free to adjust)
    ppo_params = {
        'learning_rate': 3e-4,
        'n_steps': 4096,          # must be divisible by num_envs (4096 / 8 = 512)
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,         # reduce entropy to stabilize policy (was 0.05)
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'tensorboard_log': LOG_DIR,
        'device': 'auto'
    }

    # Create or load the model
    model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.zip")
    if os.path.exists(model_path):
        print(f"Chargement du modèle existant: {model_path}")
        # When loading, pass the vec_env and load normalization separately if you saved VecNormalize
        model = PPO.load(model_path, env=vec_env)
    else:
        print("Création d'un nouveau modèle PPO...")
        model = PPO('MlpPolicy', vec_env, verbose=1, **ppo_params)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // NUM_CPU, 1),  # approximate number of steps (env steps -> freq scaled)
        save_path=MODEL_DIR,
        name_prefix=MODEL_NAME
    )

    # Evaluation environment (deterministic evaluation)
    # Must match training env wrapper structure: VecMonitor(VecNormalize(VecEnv))
    def make_eval_env():
        env = gym.make("Parking-v0")
        return env
    
    eval_vec = DummyVecEnv([make_eval_env])
    eval_vec = VecMonitor(eval_vec)  # Add VecMonitor to match training
    # Use VecNormalize for eval (will sync with training stats)
    eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    eval_callback = EvalCallback(
        eval_vec,                 # Use the normalized eval env
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False
    )

    print("\n" + "="*80)
    print("CONFIGURATION D'ENTRAÎNEMENT (LIDAR)")
    print("="*80)
    print(f"✅ Agent PPO (adjusted hyperparams)")
    print(f"✅ Environnements parallèles: {NUM_CPU}")
    print(f"✅ Observation normalization: VecNormalize (clip_obs=10.0)")
    print(f"✅ Evaluations every {EVAL_FREQ} steps ({EVAL_EPISODES} episodes)")
    print(f"✅ Entraînement pour {TOTAL_TIMESTEPS} timesteps")
    print("="*80 + "\n")

    print("Début de l'entraînement...")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur.")

    # Save final model AND the VecNormalize wrapper state (important!)
    final_model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_final.zip")
    model.save(final_model_path)
    print(f"Modèle final sauvegardé sous: {final_model_path}")

    # Save VecNormalize statistics so you can normalize at inference / evaluation time
    vecnorm_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_vecnormalize.npy")
    try:
        # VecNormalize has save/load only on the wrapper level via pickle; here we use its save method
        vec_env.save(vecnorm_path)
        print(f"VecNormalize stats sauvegardés sous: {vecnorm_path}")
    except Exception as e:
        print("Impossible de sauvegarder VecNormalize (peut dépendre de la SB3 version):", e)

    vec_env.close()
    eval_vec.close()