"""
Script pour tester un agent PPO avec logging détaillé des rewards.
Version modifiée pour analyser les calculs de reward.
"""
import gymnasium as gym
import numpy as np
from parking import ParkingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pygame
from datetime import datetime
import os

# Import the enhanced logger
from reward_logger import EnhancedRewardLogger

# --- Wrappers (identiques à test_agent.py) ---
class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.obs_low = self.observation_space.low
        self.obs_high = self.observation_space.high

        expected_shape = 17
        if self.obs_low.shape[0] != expected_shape:
            print(f"ERREUR WRAPPER: Attendu {expected_shape} observations, reçu {self.obs_low.shape[0]}")
        
        self.obs_low[self.obs_low == -np.inf] = -100.0
        self.obs_high[self.obs_high == np.inf] = 100.0
        
        self.range_ = self.obs_high - self.obs_low
        self.range_[self.range_ == 0] = 1.0

    def observation(self, obs):
        return -1.0 + 2.0 * (obs - self.obs_low) / self.range_

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self.observation(obs), reward, terminated, truncated, info

class NormalizeAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32)

    def action(self, act):
        low = self.env.action_space.low
        high = self.env.action_space.high
        return low + (high - low) * (act + 1.0) / 2.0

# --- Fonction de test MODIFIÉE avec logging ---
def test_single_episode_with_logging(agent, env, logger, render=True):
    """Test un épisode avec logging détaillé des rewards"""
    # VecEnv returns only obs, not (obs, info)
    obs = env.reset()
    
    # For VecEnv, we need to get info from the first step or environment attributes
    # Start with empty info
    logger.start_episode(initial_info=None)
    
    clock = pygame.time.Clock() if render else None
    # VecEnv doesn't have metadata directly, use the unwrapped env
    try:
        fps = env.get_attr("metadata")[0]["render_fps"]
    except:
        fps = 30  # default

    done = False
    quit_loop = False
    step = 0
    total_reward = 0
    episode_data = {
        'steps': [],
        'rewards': [],
        'min_dist': float('inf'),
        'max_zone': 1
    }

    while not (done or quit_loop):
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_loop = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    quit_loop = True

        action, _states = agent.predict(obs, deterministic=True)
        # VecEnv step returns obs, rewards, dones, infos (all as arrays)
        obs, rewards, dones, infos = env.step(action)
        
        # Extract first element (we're using a single env)
        reward = rewards[0]
        done = dones[0]
        info = infos[0]
        
        # LOGGING: Enregistrer ce step
        logger.log_step(step, obs[0], action, reward, info)
        
        if render:
            # VecEnv render needs mode specified
            try:
                env.render(mode='human')
            except:
                pass  # Rendering might not work with VecEnv wrapper

        step += 1
        total_reward += reward
        
        if 'state' in info:
            car_x, car_y, car_theta, car_v = info['state']
            target_x = info.get('target_x', car_x)
            target_y = info.get('target_y', car_y)
            
            dist = np.sqrt((car_x - target_x)**2 + (car_y - target_y)**2)
            episode_data['min_dist'] = min(episode_data['min_dist'], dist)
            
            if dist > 20: zone = 1
            elif dist > 10: zone = 2
            elif dist > 5: zone = 3
            else: zone = 4
            episode_data['max_zone'] = max(episode_data['max_zone'], zone)
            
            episode_data['steps'].append(step)
            episode_data['rewards'].append(reward)

        if render and clock:
            clock.tick(fps)
    
    # Déterminer le résultat
    result = "UNKNOWN"
    if info.get("reward_success", 0) > 0:
        result = "SUCCESS"
    elif info.get("reward_collision", 0) < 0:
        result = "COLLISION"
    elif info.get("reward_out_of_bounds", 0) < 0:
        result = "OUT_OF_BOUNDS"
    elif info.get("TimeLimit.truncated", False) or step >= 2000:
        result = "TIMEOUT"
    
    episode_data['result'] = result
    episode_data['total_reward'] = total_reward
    episode_data['total_steps'] = step
    
    # Terminer le logging de l'épisode
    logger.end_episode(result, total_reward)
    
    return episode_data

# --- Fonction de test en batch MODIFIÉE ---
def test_batch_with_logging(agent, env_id, logger, n_episodes=20, render_first=3, vecnormalize_path=None):
    """Test plusieurs épisodes avec logging détaillé"""
    print(f"\n{'='*80}")
    print(f"ÉVALUATION AVEC LOGGING DES REWARDS - {n_episodes} épisodes")
    print(f"{'='*80}")
    
    results = []
    
    for episode_idx in range(n_episodes):
        print(f"\n📍 Épisode {episode_idx+1}/{n_episodes}...")
        
        # Create base environment
        base_env = gym.make(env_id, render_mode="human" if episode_idx < render_first else None)
        
        # Wrap in DummyVecEnv for compatibility (need to use a list with the env)
        # DummyVecEnv expects a list of environment factories
        env = DummyVecEnv([lambda e=base_env: e])
        
        # Apply normalization
        if vecnormalize_path and os.path.exists(vecnormalize_path):
            env = VecNormalize.load(vecnormalize_path, env)
            env.training = False
            env.norm_reward = False
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        
        episode_data = test_single_episode_with_logging(
            agent, 
            env, 
            logger,
            render=(episode_idx < render_first)
        )
        
        results.append(episode_data)
        
        print(f"   Résultat: {episode_data['result']}")
        print(f"   Reward: {episode_data['total_reward']:.2f}")
        print(f"   Steps: {episode_data['total_steps']}")
        print(f"   Distance min: {episode_data['min_dist']:.2f}m")
        
        env.close()
    
    # Résumé final
    print(f"\n{'='*80}")
    print("RÉSUMÉ FINAL")
    print(f"{'='*80}")
    
    successes = sum(1 for r in results if r['result'] == "SUCCESS")
    collisions = sum(1 for r in results if r['result'] == "COLLISION")
    oob = sum(1 for r in results if r['result'] == "OUT_OF_BOUNDS")
    timeouts = sum(1 for r in results if r['result'] == "TIMEOUT")
    
    print(f"\nRésultats sur {n_episodes} épisodes:")
    print(f"  ✅ Succès: {successes} ({successes/n_episodes*100:.1f}%)")
    print(f"  ❌ Collisions: {collisions} ({collisions/n_episodes*100:.1f}%)")
    print(f"  ⚠️  Hors limites: {oob} ({oob/n_episodes*100:.1f}%)")
    print(f"  ⏱️  Timeouts: {timeouts} ({timeouts/n_episodes*100:.1f}%)")
    
    if results:
        avg_reward = np.mean([r['total_reward'] for r in results])
        avg_steps = np.mean([r['total_steps'] for r in results])
        avg_min_dist = np.mean([r['min_dist'] for r in results])
        
        print(f"\nMétriques moyennes:")
        print(f"  Reward totale: {avg_reward:.1f}")
        print(f"  Steps par épisode: {avg_steps:.1f}")
        print(f"  Distance minimale: {avg_min_dist:.2f}m")
    
    print(f"{'='*80}\n")
    
    return results

# --- Main avec options de logging ---
if __name__ == "__main__":
    
    # Enregistrer l'environnement
    try:
        gym.register(
            id="Parking-v0",
            entry_point="parking:ParkingEnv",
            max_episode_steps=2000
        )
    except gym.error.Error:
        pass

    print("\n" + "="*80)
    print("TEST DE L'AGENT AVEC LOGGING DES REWARDS")
    print("="*80)
    print("\nCe script va:")
    print("  1. Tester l'agent sur plusieurs épisodes")
    print("  2. Logger TOUS les composants de reward à chaque step")
    print("  3. Générer des fichiers d'analyse détaillée")
    print("\nFichiers générés dans le dossier 'reward_logs/':")
    print("  - detailed_rewards_*.txt : Log complet step par step")
    print("  - summary_*.txt : Analyse statistique des rewards")
    print("  - rewards_*.json : Données brutes pour analyse ultérieure")
    
    print("\n" + "="*80)
    print("CONFIGURATION DU TEST")
    print("="*80)
    
    try:
        n_episodes = int(input("Nombre d'épisodes à tester [défaut: 10]: ").strip() or "10")
        render_first = int(input("Nombre d'épisodes à visualiser [défaut: 2]: ").strip() or "2")
    except (KeyboardInterrupt, ValueError):
        n_episodes = 10
        render_first = 2
    
    # Charger le modèle
    model_paths = [
        "./models/ppo_parking_lidar_final.zip",
        "./models/ppo_parking_lidar.zip",
        "./ppo_parking_model_improved.zip",
        "./ppo_parking_model.zip"
    ]
    
    vecnormalize_paths = [
        "./models/ppo_parking_lidar_vecnormalize.npy",
        "./models/ppo_parking_lidar_vecnormalize.pkl",
    ]
    
    model = None
    model_loaded = None
    use_vecnormalize = False
    
    # Try to find and load VecNormalize stats first
    vecnormalize_stats = None
    for vn_path in vecnormalize_paths:
        if os.path.exists(vn_path):
            print(f"\n✅ VecNormalize stats trouvées: {vn_path}")
            vecnormalize_stats = vn_path
            use_vecnormalize = True
            break
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                # Create environment
                temp_env = gym.make("Parking-v0")
                
                # Wrap in DummyVecEnv for compatibility
                temp_env = DummyVecEnv([lambda: temp_env])
                
                # Apply normalization
                if use_vecnormalize and vecnormalize_stats:
                    # Load VecNormalize stats (training mode)
                    temp_env = VecNormalize.load(vecnormalize_stats, temp_env)
                    temp_env.training = False  # Don't update stats during testing
                    temp_env.norm_reward = False  # Don't normalize rewards during testing
                    print("✅ Utilisation de VecNormalize (mode test)")
                else:
                    # Fallback to manual normalization
                    temp_env = VecNormalize(temp_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
                    print("✅ Utilisation de VecNormalize (nouvelle normalisation)")
                
                model = PPO.load(model_path, env=temp_env)
                model_loaded = model_path
                print(f"✅ Modèle chargé depuis : {model_path}")
                break
            except Exception as e:
                print(f"❌ Erreur lors du chargement de {model_path}: {e}")
                if temp_env is not None:
                    try:
                        temp_env.close()
                    except:
                        pass
                continue
    
    if model is None:
        print("\n❌ ERREUR: Aucun modèle trouvé.")
        print("Chemins testés:")
        for path in model_paths:
            print(f"  - {path}")
        print("\nLancez d'abord 'python train_parking.py'")
        exit(1)
    
    # Créer le logger
    logger = EnhancedRewardLogger(log_dir="reward_logs")
    
    print("\n" + "="*80)
    print("DÉBUT DU TEST")
    print("="*80)
    
    if render_first > 0:
        print(f"\n🎮 Les {render_first} premiers épisodes seront visualisés")
        print("    Appuyez sur 'Q' pour passer au suivant")
    
    # Lancer les tests
    results = test_batch_with_logging(
        agent=model,
        env_id="Parking-v0",
        logger=logger,
        n_episodes=n_episodes,
        render_first=render_first,
        vecnormalize_path=vecnormalize_stats if use_vecnormalize else None
    )
    
    # Générer le résumé final
    print("\n📊 Génération des fichiers d'analyse...")
    logger.generate_final_summary()
    logger.print_summary()