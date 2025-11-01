import gymnasium as gym
import numpy as np
from parking import ParkingEnv
from performance_tracker import PerformanceTracker
from stable_baselines3 import PPO
import pygame
import csv
from datetime import datetime

# --- Wrappers (MIS À JOUR pour 25 observations) ---
class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        unwrapped_env = env.unwrapped

        # 25 observations
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
        
        max_dist = np.sqrt((2*world_w)**2 + (2*world_h)**2)
        
        high = np.array(
            [world_w, world_h, np.pi, max_v,
             world_w, world_h,
             1.0, 1.0,
             np.pi,
             world_w, world_h, world_w, world_h,
             world_w, world_h, world_w, world_h,
             max_dist, max_dist, max_dist, max_dist,
             1.0, 1.0, 1.0, 1.0],
            dtype=np.float32
        )

        self.obs_low = low
        self.obs_high = high
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(25,), dtype=np.float32)

    def observation(self, obs):
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

# --- Fonction de test avec visualisation ---
def test_single_episode(agent, env, render=True, save_video=False):
    """Test un seul épisode avec visualisation optionnelle"""
    obs, info = env.reset()
    
    # Données de départ
    start_state = info.get("start_state", np.zeros(4))
    
    clock = pygame.time.Clock() if render else None
    fps = env.metadata["render_fps"]

    term = trunc = quit_loop = False
    step = 0
    total_reward = 0
    episode_data = {
        'steps': [],
        'rewards': [],
        'positions': [],
        'actions': [],
        'min_dist': float('inf'),
        'max_zone': 1
    }

    while not (term or trunc or quit_loop):
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_loop = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    quit_loop = True

        action = agent(obs)
        obs, reward, term, trunc, info = env.step(action)
        
        if render:
            env.render()

        step += 1
        total_reward += reward
        
        # Enregistrer les données
        if 'state' in info:
            car_x, car_y, car_theta, car_v = info['state']
            target_x, target_y = 30.0, 0.0  # À ajuster selon votre environnement
            
            dist = np.sqrt((car_x - target_x)**2 + (car_y - target_y)**2)
            episode_data['min_dist'] = min(episode_data['min_dist'], dist)
            
            # Zone
            if dist > 20:
                zone = 1
            elif dist > 10:
                zone = 2
            elif dist > 5:
                zone = 3
            else:
                zone = 4
            episode_data['max_zone'] = max(episode_data['max_zone'], zone)
            
            episode_data['steps'].append(step)
            episode_data['rewards'].append(reward)
            episode_data['positions'].append((car_x, car_y))
            episode_data['actions'].append(action)

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
    elif trunc:
        result = "TIMEOUT"
    
    episode_data['result'] = result
    episode_data['total_reward'] = total_reward
    episode_data['total_steps'] = step
    
    if render:
        env.close()
    
    return episode_data

# --- Fonction de test en batch ---
def test_batch(agent, env_id, n_episodes=100, render_first=5):
    """Test plusieurs épisodes pour évaluation statistique"""
    
    print(f"\n{'='*80}")
    print(f"ÉVALUATION SUR {n_episodes} ÉPISODES")
    print(f"{'='*80}")
    
    results = []
    tracker = PerformanceTracker(log_file="test_performance.txt", batch_size=10)
    
    for episode_idx in range(n_episodes):
        # Créer un nouvel environnement pour chaque épisode
        env = gym.make(env_id, render_mode="human" if episode_idx < render_first else None)
        env = NormalizeObservation(env)
        env = NormalizeAction(env)
        
        # Test de l'épisode
        episode_data = test_single_episode(
            agent, 
            env, 
            render=(episode_idx < render_first),
            save_video=False
        )
        
        results.append(episode_data)
        
        # Logger
        tracker.reset_episode()
        for step in range(episode_data['total_steps']):
            tracker.step(
                state=[0, 0, 0, 0],  # Simplifié pour le test
                reward=episode_data['rewards'][step] if step < len(episode_data['rewards']) else 0,
                terminated=(episode_data['result'] != "TIMEOUT"),
                truncated=(episode_data['result'] == "TIMEOUT"),
                info={'state': [0, 0, 0, 0]}
            )
        tracker.end_episode()
        
        # Affichage périodique
        if (episode_idx + 1) % 10 == 0:
            successes = sum(1 for r in results if r['result'] == "SUCCESS")
            collisions = sum(1 for r in results if r['result'] == "COLLISION")
            avg_min_dist = np.mean([r['min_dist'] for r in results])
            zone4_reached = sum(1 for r in results if r['max_zone'] == 4)
            
            print(f"\nÉpisodes {episode_idx+1}/{n_episodes}:")
            print(f"  Succès: {successes}/{episode_idx+1} ({successes/(episode_idx+1)*100:.1f}%)")
            print(f"  Collisions: {collisions}/{episode_idx+1} ({collisions/(episode_idx+1)*100:.1f}%)")
            print(f"  Distance min moyenne: {avg_min_dist:.2f}m")
            print(f"  Zone 4 atteinte: {zone4_reached}/{episode_idx+1} ({zone4_reached/(episode_idx+1)*100:.1f}%)")
        
        env.close()
    
    # Résumé final
    print(f"\n{'='*80}")
    print("RÉSUMÉ FINAL")
    print(f"{'='*80}")
    
    successes = sum(1 for r in results if r['result'] == "SUCCESS")
    collisions = sum(1 for r in results if r['result'] == "COLLISION")
    oob = sum(1 for r in results if r['result'] == "OUT_OF_BOUNDS")
    timeouts = sum(1 for r in results if r['result'] == "TIMEOUT")
    
    print(f"Résultats sur {n_episodes} épisodes:")
    print(f"  ✅ Succès: {successes} ({successes/n_episodes*100:.1f}%)")
    print(f"  ❌ Collisions: {collisions} ({collisions/n_episodes*100:.1f}%)")
    print(f"  ⚠️  Hors limites: {oob} ({oob/n_episodes*100:.1f}%)")
    print(f"  ⏱️  Timeouts: {timeouts} ({timeouts/n_episodes*100:.1f}%)")
    
    avg_min_dist = np.mean([r['min_dist'] for r in results])
    avg_steps = np.mean([r['total_steps'] for r in results])
    avg_reward = np.mean([r['total_reward'] for r in results])
    
    print(f"\nMétriques moyennes:")
    print(f"  Distance minimale: {avg_min_dist:.2f}m")
    print(f"  Steps par épisode: {avg_steps:.1f}")
    print(f"  Récompense totale: {avg_reward:.1f}")
    
    zones = [r['max_zone'] for r in results]
    for zone in range(1, 5):
        count = sum(1 for z in zones if z >= zone)
        print(f"  Zone {zone} atteinte: {count}/{n_episodes} ({count/n_episodes*100:.1f}%)")
    
    print(f"{'='*80}\n")
    
    return results

# --- Main Testing Logic ---
if __name__ == "__main__":
    
    # 1. Register the environment
    gym.register(
        id="Parking-v0",
        entry_point="parking:ParkingEnv",
        max_episode_steps=1500
    )

    # 2. Choix du mode de test
    print("\n" + "="*80)
    print("TEST DE L'AGENT DE PARKING")
    print("="*80)
    print("Modes disponibles:")
    print("1. Test visuel (1 épisode avec rendu)")
    print("2. Évaluation statistique (100 épisodes)")
    print("3. Les deux")
    
    try:
        mode = input("\nChoisissez le mode (1/2/3) [défaut: 1]: ").strip() or "1"
    except KeyboardInterrupt:
        mode = "1"
    
    # 3. Charger le modèle
    model_load_path = "./ppo_parking_model_improved.zip"
    try:
        # Créer un environnement temporaire pour le chargement
        temp_env = gym.make("Parking-v0")
        temp_env = NormalizeObservation(temp_env)
        temp_env = NormalizeAction(temp_env)
        
        model = PPO.load(model_load_path, env=temp_env)
        print(f"✅ Modèle chargé depuis : {model_load_path}")
        temp_env.close()
    except FileNotFoundError:
        print(f"❌ ERREUR: Fichier modèle non trouvé à {model_load_path}")
        
        # Essayer l'ancien modèle
        old_model_path = "./ppo_parking_model.zip"
        try:
            print(f"Tentative de chargement de l'ancien modèle : {old_model_path}")
            temp_env = gym.make("Parking-v0")
            temp_env = NormalizeObservation(temp_env)
            temp_env = NormalizeAction(temp_env)
            
            model = PPO.load(old_model_path, env=temp_env)
            print(f"✅ Ancien modèle chargé")
            print("⚠️  NOTE: Ce modèle a été entraîné avec l'ancienne configuration")
            temp_env.close()
        except:
            print("❌ Aucun modèle trouvé. Lancez d'abord train_parking.py")
            exit()

    # 4. Définir la fonction agent
    def agent(obs):
        action, _states = model.predict(obs, deterministic=True)
        return action

    # 5. Exécuter le test selon le mode choisi
    if mode == "1" or mode == "3":
        # Test visuel simple
        print("\n🎮 Lancement du test visuel...")
        print("Appuyez sur 'Q' pour quitter")
        
        env = gym.make("Parking-v0", render_mode="human")
        env = NormalizeObservation(env)
        env = NormalizeAction(env)
        
        episode_data = test_single_episode(agent, env, render=True)
        
        print(f"\nRésultat: {episode_data['result']}")
        print(f"Steps: {episode_data['total_steps']}")
        print(f"Récompense totale: {episode_data['total_reward']:.2f}")
        print(f"Distance minimale: {episode_data['min_dist']:.2f}m")
        print(f"Zone maximale atteinte: {episode_data['max_zone']}/4")
    
    if mode == "2" or mode == "3":
        # Évaluation statistique
        print("\n📊 Lancement de l'évaluation statistique...")
        results = test_batch(agent, "Parking-v0", n_episodes=100, render_first=0)
        
        # Sauvegarder un résumé
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"test_summary_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("RÉSUMÉ DE TEST - " + timestamp + "\n")
            f.write("="*80 + "\n")
            
            successes = sum(1 for r in results if r['result'] == "SUCCESS")
            collisions = sum(1 for r in results if r['result'] == "COLLISION")
            
            f.write(f"Taux de succès: {successes}/100 = {successes}%\n")
            f.write(f"Taux de collision: {collisions}/100 = {collisions}%\n")
            f.write(f"Distance min moyenne: {np.mean([r['min_dist'] for r in results]):.2f}m\n")
        
        print(f"\n✅ Résumé sauvegardé dans {summary_file}")
        print("📋 Pour partager avec moi, copiez le contenu de 'parking_performance.txt'")