import numpy as np
import json
from collections import deque
from datetime import datetime

class PerformanceTracker:
    """
    Tracker de performance compact pour l'environnement Parking
    Format de sortie: une ligne par batch de 100 épisodes
    """
    
    def __init__(self, log_file="parking_performance.txt", batch_size=100):
        self.log_file = log_file
        self.batch_size = batch_size
        
        # Métriques du batch actuel
        self.current_batch = 0
        self.episodes_in_batch = 0
        
        # Compteurs pour le batch
        self.successes = 0
        self.collisions = 0
        self.timeouts = 0
        self.out_of_bounds = 0
        
        # Listes pour moyennes
        self.min_distances = []
        self.episode_steps = []
        self.total_rewards = []
        self.max_zones = []  # Zone max atteinte par épisode
        self.zone_times = {1: [], 2: [], 3: [], 4: []}  # Temps passé dans chaque zone
        
        # Comportement
        self.reverse_usage = []  # % de temps en marche arrière
        self.avg_speeds = []
        self.immobility_times = []  # % de temps immobile
        
        # Buffer pour les derniers N épisodes (pour détecter les tendances)
        self.recent_success_rate = deque(maxlen=500)
        
        # Initialiser le fichier avec header
        with open(self.log_file, 'w') as f:
            f.write(f"# Performance Tracking - Démarré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Format: Batch X-Y: Success=% Collision=% MinDist=Xm Zone4=% AvgSteps=X AvgSpeed=Xm/s Reverse=% Trend=X\n")
            f.write("#" + "="*100 + "\n")
    
    def reset_episode(self):
        """Appelé au début de chaque épisode"""
        self.current_episode_data = {
            'steps': 0,
            'min_dist': float('inf'),
            'max_zone': 1,
            'zone_visits': {1: 0, 2: 0, 3: 0, 4: 0},
            'reverse_steps': 0,
            'immobile_steps': 0,
            'speeds': [],
            'total_reward': 0,
            'terminated': False,
            'collision': False,
            'out_of_bounds': False,
            'success': False
        }
    
    def step(self, state, reward, terminated, truncated, info):
        """Appelé à chaque step de l'environnement"""
        if not hasattr(self, 'current_episode_data'):
            self.reset_episode()
        
        data = self.current_episode_data
        data['steps'] += 1
        data['total_reward'] += reward
        
        # Position et distance
        if 'state' in info:
            car_x, car_y, car_theta, car_v = info['state']
            
            # Distance au target (supposons target à (30, 0) par défaut)
            # À ajuster selon votre environnement
            target_x, target_y = 30.0, 0.0
            if 'target_x' in info:
                target_x, target_y = info['target_x'], info['target_y']
            
            dist = np.sqrt((car_x - target_x)**2 + (car_y - target_y)**2)
            data['min_dist'] = min(data['min_dist'], dist)
            
            # Zone actuelle
            zone = self._get_zone(dist)
            data['max_zone'] = max(data['max_zone'], zone)
            data['zone_visits'][zone] += 1
            
            # Vitesse et comportement
            data['speeds'].append(abs(car_v))
            if car_v < -0.1:
                data['reverse_steps'] += 1
            if abs(car_v) < 0.1:
                data['immobile_steps'] += 1
        
        # Détection du type de terminaison
        if terminated or truncated:
            if 'reward_collision' in info and info['reward_collision'] < 0:
                data['collision'] = True
            elif 'reward_out_of_bounds' in info and info['reward_out_of_bounds'] < 0:
                data['out_of_bounds'] = True
            elif 'reward_success' in info and info['reward_success'] > 0:
                data['success'] = True
            data['terminated'] = terminated or truncated
    
    def end_episode(self):
        """Appelé à la fin de chaque épisode"""
        if not hasattr(self, 'current_episode_data'):
            return
        
        data = self.current_episode_data
        self.episodes_in_batch += 1
        
        # Mise à jour des compteurs
        if data['success']:
            self.successes += 1
            self.recent_success_rate.append(1)
        else:
            self.recent_success_rate.append(0)
            
        if data['collision']:
            self.collisions += 1
        elif data['out_of_bounds']:
            self.out_of_bounds += 1
        elif not data['success'] and data['terminated']:
            self.timeouts += 1
        
        # Métriques
        self.min_distances.append(data['min_dist'])
        self.episode_steps.append(data['steps'])
        self.total_rewards.append(data['total_reward'])
        self.max_zones.append(data['max_zone'])
        
        # Temps dans chaque zone (en %)
        total_steps = max(data['steps'], 1)
        for zone in range(1, 5):
            zone_pct = (data['zone_visits'][zone] / total_steps) * 100
            self.zone_times[zone].append(zone_pct)
        
        # Comportement
        if data['steps'] > 0:
            self.reverse_usage.append((data['reverse_steps'] / data['steps']) * 100)
            self.immobility_times.append((data['immobile_steps'] / data['steps']) * 100)
        
        if data['speeds']:
            self.avg_speeds.append(np.mean(data['speeds']))
        
        # Logging si batch complet
        if self.episodes_in_batch >= self.batch_size:
            self._log_batch()
            self._reset_batch()
    
    def _get_zone(self, distance):
        """Détermine la zone basée sur la distance"""
        if distance > 20:
            return 1
        elif distance > 10:
            return 2
        elif distance > 5:
            return 3
        else:
            return 4
    
    def _log_batch(self):
        """Écrit les statistiques du batch dans le fichier"""
        batch_start = self.current_batch * self.batch_size
        batch_end = batch_start + self.episodes_in_batch - 1
        
        # Calcul des moyennes
        success_rate = (self.successes / self.episodes_in_batch) * 100
        collision_rate = (self.collisions / self.episodes_in_batch) * 100
        timeout_rate = (self.timeouts / self.episodes_in_batch) * 100
        oob_rate = (self.out_of_bounds / self.episodes_in_batch) * 100
        
        avg_min_dist = np.mean(self.min_distances) if self.min_distances else 999
        avg_steps = np.mean(self.episode_steps) if self.episode_steps else 0
        avg_speed = np.mean(self.avg_speeds) if self.avg_speeds else 0
        
        # Zone 4 atteinte (%)
        zone4_reached = sum(1 for z in self.max_zones if z == 4) / len(self.max_zones) * 100 if self.max_zones else 0
        
        # Utilisation marche arrière
        avg_reverse = np.mean(self.reverse_usage) if self.reverse_usage else 0
        
        # Calcul de la tendance
        if len(self.recent_success_rate) >= 200:
            recent_100 = list(self.recent_success_rate)[-100:]
            previous_100 = list(self.recent_success_rate)[-200:-100]
            recent_avg = np.mean(recent_100) * 100
            previous_avg = np.mean(previous_100) * 100
            trend_value = recent_avg - previous_avg
            if trend_value > 2:
                trend = f"↑{trend_value:+.1f}%"
            elif trend_value < -2:
                trend = f"↓{trend_value:+.1f}%"
            else:
                trend = "→"
        else:
            trend = "NEW"
        
        # Format compact sur une ligne
        log_line = (f"Batch {batch_start:04d}-{batch_end:04d}: "
                   f"Success={success_rate:5.1f}% "
                   f"Collision={collision_rate:5.1f}% "
                   f"Timeout={timeout_rate:5.1f}% "
                   f"OOB={oob_rate:5.1f}% "
                   f"MinDist={avg_min_dist:5.1f}m "
                   f"Zone4={zone4_reached:5.1f}% "
                   f"Steps={avg_steps:4.0f} "
                   f"Speed={avg_speed:3.1f}m/s "
                   f"Reverse={avg_reverse:4.1f}% "
                   f"Trend={trend}")
        
        # Écriture dans le fichier
        with open(self.log_file, 'a') as f:
            f.write(log_line + "\n")
        
        # Print aussi dans la console pour feedback immédiat
        print(log_line)
        
        # Rapport détaillé tous les 10 batches (1000 épisodes)
        if (self.current_batch + 1) % 10 == 0:
            self._log_detailed_report()
    
    def _log_detailed_report(self):
        """Rapport détaillé tous les 1000 épisodes"""
        with open(self.log_file, 'a') as f:
            f.write("\n" + "="*100 + "\n")
            f.write(f"=== CHECKPOINT {(self.current_batch + 1) * self.batch_size} ÉPISODES ===\n")
            
            # Taux de succès sur les derniers 1000
            if len(self.recent_success_rate) >= 100:
                recent_success = np.mean(list(self.recent_success_rate)[-100:]) * 100
                f.write(f"Taux de succès récent (100 derniers): {recent_success:.1f}%\n")
            
            # Distribution des zones
            zones_distribution = []
            for zone in range(1, 5):
                zone_count = sum(1 for z in self.max_zones[-1000:] if z == zone)
                zones_distribution.append(f"Zone{zone}={zone_count/10:.0f}%")
            f.write(f"Distribution zones max: {' '.join(zones_distribution)}\n")
            
            # Problèmes détectés
            avg_immobility = np.mean(self.immobility_times[-100:]) if len(self.immobility_times) >= 100 else 0
            if avg_immobility > 30:
                f.write(f"⚠️ Problème détecté: Immobilité excessive ({avg_immobility:.1f}% du temps)\n")
            
            if recent_success < 5 and (self.current_batch + 1) > 20:
                f.write("⚠️ Problème détecté: Taux de succès très faible après 2000+ épisodes\n")
            
            f.write("="*100 + "\n\n")
    
    def _reset_batch(self):
        """Réinitialise les compteurs pour le prochain batch"""
        self.current_batch += 1
        self.episodes_in_batch = 0
        self.successes = 0
        self.collisions = 0
        self.timeouts = 0
        self.out_of_bounds = 0
        self.min_distances = []
        self.episode_steps = []
        self.total_rewards = []
        self.max_zones = []
        self.zone_times = {1: [], 2: [], 3: [], 4: []}
        self.reverse_usage = []
        self.avg_speeds = []
        self.immobility_times = []
    
    def get_summary(self):
        """Retourne un résumé des performances actuelles"""
        if len(self.recent_success_rate) > 0:
            recent_success = np.mean(list(self.recent_success_rate)[-min(100, len(self.recent_success_rate)):]) * 100
        else:
            recent_success = 0
        
        return {
            'total_episodes': self.current_batch * self.batch_size + self.episodes_in_batch,
            'recent_success_rate': recent_success,
            'current_batch': self.current_batch
        }