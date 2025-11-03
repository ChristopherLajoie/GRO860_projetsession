"""
Enhanced logger for reward components and performance metrics.
Keeps the same JSON structure but adds useful metrics for:
- Understanding agent behavior
- Tuning the reward function
- Analyzing failure patterns
"""
import json
import os
import numpy as np
from datetime import datetime

class EnhancedRewardLogger:
    def __init__(self, log_dir="reward_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_log = os.path.join(log_dir, f"rewards_{timestamp}.json")
        self.summary_log = os.path.join(log_dir, f"summary_{timestamp}.txt")
        
        self.episodes = []
        self.current_episode = []
        self.episode_metrics = {}
        
    def start_episode(self, initial_info=None):
        """Start tracking a new episode"""
        self.current_episode = []
        self.episode_metrics = {
            'distances': [],
            'angles': [],
            'velocities': [],
            'min_lidar_distances': [],
            'positions': [],
            'total_distance_traveled': 0.0,
            'direction_changes': 0,
            'time_in_reverse': 0,
            'close_calls': 0,
            'reward_components_sum': {},
            'initial_distance': None,
            'initial_angle': None,
            'final_distance': None,
            'final_angle': None,
        }
        
        if initial_info and 'state' in initial_info:
            state = initial_info['state']
            target_x = initial_info.get('target_x', 0)
            target_y = initial_info.get('target_y', 0)
            dx = target_x - state[0]
            dy = target_y - state[1]
            self.episode_metrics['initial_distance'] = float(np.sqrt(dx**2 + dy**2))
    
    def log_step(self, step_num, obs, action, reward, info):
        """Log a single step with reward components and performance metrics"""
        # Extract reward components
        components = {}
        for key in info.keys():
            if key.startswith('reward_'):
                comp_name = key.replace('reward_', '')
                value = info[key]
                components[comp_name] = float(value) if value is not None else 0.0
                
                # Accumulate component totals
                if comp_name not in self.episode_metrics['reward_components_sum']:
                    self.episode_metrics['reward_components_sum'][comp_name] = 0.0
                self.episode_metrics['reward_components_sum'][comp_name] += components[comp_name]
        
        # Extract state information for metrics
        if 'state' in info:
            state = info['state']
            car_x, car_y, car_theta, car_v = state
            
            # Track position for distance traveled calculation
            self.episode_metrics['positions'].append([float(car_x), float(car_y)])
            
            # Calculate distance to target
            target_x = info.get('target_x', car_x)
            target_y = info.get('target_y', car_y)
            distance = float(np.sqrt((car_x - target_x)**2 + (car_y - target_y)**2))
            self.episode_metrics['distances'].append(distance)
            
            # Track angle error
            if 'angle_error' in info:
                angle_error = abs(float(info['angle_error']))
                self.episode_metrics['angles'].append(angle_error)
            
            # Track velocity
            velocity = float(abs(car_v))
            self.episode_metrics['velocities'].append(velocity)
            
            # Count reverse time
            if car_v < -0.1:
                self.episode_metrics['time_in_reverse'] += 1
            
            # Detect direction changes (compare with previous step)
            if len(self.current_episode) > 0:
                prev_v = self.current_episode[-1].get('velocity', 0)
                if prev_v * car_v < -0.1:  # Sign change
                    self.episode_metrics['direction_changes'] += 1
        
        # Track lidar safety
        if 'lidar_distances' in info:
            lidar_dists = info['lidar_distances']
            min_lidar = float(np.min(lidar_dists))
            self.episode_metrics['min_lidar_distances'].append(min_lidar)
            
            # Count close calls (obstacles within 2 meters)
            if min_lidar < 2.0:
                self.episode_metrics['close_calls'] += 1
        
        # Store step data
        step_data = {
            'step': int(step_num),
            'total_reward': float(reward),
            'components': components,
            'distance_to_target': self.episode_metrics['distances'][-1] if self.episode_metrics['distances'] else None,
            'velocity': velocity if 'state' in info else None,
        }
        
        self.current_episode.append(step_data)
    
    def end_episode(self, result, total_reward):
        """Finalize episode and compute summary metrics"""
        # Calculate total distance traveled
        positions = self.episode_metrics['positions']
        if len(positions) > 1:
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                self.episode_metrics['total_distance_traveled'] += np.sqrt(dx**2 + dy**2)
        
        # Final metrics
        if self.episode_metrics['distances']:
            self.episode_metrics['final_distance'] = float(self.episode_metrics['distances'][-1])
            self.episode_metrics['min_distance'] = float(min(self.episode_metrics['distances']))
            self.episode_metrics['avg_distance'] = float(np.mean(self.episode_metrics['distances']))
            
            # Distance improvement
            if self.episode_metrics['initial_distance']:
                improvement = self.episode_metrics['initial_distance'] - self.episode_metrics['final_distance']
                self.episode_metrics['distance_improvement'] = float(improvement)
        
        if self.episode_metrics['angles']:
            self.episode_metrics['final_angle'] = float(self.episode_metrics['angles'][-1])
            self.episode_metrics['avg_angle_error'] = float(np.mean(self.episode_metrics['angles']))
        
        if self.episode_metrics['velocities']:
            self.episode_metrics['max_velocity'] = float(max(self.episode_metrics['velocities']))
            self.episode_metrics['avg_velocity'] = float(np.mean(self.episode_metrics['velocities']))
        
        if self.episode_metrics['min_lidar_distances']:
            self.episode_metrics['min_lidar_observed'] = float(min(self.episode_metrics['min_lidar_distances']))
            self.episode_metrics['avg_min_lidar'] = float(np.mean(self.episode_metrics['min_lidar_distances']))
        
        # Path efficiency (actual distance / initial straight-line distance)
        if self.episode_metrics['initial_distance'] and self.episode_metrics['initial_distance'] > 0:
            efficiency = self.episode_metrics['initial_distance'] / max(self.episode_metrics['total_distance_traveled'], 0.01)
            self.episode_metrics['path_efficiency'] = float(efficiency)
        
        # Clean up large lists (keep only summary stats)
        del self.episode_metrics['positions']
        del self.episode_metrics['distances']
        del self.episode_metrics['angles']
        del self.episode_metrics['velocities']
        del self.episode_metrics['min_lidar_distances']
        
        # Store episode
        episode_data = {
            'result': str(result),
            'total_reward': float(total_reward),
            'steps': self.current_episode,
            'metrics': self.episode_metrics
        }
        
        self.episodes.append(episode_data)
    
    def generate_final_summary(self):
        """Generate JSON log and text summary"""
        # Save JSON
        with open(self.json_log, 'w', encoding='utf-8') as f:
            json.dump({'episodes': self.episodes}, f, indent=2)
        
        # Generate text summary
        self._generate_text_summary()
        
        print(f"\n✅ Logs saved:")
        print(f"   JSON: {self.json_log}")
        print(f"   Summary: {self.summary_log}")
    
    def _generate_text_summary(self):
        """Generate human-readable summary for reward tuning"""
        with open(self.summary_log, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("REWARD FUNCTION ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # Episode results
            f.write("EPISODE RESULTS\n")
            f.write("-"*80 + "\n")
            successes = sum(1 for e in self.episodes if e['result'] == "SUCCESS")
            collisions = sum(1 for e in self.episodes if e['result'] == "COLLISION")
            oob = sum(1 for e in self.episodes if e['result'] == "OUT_OF_BOUNDS")
            timeouts = sum(1 for e in self.episodes if e['result'] == "TIMEOUT")
            total = len(self.episodes)
            
            f.write(f"Total Episodes: {total}\n")
            f.write(f"✅ Successes: {successes} ({successes/total*100:.1f}%)\n")
            f.write(f"❌ Collisions: {collisions} ({collisions/total*100:.1f}%)\n")
            f.write(f"⚠️  Out of Bounds: {oob} ({oob/total*100:.1f}%)\n")
            f.write(f"⏱️  Timeouts: {timeouts} ({timeouts/total*100:.1f}%)\n\n")
            
            # Reward component analysis
            f.write("REWARD COMPONENTS ANALYSIS (for tuning)\n")
            f.write("-"*80 + "\n")
            
            # Average contribution of each component
            all_components = set()
            for ep in self.episodes:
                all_components.update(ep['metrics']['reward_components_sum'].keys())
            
            f.write("\nAverage contribution per episode (across all episodes):\n")
            for comp in sorted(all_components):
                values = [ep['metrics']['reward_components_sum'].get(comp, 0) for ep in self.episodes]
                avg = np.mean(values)
                std = np.std(values)
                f.write(f"  {comp:25s}: {avg:8.2f} ± {std:.2f}\n")
            
            # Success vs Failure comparison
            success_episodes = [e for e in self.episodes if e['result'] == "SUCCESS"]
            failure_episodes = [e for e in self.episodes if e['result'] != "SUCCESS"]
            
            if success_episodes and failure_episodes:
                f.write("\nComponent comparison (Success vs Failure):\n")
                f.write(f"{'Component':<25s} {'Success':>12s} {'Failure':>12s} {'Difference':>12s}\n")
                f.write("-"*65 + "\n")
                
                for comp in sorted(all_components):
                    success_vals = [e['metrics']['reward_components_sum'].get(comp, 0) for e in success_episodes]
                    failure_vals = [e['metrics']['reward_components_sum'].get(comp, 0) for e in failure_episodes]
                    
                    success_avg = np.mean(success_vals) if success_vals else 0
                    failure_avg = np.mean(failure_vals) if failure_vals else 0
                    diff = success_avg - failure_avg
                    
                    f.write(f"{comp:<25s} {success_avg:12.2f} {failure_avg:12.2f} {diff:12.2f}\n")
            
            # Performance metrics
            f.write("\n\nPERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            
            metrics_to_report = [
                ('min_distance', 'Minimum Distance to Target (m)'),
                ('final_distance', 'Final Distance (m)'),
                ('distance_improvement', 'Distance Improvement (m)'),
                ('avg_velocity', 'Average Velocity (m/s)'),
                ('max_velocity', 'Max Velocity (m/s)'),
                ('path_efficiency', 'Path Efficiency (higher=better)'),
                ('direction_changes', 'Direction Changes'),
                ('time_in_reverse', 'Steps in Reverse'),
                ('close_calls', 'Close Calls (<2m)'),
                ('min_lidar_observed', 'Closest Obstacle (m)'),
            ]
            
            f.write("\nOverall averages:\n")
            for metric_key, metric_name in metrics_to_report:
                values = [e['metrics'].get(metric_key) for e in self.episodes if e['metrics'].get(metric_key) is not None]
                if values:
                    avg = np.mean(values)
                    f.write(f"  {metric_name:40s}: {avg:.2f}\n")
            
            # Success-specific metrics
            if success_episodes:
                f.write("\n\nSUCCESS EPISODES ANALYSIS:\n")
                f.write("-"*80 + "\n")
                for metric_key, metric_name in metrics_to_report:
                    values = [e['metrics'].get(metric_key) for e in success_episodes if e['metrics'].get(metric_key) is not None]
                    if values:
                        avg = np.mean(values)
                        f.write(f"  {metric_name:40s}: {avg:.2f}\n")
            
            # Failure analysis
            if failure_episodes:
                f.write("\n\nFAILURE EPISODES ANALYSIS:\n")
                f.write("-"*80 + "\n")
                
                # Breakdown by failure type
                f.write("Failure breakdown:\n")
                for failure_type in ["COLLISION", "OUT_OF_BOUNDS", "TIMEOUT"]:
                    type_episodes = [e for e in failure_episodes if e['result'] == failure_type]
                    if type_episodes:
                        f.write(f"\n{failure_type}:\n")
                        for metric_key, metric_name in metrics_to_report:
                            values = [e['metrics'].get(metric_key) for e in type_episodes if e['metrics'].get(metric_key) is not None]
                            if values:
                                avg = np.mean(values)
                                f.write(f"  {metric_name:40s}: {avg:.2f}\n")
            
            # Recommendations
            f.write("\n\nRECOMMENDATIONS FOR REWARD TUNING:\n")
            f.write("-"*80 + "\n")
            self._generate_recommendations(f)
            
            f.write("\n" + "="*80 + "\n")
    
    def _generate_recommendations(self, f):
        """Generate automatic recommendations based on metrics"""
        if not self.episodes:
            f.write("Not enough data for recommendations.\n")
            return
        
        successes = [e for e in self.episodes if e['result'] == "SUCCESS"]
        collisions = [e for e in self.episodes if e['result'] == "COLLISION"]
        
        success_rate = len(successes) / len(self.episodes) * 100
        collision_rate = len(collisions) / len(self.episodes) * 100
        
        # Success rate analysis
        if success_rate < 30:
            f.write("⚠️  LOW SUCCESS RATE:\n")
            f.write("   - Consider increasing progress reward scale\n")
            f.write("   - Check if success conditions are too strict\n")
            f.write("   - May need more training time\n\n")
        elif success_rate > 80:
            f.write("✅ GOOD SUCCESS RATE\n\n")
        
        # Collision analysis
        if collision_rate > 30:
            f.write("⚠️  HIGH COLLISION RATE:\n")
            f.write("   - Increase safety penalty (safety_scale)\n")
            f.write("   - Increase collision penalty\n")
            f.write("   - Check lidar readings are working properly\n\n")
        
        # Close calls analysis
        close_calls = [e['metrics'].get('close_calls', 0) for e in self.episodes]
        avg_close_calls = np.mean(close_calls) if close_calls else 0
        if avg_close_calls > 20:
            f.write("⚠️  MANY CLOSE CALLS:\n")
            f.write("   - Agent is taking risky paths\n")
            f.write("   - Consider increasing safety_threshold\n\n")
        
        # Path efficiency
        efficiencies = [e['metrics'].get('path_efficiency', 0) for e in self.episodes if e['metrics'].get('path_efficiency')]
        if efficiencies:
            avg_efficiency = np.mean(efficiencies)
            if avg_efficiency < 0.3:
                f.write("⚠️  LOW PATH EFFICIENCY:\n")
                f.write("   - Agent is taking very indirect routes\n")
                f.write("   - Progress reward may not be strong enough\n\n")
        
        # Velocity analysis
        velocities = [e['metrics'].get('avg_velocity', 0) for e in self.episodes if e['metrics'].get('avg_velocity')]
        if velocities:
            avg_vel = np.mean(velocities)
            if avg_vel < 0.5:
                f.write("⚠️  VERY LOW VELOCITY:\n")
                f.write("   - Agent is moving too cautiously\n")
                f.write("   - May need to reduce step penalty\n\n")
            elif avg_vel > 3.0:
                f.write("⚠️  HIGH VELOCITY:\n")
                f.write("   - Agent moving too fast (dangerous)\n")
                f.write("   - Increase velocity_near_goal_penalty\n\n")
    
    def print_summary(self):
        """Print quick summary to console"""
        if not self.episodes:
            print("No episodes recorded.")
            return
        
        print(f"\n{'='*80}")
        print(f"Quick Summary: {len(self.episodes)} episodes")
        print(f"{'='*80}")
        
        successes = sum(1 for e in self.episodes if e['result'] == "SUCCESS")
        print(f"✅ Success Rate: {successes}/{len(self.episodes)} ({successes/len(self.episodes)*100:.1f}%)")
        
        avg_reward = np.mean([e['total_reward'] for e in self.episodes])
        print(f"📊 Average Reward: {avg_reward:.1f}")
        
        print(f"\n📁 Full analysis saved to: {self.summary_log}")
        print(f"{'='*80}\n")