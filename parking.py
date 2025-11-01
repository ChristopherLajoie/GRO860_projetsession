import gymnasium as gym
import numpy as np
from shapely.geometry import Polygon
import pygame

class ParkingEnv(gym.Env):
    """
    PARKING RÉALISTE AMÉLIORÉ avec:
    - 5 colonnes × 10 voitures (50 voitures)
    - 1 place libre aléatoire
    - Allées 12m + zones circulation
    - Spawn dans allées/circulation (centré et parallèle)
    - Observations 25 éléments (car + target + distances + danger scores + directions obstacles)
    - Système de zones de progression
    - Reward améliorée avec danger awareness
    """
    
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # Constantes véhicule
        self.dt = 1 / self.metadata["render_fps"]
        self.CAR_LENGTH = 4.5
        self.CAR_WIDTH = 2.0
        self.MAX_STEER_ANGLE = np.pi / 6
        self.MAX_ACCEL = 2.0
        self.MAX_VELOCITY = 5.0
        self.VELOCITY_DECAY = 0.99
        
        # Constantes parking
        self.SPOT_WIDTH = self.CAR_WIDTH * 1.5
        self.SPOT_DEPTH = self.CAR_LENGTH * 1.2
        self.AISLE_WIDTH = 12.0
        self.CIRCULATION_ZONE = 15.0
        self.NUM_COLUMNS = 5
        self.CARS_PER_COLUMN = 10
        
        # État
        self.state = np.zeros(4, dtype=np.float32)
        self.prev_dist_to_target = None
        self._prev_car_pos = None
        self.current_zone = None
        self.visited_zones = set()

        # Actions
        self.action_space = gym.spaces.Box(
            low=np.array([-self.MAX_ACCEL, -self.MAX_STEER_ANGLE], dtype=np.float32),
            high=np.array([self.MAX_ACCEL, self.MAX_STEER_ANGLE], dtype=np.float32),
            shape=(2,)
        )
        
        # Observations: 25 éléments
        # car(4) + target(2) + target_dir(2) + angle_error(1) + 
        # obstacles_pos(8) + obstacles_dist(4) + danger_scores(4)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )

        # Variables
        self.target_polygon = None
        self.target_pose = np.zeros(3)
        self.obstacles = []
        self.obstacle_poses = []
        self.empty_spot_index = None
        
        # Calcul dimensions pour centrer le parking
        parking_width = (self.NUM_COLUMNS * self.SPOT_DEPTH) + ((self.NUM_COLUMNS - 1) * self.AISLE_WIDTH)
        parking_height = (self.CARS_PER_COLUMN * self.SPOT_WIDTH)
        
        # Ajouter marge pour circulation et centrer
        self.world_width = parking_width + 2 * self.CIRCULATION_ZONE + self.AISLE_WIDTH
        self.world_height = parking_height + 2 * self.CIRCULATION_ZONE
        
        self.render_mode = render_mode
        self.pixels_per_meter = 8  # Augmenté pour meilleure visualisation
        self.screen_width = int(self.world_width * self.pixels_per_meter)
        self.screen_height = int(self.world_height * self.pixels_per_meter)
        self.screen = None
        self.clock = None
        self.font = None
        
        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Parking RL - Vue de dessus")
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        
        print(f"✅ Parking: {self.NUM_COLUMNS}cols × {self.CARS_PER_COLUMN}cars = 50 voitures")
        print(f"✅ Monde: {self.world_width:.1f}m × {self.world_height:.1f}m")
        print(f"✅ Observations: {self.observation_space.shape[0]} éléments")
        print(f"✅ Système de zones activé (4 zones)")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Créer parking centré
        self.obstacles = []
        self.obstacle_poses = []
        total_spots = self.NUM_COLUMNS * self.CARS_PER_COLUMN
        self.empty_spot_index = self.np_random.integers(0, total_spots)
        
        # Calculer position de départ du parking (centré)
        parking_width = (self.NUM_COLUMNS * self.SPOT_DEPTH) + ((self.NUM_COLUMNS - 1) * self.AISLE_WIDTH)
        start_x = -parking_width / 2
        
        spot_idx = 0
        for col in range(self.NUM_COLUMNS):
            col_x = start_x + col * (self.SPOT_DEPTH + self.AISLE_WIDTH) + self.SPOT_DEPTH / 2
            
            for row in range(self.CARS_PER_COLUMN):
                spot_y = -self.world_height/2 + self.CIRCULATION_ZONE + row * self.SPOT_WIDTH + self.SPOT_WIDTH/2
                
                if spot_idx == self.empty_spot_index:
                    # TARGET
                    self.target_pose = np.array([col_x, spot_y, 0.0])
                    half_d, half_w = self.SPOT_DEPTH/2, self.SPOT_WIDTH/2
                    coords = [
                        (col_x - half_d, spot_y - half_w), 
                        (col_x - half_d, spot_y + half_w),
                        (col_x + half_d, spot_y + half_w), 
                        (col_x + half_d, spot_y - half_w)
                    ]
                    self.target_polygon = Polygon(coords)
                else:
                    # VOITURE GARÉE
                    car_state = np.array([col_x, spot_y, 0.0, 0.0])
                    self.obstacles.append(Polygon(self._get_car_polygon(car_state)))
                    self.obstacle_poses.append([col_x, spot_y])
                
                spot_idx += 1
        
        # Spawn intelligent dans allées ou circulation (centré et parallèle)
        for _ in range(100):
            zone_type = self.np_random.integers(0, 3)
            
            if zone_type == 0:  # ALLÉE (entre les colonnes)
                aisle_idx = self.np_random.integers(0, self.NUM_COLUMNS + 1)
                
                if aisle_idx == 0:
                    # Première allée (à gauche)
                    aisle_center_x = start_x - self.AISLE_WIDTH / 2
                elif aisle_idx == self.NUM_COLUMNS:
                    # Dernière allée (à droite)
                    aisle_center_x = start_x + self.NUM_COLUMNS * self.SPOT_DEPTH + (self.NUM_COLUMNS - 1) * self.AISLE_WIDTH + self.AISLE_WIDTH / 2
                else:
                    # Allées centrales
                    aisle_center_x = start_x + aisle_idx * self.SPOT_DEPTH + (aisle_idx - 0.5) * self.AISLE_WIDTH
                
                x = aisle_center_x  # Toujours centré dans l'allée
                y = self.np_random.uniform(
                    -self.world_height/2 + self.CIRCULATION_ZONE + self.CAR_LENGTH,
                    self.world_height/2 - self.CIRCULATION_ZONE - self.CAR_LENGTH
                )
                theta = self.np_random.choice([np.pi/2, -np.pi/2])  # Toujours parallèle
                
            elif zone_type == 1:  # CIRCULATION HAUT
                x = self.np_random.uniform(
                    -self.world_width/2 + self.CAR_LENGTH,
                    self.world_width/2 - self.CAR_LENGTH
                )
                y = self.world_height/2 - self.CIRCULATION_ZONE / 2
                theta = self.np_random.choice([0.0, np.pi])  # Horizontal
                
            else:  # CIRCULATION BAS
                x = self.np_random.uniform(
                    -self.world_width/2 + self.CAR_LENGTH,
                    self.world_width/2 - self.CAR_LENGTH
                )
                y = -self.world_height/2 + self.CIRCULATION_ZONE / 2
                theta = self.np_random.choice([0.0, np.pi])  # Horizontal
            
            self.state = np.array([x, y, theta, 0.0], dtype=np.float32)
            
            if not self._check_collision() and not self._check_out_of_bounds():
                break
        
        start_state_for_info = self.state.copy()
        
        # Init reward vars
        target_center = self.target_pose[:2]
        car_center = self.state[:2]
        self.prev_dist_to_target = np.linalg.norm(target_center - car_center)
        self._prev_car_pos = car_center.copy()
        
        # Init zone system
        self.current_zone = self._get_zone(self.prev_dist_to_target)
        self.visited_zones = {self.current_zone}

        obs_dict = self._get_obs_dict()
        observation = self._obs_dict_to_array(obs_dict)
        info = {"state": self.state, "start_state": start_state_for_info}
        return observation, info
    
    def step(self, action):
        self._dynamic_step(action)
        collision = self._check_collision()
        out_of_bounds = self._check_out_of_bounds()
        success = self._check_success()

        obs_dict = self._get_obs_dict()
        observation = self._obs_dict_to_array(obs_dict)

        total_reward, reward_info = self._calculate_reward(action, collision, out_of_bounds, success, obs_dict)

        terminated = collision or success or out_of_bounds
        truncated = False
        info = {"state": self.state, "reward_total": total_reward, **reward_info}

        return observation, total_reward, terminated, truncated, info
    
    def render(self):
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill((240, 240, 240))  # Gris clair pour le fond

        def world_to_pixels(coords):
            px = (coords[0] + self.world_width / 2) * self.pixels_per_meter
            py = (-coords[1] + self.world_height / 2) * self.pixels_per_meter
            return (px, py)

        # Dessiner les lignes de parking
        parking_width = (self.NUM_COLUMNS * self.SPOT_DEPTH) + ((self.NUM_COLUMNS - 1) * self.AISLE_WIDTH)
        start_x = -parking_width / 2
        
        for col in range(self.NUM_COLUMNS + 1):
            if col < self.NUM_COLUMNS:
                line_x = start_x + col * (self.SPOT_DEPTH + self.AISLE_WIDTH)
            else:
                line_x = start_x + self.NUM_COLUMNS * self.SPOT_DEPTH + (self.NUM_COLUMNS - 1) * self.AISLE_WIDTH
            
            start_point = world_to_pixels((line_x, -self.world_height/2 + self.CIRCULATION_ZONE))
            end_point = world_to_pixels((line_x, self.world_height/2 - self.CIRCULATION_ZONE))
            pygame.draw.line(canvas, (200, 200, 200), start_point, end_point, 2)

        # Target (vert clair avec contour)
        target_poly_pixels = [world_to_pixels(p) for p in self.target_polygon.exterior.coords]
        pygame.draw.polygon(canvas, (144, 238, 144), target_poly_pixels)
        pygame.draw.polygon(canvas, (0, 128, 0), target_poly_pixels, 3)

        # Obstacles (voitures garées - vue de dessus)
        for obs_poly in self.obstacles:
            self._draw_car_top_view(canvas, obs_poly, (100, 100, 100), world_to_pixels)

        # Agent (voiture bleue)
        car_poly = Polygon(self._get_car_polygon(self.state))
        self._draw_car_top_view(canvas, car_poly, (70, 130, 180), world_to_pixels, is_agent=True)

        # Afficher infos
        if self.font:
            target_dist = np.linalg.norm(self.state[:2] - self.target_pose[:2])
            zone = self._get_zone(target_dist)
            info_text = [
                f"Distance: {target_dist:.1f}m",
                f"Zone: {zone}/4",
                f"Vitesse: {abs(self.state[3]):.1f}m/s"
            ]
            for i, text in enumerate(info_text):
                text_surface = self.font.render(text, True, (0, 0, 0))
                canvas.blit(text_surface, (10, 10 + i * 25))

        if self.render_mode == "human":
            self.screen.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:
            return np.transpose(pygame.surfarray.pixels3d(canvas), axes=(1, 0, 2))

    def _draw_car_top_view(self, canvas, car_poly, color, world_to_pixels, is_agent=False):
        """Dessine une voiture vue de dessus avec détails réalistes"""
        # Corps principal
        poly_pixels = [world_to_pixels(p) for p in car_poly.exterior.coords]
        pygame.draw.polygon(canvas, color, poly_pixels)
        pygame.draw.polygon(canvas, (50, 50, 50), poly_pixels, 2)  # Contour
        
        # Calculer l'orientation de la voiture
        coords = list(car_poly.exterior.coords)
        front_center = ((coords[2][0] + coords[3][0])/2, (coords[2][1] + coords[3][1])/2)
        rear_center = ((coords[0][0] + coords[1][0])/2, (coords[0][1] + coords[1][1])/2)
        
        # Pare-brise avant (plus clair)
        windshield_color = tuple(min(255, c + 40) for c in color)
        front_pixels = world_to_pixels(front_center)
        pygame.draw.circle(canvas, windshield_color, front_pixels, 8)
        
        # Feux arrière (rouge)
        if is_agent:
            rear_pixels = world_to_pixels(rear_center)
            pygame.draw.circle(canvas, (200, 0, 0), rear_pixels, 5)
        
        # Rétroviseurs (petits rectangles sur les côtés)
        if is_agent:
            left_mirror = ((coords[1][0] + coords[2][0])/2, (coords[1][1] + coords[2][1])/2)
            right_mirror = ((coords[0][0] + coords[3][0])/2, (coords[0][1] + coords[3][1])/2)
            pygame.draw.circle(canvas, (50, 50, 50), world_to_pixels(left_mirror), 3)
            pygame.draw.circle(canvas, (50, 50, 50), world_to_pixels(right_mirror), 3)

    def _dynamic_step(self, action):
        x, y, theta, v = self.state
        accel = np.clip(action[0], -self.MAX_ACCEL, self.MAX_ACCEL)
        steer_angle = np.clip(action[1], -self.MAX_STEER_ANGLE, self.MAX_STEER_ANGLE)
        v_next = v * self.VELOCITY_DECAY + accel * self.dt
        v_next = np.clip(v_next, -self.MAX_VELOCITY / 2, self.MAX_VELOCITY)
        theta_next = theta + (v_next / self.CAR_LENGTH) * np.tan(steer_angle) * self.dt
        x_next = x + v_next * np.cos(theta_next) * self.dt
        y_next = y + v_next * np.sin(theta_next) * self.dt
        self.state = np.array([x_next, y_next, theta_next, v_next], dtype=np.float32)
    
    def _get_obs_dict(self):
        target_center_x, target_center_y, target_angle = self.target_pose
        car_x, car_y, car_theta, car_v = self.state
        car_theta = (car_theta + np.pi) % (2 * np.pi) - np.pi
        
        # Direction vers target
        dx, dy = target_center_x - car_x, target_center_y - car_y
        dist_to_target = np.sqrt(dx**2 + dy**2)
        
        if dist_to_target > 0.01:
            target_dir_x = dx / dist_to_target
            target_dir_y = dy / dist_to_target
        else:
            target_dir_x = target_dir_y = 0.0
        
        # Angle error
        angle_to_target = np.arctan2(dy, dx)
        angle_error = angle_to_target - car_theta
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        
        # 4 obstacles les plus proches avec distances et danger scores
        car_pos = np.array([car_x, car_y])
        dists = [np.linalg.norm(car_pos - np.array(obs_pos)) for obs_pos in self.obstacle_poses]
        closest_indices = np.argsort(dists)[:4]
        
        closest_obs = []
        closest_dists = []
        danger_scores = []
        
        for idx in closest_indices:
            obs_pos = self.obstacle_poses[idx]
            closest_obs.extend(obs_pos)
            
            # Distance
            dist = dists[idx]
            closest_dists.append(dist)
            
            # Danger score exponentiel
            danger = np.exp(-dist / 2.0)  # Plus la distance est petite, plus le danger est grand
            danger_scores.append(danger)

        return {
            "car_x": car_x, "car_y": car_y, "car_theta": car_theta, "car_v": car_v,
            "target_x": target_center_x, "target_y": target_center_y,
            "target_dir_x": target_dir_x, "target_dir_y": target_dir_y,
            "angle_error": angle_error,
            "closest_obs": closest_obs,
            "closest_dists": closest_dists,
            "danger_scores": danger_scores
        }

    def _obs_dict_to_array(self, obs_dict):
        return np.array([
            # Voiture (4)
            obs_dict["car_x"], obs_dict["car_y"], obs_dict["car_theta"], obs_dict["car_v"],
            # Target (2)
            obs_dict["target_x"], obs_dict["target_y"],
            # Direction vers target (2)
            obs_dict["target_dir_x"], obs_dict["target_dir_y"],
            # Erreur d'angle (1)
            obs_dict["angle_error"],
            # Positions des 4 obstacles (8)
            *obs_dict["closest_obs"],
            # Distances aux 4 obstacles (4)
            *obs_dict["closest_dists"],
            # Danger scores (4)
            *obs_dict["danger_scores"]
        ], dtype=np.float32)
    
    def _get_car_polygon(self, state):
        x, y, theta, v = state
        half_len, half_wid = self.CAR_LENGTH / 2, self.CAR_WIDTH / 2
        corners = [(-half_len, -half_wid), (-half_len, +half_wid),
                   (+half_len, +half_wid), (+half_len, -half_wid)]
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        transformed = []
        for (cx, cy) in corners:
            tx = x + (cx * cos_theta - cy * sin_theta)
            ty = y + (cx * sin_theta + cy * cos_theta)
            transformed.append((tx, ty))
        return transformed
    
    def _check_collision(self):
        car_poly = Polygon(self._get_car_polygon(self.state))
        for obs_poly in self.obstacles:
            if car_poly.intersects(obs_poly):
                return True
        return False
    
    def _check_out_of_bounds(self):
        x, y = self.state[0], self.state[1]
        return (x < -self.world_width/2 or x > self.world_width/2 or
                y < -self.world_height/2 or y > self.world_height/2)

    def _check_success(self):
        car_poly = Polygon(self._get_car_polygon(self.state))
        intersection = car_poly.intersection(self.target_polygon).area
        car_area = car_poly.area
        
        is_parked = (intersection / car_area) > 0.95
        is_stopped = np.abs(self.state[3]) < 0.1
        
        target_angle = self.target_pose[2]
        car_angle = self.state[2]
        angle_diff = np.abs(target_angle - car_angle)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)
        is_aligned = angle_diff < 0.2
        
        return is_parked and is_stopped and is_aligned
    
    def _get_zone(self, distance):
        """Retourne la zone actuelle basée sur la distance au target"""
        if distance > 20:
            return 1
        elif distance > 10:
            return 2
        elif distance > 5:
            return 3
        else:
            return 4
    
    def _calculate_reward(self, action, collision, out_of_bounds, success, obs_dict):
        reward_info = {}

        # Pénalités terminales (réduites)
        if collision:
            reward_info["reward_collision"] = -150.0
            return sum(reward_info.values()), reward_info
        
        if out_of_bounds:
            reward_info["reward_out_of_bounds"] = -150.0
            return sum(reward_info.values()), reward_info
            
        if success:
            reward_info["reward_success"] = 100.0
            return sum(reward_info.values()), reward_info

        # === RÉCOMPENSES CONTINUES ===
        
        # 1. Progress reward (système existant amélioré)
        target_center = self.target_pose[:2]
        car_center = self.state[:2]
        current_dist = np.linalg.norm(target_center - car_center)
        
        reward_progress = 0.0
        if self.prev_dist_to_target is not None:
            delta_dist = self.prev_dist_to_target - current_dist
            # Échelle adaptative
            if current_dist < 5.0:
                weight = 8.0
            elif current_dist < 10.0:
                weight = 4.0
            elif current_dist < 20.0:
                weight = 2.0
            else:
                weight = 1.0
            reward_progress = delta_dist * weight
        
        self.prev_dist_to_target = current_dist
        reward_info["reward_progress"] = reward_progress

        # 2. Zone progression bonus
        new_zone = self._get_zone(current_dist)
        reward_zone = 0.0
        if new_zone > self.current_zone and new_zone not in self.visited_zones:
            # Progression vers une meilleure zone
            reward_zone = 10.0
            self.visited_zones.add(new_zone)
        self.current_zone = new_zone
        reward_info["reward_zone_progression"] = reward_zone

        # 3. Danger-aware reward (pénalité si trop proche des obstacles)
        danger_scores = obs_dict["danger_scores"]
        max_danger = max(danger_scores) if danger_scores else 0
        reward_danger = 0.0
        if max_danger > 0.3: # Seuil un peu plus large
            reward_danger = -(max_danger**2) * 15.0
        reward_info["reward_danger"] = reward_danger

        # 4. Angle alignment (système existant)
        angle_error = obs_dict["angle_error"]
        if current_dist < 10.0:
            angle_weight = 2.0
        elif current_dist < 20.0:
            angle_weight = 1.0
        else:
            angle_weight = 0.5
        
        reward_angle_alignment = -abs(angle_error) * angle_weight * 0.5  # Réduit un peu
        reward_info["reward_angle_alignment"] = reward_angle_alignment

        # 5. Direction bonus (système existant mais ajusté)
        if hasattr(self, '_prev_car_pos') and self._prev_car_pos is not None:
            movement = car_center - self._prev_car_pos
            movement_norm = np.linalg.norm(movement)
            
            if movement_norm > 0.01:
                direction_to_target = target_center - self._prev_car_pos
                direction_norm = np.linalg.norm(direction_to_target)
                
                if direction_norm > 0.01:
                    dot_product = np.dot(movement, direction_to_target) / (movement_norm * direction_norm)
                    reward_info["reward_direction"] = dot_product * 0.3
                else:
                    reward_info["reward_direction"] = 0.0
            else:
                reward_info["reward_direction"] = 0.0
        else:
            reward_info["reward_direction"] = 0.0
        
        self._prev_car_pos = car_center.copy()

        # 6. Time penalty (constant)
        reward_info["reward_time_penalty"] = -0.1

        # 7. Immobility penalty
        reward_immobility = 0.0
        if abs(self.state[3]) < 0.1 and current_dist > 5.0:
            reward_immobility = -0.5
        reward_info["reward_immobility"] = reward_immobility

        # 8. Bonus final (proximité + alignement)
        reward_final_alignment = 0.0
        if current_dist < 3.0 and abs(angle_error) < 0.3:
            reward_final_alignment = 5.0
        elif current_dist < 2.0 and abs(angle_error) < 0.2:
            reward_final_alignment = 10.0
        elif current_dist < 1.0 and abs(angle_error) < 0.1:
            reward_final_alignment = 20.0
        reward_info["reward_final_alignment"] = reward_final_alignment
        
        total_reward = sum(reward_info.values())
        return total_reward, reward_info
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()