import gymnasium as gym
import numpy as np
from shapely.geometry import Polygon
import pygame

# On utilisera pygame pour le rendu plus tard
# import pygame 

class ParkingEnv(gym.Env):
    """
    Un environnement Gymnasium pour un problème de stationnement autonome.
    
    État interne (non-observable directement par l'agent):
    - x: position en m
    - y: position en m
    - theta: orientation en rad
    - v: vitesse en m/s
    """
    
    # Métadonnées pour le rendu (bonnes pratiques)
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # --- Constantes de l'environnement ---
        self.dt = 1 / self.metadata["render_fps"]
        
        # Dimensions du véhicule
        self.CAR_LENGTH = 4.5  # m
        self.CAR_WIDTH = 2.0   # m

        # Limites de la dynamique (basé sur devoir5.pdf)
        self.MAX_STEER_ANGLE = np.pi / 6  # 30 degrés
        self.MAX_ACCEL = 2.0              # m/s^2 (valeur arbitraire, à ajuster)
        self.MAX_VELOCITY = 5.0           # m/s (valeur arbitraire, à ajuster)
        self.VELOCITY_DECAY = 0.99 # Multiplicateur par step (simule la friction/traînée)
        
        # Limites des capteurs (basé sur devoir5.pdf)
        self.LIDAR_RANGE = 10.0 # m
        self.LIDAR_RAYS = 16
        self.PROX_RANGE = 5.0   # m
        self.PROX_SENSORS = 8
        
        # --- État interne ---
        # [x, y, theta, v]
        # On l'initialise à zéro, mais reset() le placera correctement.
        self.state = np.zeros(4, dtype=np.float32)

        
        # === 1. DÉFINITION DE L'ESPACE D'ACTIONS ===
        # [Accélération, Angle de direction]
        # Basé sur ton fichier devoir5.pdf
        self.action_space = gym.spaces.Box(
            low=np.array([-self.MAX_ACCEL, -self.MAX_STEER_ANGLE], dtype=np.float32),
            high=np.array([self.MAX_ACCEL, self.MAX_STEER_ANGLE], dtype=np.float32),
            shape=(2,)
        )
        
        # === 2. DÉFINITION DE L'ESPACE D'OBSERVATIONS ===
        # On va simplifier l'observation (MDP au lieu de POMDP)
        # L'agent connaîtra sa position relative par rapport à la cible.
        # [dist_x, dist_y, dist_theta, v_voiture]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # On va définir la cible et les obstacles dans reset()
        self.target_polygon = None
        self.obstacles = [] # Liste de polygones
                
        self.render_mode = render_mode
        print(f"Espace d'action initialisé : {self.action_space}")
        print(f"Espace d'observation initialisé (shape) : {self.observation_space.shape}")

        # (Dans __init__, après la définition des espaces)
        
        self.world_width = 100 # m (taille du parking)
        self.world_height = 100 # m
        
        # Facteur d'échelle pour l'affichage
        self.pixels_per_meter = 5 
        
        self.screen_width = int(self.world_width * self.pixels_per_meter)
        self.screen_height = int(self.world_height * self.pixels_per_meter)
        
        self.screen = None
        self.clock = None
        
        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Parking Environment")
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()


    def reset(self, seed=None, options=None):
        from shapely.geometry import Polygon
        super().reset(seed=seed)

        start_x = self.np_random.uniform(-5.0, 5.0)
        start_y = self.np_random.uniform(-5.0, 5.0)
        start_theta = self.np_random.uniform(-np.pi/18, np.pi/18) # +/- 10 degrees
        start_v = 0.0
        
        self.state = np.array([start_x, start_y, start_theta, start_v], dtype=np.float32)

        # --- PARKING LOT SCENARIO (Perpendicular Example) ---
        spot_width = self.CAR_WIDTH * 1.5 # Standard spot width
        spot_depth = self.CAR_LENGTH * 1.2 # Generous depth

        # 1. Define the target parking spot (Empty Spot)
        #    Let's place it at x=20, centered at y=0
        target_center_x = 30.0
        target_center_y = 0.0
        target_angle = 0.0 # Perpendicular
        
        # Define target polygon using its corners relative to its center
        half_depth = spot_depth / 2
        half_width = spot_width / 2
        target_coords_local = [
            (-half_depth, -half_width), (-half_depth, +half_width),
            (+half_depth, +half_width), (+half_depth, -half_width)
        ]
        # Rotate and translate (though rotation is 0 here)
        target_coords_world = []
        cos_t, sin_t = np.cos(target_angle), np.sin(target_angle)
        for x, y in target_coords_local:
             wx = target_center_x + (x * cos_t - y * sin_t)
             wy = target_center_y + (y * cos_t + x * sin_t)
             target_coords_world.append((wx, wy))
        self.target_polygon = Polygon(target_coords_world)
        
        # Keep track of the target pose for observation/reward
        self.target_pose = np.array([target_center_x, target_center_y, target_angle])

        # 2. Define the obstacles (Parked Cars) - MODIFIED
        self.obstacles = []
        # Car to the left of the spot (KEEP THIS)
        left_car_center_x = target_center_x
        left_car_center_y = target_center_y + spot_width
        left_car_angle = target_angle
        left_car_state = np.array([left_car_center_x, left_car_center_y, left_car_angle, 0.0])
        self.obstacles.append(Polygon(self._get_car_polygon(left_car_state)))

        # Car to the right of the spot
        right_car_center_x = target_center_x
        right_car_center_y = target_center_y - spot_width
        right_car_angle = target_angle
        right_car_state = np.array([right_car_center_x, right_car_center_y, right_car_angle, 0.0])
        self.obstacles.append(Polygon(self._get_car_polygon(right_car_state)))
        
        # Optional: Add cars across the aisle
        # car_across_center_x = target_center_x - spot_depth * 1.5
        # car_across_center_y = target_center_y
        # car_across_angle = target_angle + np.pi # Facing the other way
        # car_across_state = np.array([car_across_center_x, car_across_center_y, car_across_angle, 0.0])
        # self.obstacles.append(Polygon(self._get_car_polygon(car_across_state)))
        
        # ---

        # 3. Initial Observation and Info
        # Make sure the start state isn't colliding
        while self._check_collision():
            print("WARN: Collision au démarrage, régénération...")
            start_x = self.np_random.uniform(-5.0, 5.0)
            start_y = self.np_random.uniform(-5.0, 5.0)
            start_theta = self.np_random.uniform(-np.pi/18, np.pi/18)
            self.state = np.array([start_x, start_y, start_theta, start_v], dtype=np.float32)

        obs_dict = self._get_obs_dict()
        observation = self._obs_dict_to_array(obs_dict)
        info = {"state": self.state}

        print("Environnement réinitialisé avec scénario parking.")
        return observation, info
    
    def step(self, action):

        # 1. Mettre à jour l'état du véhicule
        self._dynamic_step(action)

        # 2. Vérifier les états terminaux
        collision = self._check_collision()
        success = self._check_success()

        # 3. Obtenir les nouvelles observations
        obs_dict = self._get_obs_dict()
        observation = self._obs_dict_to_array(obs_dict)

        # 4. Calculer la récompense
        reward = self._calculate_reward(action, collision, success, obs_dict)

        # 6. Définir les flags de terminaison
        terminated = collision or success
        truncated = False # (On ajoutera une limite de temps plus tard)

        # 7. Infos (utile pour le débogage)
        info = {"state": self.state, "reward": reward} 

        return observation, reward, terminated, truncated, info
        
    def render(self):
        """
        Dessine l'état actuel de l'environnement.
        """
        
        # Créer la surface de dessin (Canevas)
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill((255, 255, 255)) # Fond blanc

        # --- Coordonnées du monde (m) vers coordonnées de l'écran (pixels) ---
        # Pygame a (0,0) en HAUT à gauche, notre simulation a (0,0) au MILIEU
        def world_to_pixels(coords):
            px = (coords[0] + self.world_width / 2) * self.pixels_per_meter
            py = (-coords[1] + self.world_height / 2) * self.pixels_per_meter # Inverser Y
            return (px, py)

        # 1. Dessiner la place de parking cible
        #    Pour l'instant, une place fixe à (20, 0)
        # 1. Dessiner la place de parking cible
        target_poly_pixels = [world_to_pixels(p) for p in self.target_polygon.exterior.coords]
        pygame.draw.polygon(canvas, (200, 200, 200), target_poly_pixels) # Gris clair

        # 1.5. Dessiner les obstacles
        for obs_poly in self.obstacles:
            obs_coords_pixels = [world_to_pixels(p) for p in obs_poly.exterior.coords]
            pygame.draw.polygon(canvas, (100, 100, 100), obs_coords_pixels) # Gris foncé

        # 2. Dessiner la voiture
        car_poly_world = self._get_car_polygon(self.state)
        car_poly_pixels = [world_to_pixels(p) for p in car_poly_world]
        pygame.draw.polygon(canvas, (0, 0, 255), car_poly_pixels) # Bleu
        
        # Dessiner une ligne pour l'avant de la voiture
        front_mid_point = ((car_poly_world[1][0] + car_poly_world[2][0]) / 2, 
                           (car_poly_world[1][1] + car_poly_world[2][1]) / 2)
        center_point = (self.state[0], self.state[1])
        pygame.draw.line(canvas, (255, 0, 0), world_to_pixels(center_point), 
                         world_to_pixels(front_mid_point), 3) # Rouge

        
        if self.render_mode == "human":
            # Copier le canevas sur l'écran principal
            self.screen.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:  # rgb_array
            # Transposer pour correspondre au format de Gymnasium (H, W, C)
            return np.transpose(pygame.surfarray.pixels3d(canvas), axes=(1, 0, 2))

    def _dynamic_step(self, action):
        """
        Met à jour le self.state du véhicule en utilisant un modèle cinématique de bicyclette.
        """
        
        # Récupérer l'état actuel [x, y, theta, v]
        x, y, theta, v = self.state
        
        # 1. Extraire et contraindre (clip) les actions
        # action[0] = accélération, action[1] = angle de direction
        accel = np.clip(action[0], -self.MAX_ACCEL, self.MAX_ACCEL)
        steer_angle = np.clip(action[1], -self.MAX_STEER_ANGLE, self.MAX_STEER_ANGLE)
        
        # 2. Mettre à jour la vitesse
        #    (Intégration d'Euler simple)
        v_next = v * self.VELOCITY_DECAY + accel * self.dt
        #v_next = v + accel * self.dt
        v_next = np.clip(v_next, -self.MAX_VELOCITY / 2, self.MAX_VELOCITY) # Limiter la vitesse (ex: marche arrière plus lente)
        
        # 3. Mettre à jour l'orientation (theta)
        #    Cette équation vient du modèle bicyclette: tan(steer) * v / L
        #    On utilise v_next pour une intégration plus stable (Euler semi-implicite)
        theta_next = theta + (v_next / self.CAR_LENGTH) * np.tan(steer_angle) * self.dt
        
        # 4. Mettre à jour la position (x, y)
        x_next = x + v_next * np.cos(theta_next) * self.dt
        y_next = y + v_next * np.sin(theta_next) * self.dt
        
        # 5. Sauvegarder le nouvel état
        self.state = np.array([x_next, y_next, theta_next, v_next], dtype=np.float32)
    
    def _get_obs_dict(self):
        # Center and angle of the target spot
        target_center_x, target_center_y, target_angle = self.target_pose

        # State of the car
        car_x, car_y, car_theta, car_v = self.state

        # Normalize car_angle to [-pi, pi]
        car_theta = (car_theta + np.pi) % (2 * np.pi) - np.pi

        return {
            "car_x": car_x,
            "car_y": car_y,
            "car_theta": car_theta,
            "car_v": car_v,
            "target_x": target_center_x,
            "target_y": target_center_y
        }

    def _obs_dict_to_array(self, obs_dict):
        return np.array([
            obs_dict["car_x"],
            obs_dict["car_y"],
            obs_dict["car_theta"],
            obs_dict["car_v"],
            obs_dict["target_x"],
            obs_dict["target_y"]
        ], dtype=np.float32)

    def _get_observation(self):
        """
        Calcule et retourne le vecteur d'observation final.
        """
        obs_dict = self._get_obs_dict()
        return self._obs_dict_to_array(obs_dict)
    
    def _get_car_polygon(self, state):
        """
        Calcule les 4 coins d'un rectangle centré sur (x, y) et tourné de theta.
        """
        x, y, theta, v = state
        
        # Définir les coins par rapport au centre (0,0) avant rotation
        half_len = self.CAR_LENGTH / 2
        half_wid = self.CAR_WIDTH / 2
        
        corners = [
            (-half_len, -half_wid), # Arrière gauche
            (-half_len, +half_wid), # Avant gauche
            (+half_len, +half_wid), # Avant droit
            (+half_len, -half_wid)  # Arrière droit
        ]
        
        # Créer la matrice de rotation
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Tourner et translater les coins
        transformed_corners = []
        for (cx, cy) in corners:
            tx = x + (cx * cos_theta - cy * sin_theta)
            ty = y + (cy * cos_theta + cx * sin_theta)
            transformed_corners.append((tx, ty))
            
        return transformed_corners
    
    def _check_collision(self):
        """
        Vérifie si le polygone de la voiture touche un obstacle.
        """
        from shapely.geometry import Polygon
        car_coords = self._get_car_polygon(self.state)
        car_poly = Polygon(car_coords)
        
        for obs_poly in self.obstacles:
            if car_poly.intersects(obs_poly):
                return True # Il y a collision
        return False

    def _check_success(self):
        """
        Vérifie si la voiture est "suffisamment" dans la cible.
        Condition : 90% de la voiture doit être dans la zone cible.
        """
        from shapely.geometry import Polygon
        car_coords = self._get_car_polygon(self.state)
        car_poly = Polygon(car_coords)
        
        # Calcule l'aire de l'intersection
        intersection_area = car_poly.intersection(self.target_polygon).area
        car_area = car_poly.area
        
        # Succès si 90% de la voiture est dans la cible ET la vitesse est quasi-nulle
        is_parked = (intersection_area / car_area) > 0.9
        is_stopped = np.abs(self.state[3]) < 0.1 # Vitesse inférieure à 0.1 m/s
        
        return is_parked and is_stopped
    
    def _calculate_reward(self, action, collision, success, obs_dict):
        """
        Calcule la récompense shapée.
        VERSION 12: Simple, dense penalty + terminal rewards.
        """

        # 1. Terminal Rewards/Penalties
        if collision:
            return -100.0 # Large penalty for crashing
        if success:
            return 100.0  # Large reward for success

        # 2. Calculate Final Error (how far from success?)
        # We need car's distance and angle to the target
        target_center = self.target_pose[:2]
        car_center = self.state[:2]
        dist_error = np.linalg.norm(target_center - car_center)

        target_angle = self.target_pose[2]
        # car_angle is now in obs_dict
        car_angle = obs_dict["car_theta"] # Already normalized in _get_obs_dict
        angle_error = abs(target_angle - car_angle)

        # 3. Dense Penalty
        # This is the agent's main "score" at every step.
        # It's a combination of distance error, angle error, and speed.
        # The agent's goal is to minimize this penalty (get it to 0).

        # Penalize distance from target
        reward = -1.0 * (dist_error / 10.0) # Scaled by ~10m

        # Penalize angle error
        reward -= 0.5 * angle_error # Scaled

        # Penalize speed (encourages stopping)
        reward -= 0.1 * abs(self.state[3])

        # Penalize actions (encourages efficiency/comfort)
        reward -= 0.01 * (action[0]**2)
        reward -= 0.01 * (action[1]**2)

        return reward
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()