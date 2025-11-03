import gymnasium as gym
import numpy as np
from shapely.geometry import Polygon, LineString, Point
import pygame

class ParkingEnv(gym.Env):    
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # ----- simulation parameters -----
        self.dt = 1 / self.metadata["render_fps"]
        self.CAR_LENGTH = 4.5
        self.CAR_WIDTH = 2.0
        self.MAX_STEER_ANGLE = np.pi / 6
        self.MAX_ACCEL = 2.0
        self.MAX_VELOCITY = 5.0
        self.VELOCITY_DECAY = 0.99
        
        # parking layout
        self.SPOT_WIDTH = self.CAR_WIDTH * 1.5
        self.SPOT_DEPTH = self.CAR_LENGTH * 1.2
        self.AISLE_WIDTH = 12.0
        self.CIRCULATION_ZONE = 15.0
        self.NUM_COLUMNS = 1  # Changed to 1 column
        self.CARS_PER_COLUMN = 10
        
        # lidar
        self.NUM_LIDAR_RAYS = 8
        self.LIDAR_MAX_RANGE = 20.0 # meters
        # Angles des rayons (0=avant, pi/2=gauche, pi=arrière, 3pi/2=droite)
        self.LIDAR_ANGLES = np.linspace(0, 2 * np.pi, self.NUM_LIDAR_RAYS, endpoint=False)
        
        # ----- agent state -----
        self.state = np.zeros(4, dtype=np.float32)  # x, y, theta, v
        self.prev_dist_to_target = None
        self._prev_car_pos = None

        # ----- actions -----
        self.action_space = gym.spaces.Box(
            low=np.array([-self.MAX_ACCEL, -self.MAX_STEER_ANGLE], dtype=np.float32),
            high=np.array([self.MAX_ACCEL, self.MAX_STEER_ANGLE], dtype=np.float32),
            shape=(2,)
        )
        
        # Compute parking/world size
        parking_width = (self.NUM_COLUMNS * self.SPOT_DEPTH) + ((self.NUM_COLUMNS - 1) * self.AISLE_WIDTH)
        parking_height = (self.CARS_PER_COLUMN * self.SPOT_WIDTH)
        self.world_width = parking_width + 2 * self.CIRCULATION_ZONE + self.AISLE_WIDTH
        self.world_height = parking_height + 2 * self.CIRCULATION_ZONE

        # world boundaries (for lidar intersections)
        w, h = self.world_width / 2, self.world_height / 2
        self.world_boundaries = [
            LineString([(-w, -h), (w, -h)]), # bottom
            LineString([(w, -h), (w, h)]),   # right
            LineString([(w, h), (-w, h)]),   # top
            LineString([(-w, h), (-w, -h)])  # left
        ]

        # ----- observation space -----
        # We will expose relative target coordinates (dx, dy) instead of global target_x/target_y
        # obs: car_x, car_y, car_theta, car_v, dx, dy, target_dir_x, target_dir_y, angle_error, lidar(8)
        low = np.array(
            [-np.inf, -np.inf, -np.pi, -self.MAX_VELOCITY] +  # car_state
            [-np.inf] * 2 +  # dx, dy
            [-1.0] * 2 +     # target_dir
            [-np.pi] +       # angle_error
            [0.0] * self.NUM_LIDAR_RAYS,
            dtype=np.float32
        )
        high = np.array(
            [np.inf, np.inf, np.pi, self.MAX_VELOCITY] +  # car_state
            [np.inf] * 2 +  # dx, dy
            [1.0] * 2 +     # target_dir
            [np.pi] +       # angle_error
            [self.LIDAR_MAX_RANGE] * self.NUM_LIDAR_RAYS,
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(4 + 2 + 2 + 1 + self.NUM_LIDAR_RAYS,), dtype=np.float32)

        # ----- variables -----
        self.target_polygon = None
        self.target_pose = np.zeros(3)
        self.obstacles = []
        self.obstacle_poses = []
        self.empty_spot_index = None
        
        # rendering
        self.render_mode = render_mode
        self.pixels_per_meter = 8
        self.screen_width = int(self.world_width * self.pixels_per_meter)
        self.screen_height = int(self.world_height * self.pixels_per_meter)
        self.screen = None
        self.clock = None
        self.font = None
        
        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Parking")
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

        # ----- reward and behavior tuning params (exposed for easy tuning) -----
        self.progress_scale = 10.0           # Was 3.0 → stronger approach motivation
        self.angle_activation_dist = 2.0      # Was 5.0 → only align when very close
        self.alignment_weight = -0.2          # Was -1.0 → 80% reduction, less harsh
        self.safety_threshold = 2.0           # Was 3.0 → allow closer approach to obstacles
        self.safety_scale = 0.5               # Was 2.0 → 75% reduction, less harsh
        self.safety_cap = -5.0                # Was -20.0 → less harsh maximum penalty
        self.step_penalty = -0.02
        self.velocity_near_goal_penalty = -0.5
        self.goal_slow_radius = 2.0
        self.collision_penalty = -50.0
        self.out_of_bounds_penalty = -30.0
        self.success_reward = 500.0           # Was 200.0 → make success VERY attractive
        self.per_step_clip = 10.0             # Applies only to per-step rewards (NOT terminals)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.obstacles = []
        self.obstacle_poses = []
        total_spots = self.NUM_COLUMNS * self.CARS_PER_COLUMN
        self.empty_spot_index = 7  # 8th spot from top (0-indexed)
        
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
                    # PARKED CAR
                    car_state = np.array([col_x, spot_y, 0.0, 0.0])
                    self.obstacles.append(Polygon(self._get_car_polygon(car_state)))
                    self.obstacle_poses.append([col_x, spot_y])
                
                spot_idx += 1
        
        # Spawn perpendicular to parked cars with vertical position variation
        for _ in range(100):
            # Spawn in the aisle, perpendicular to the single column
            # The column is centered at x = 0
            # Agent spawns at aisle distance from the column
            
            # Vary vertical position along the parking area
            y = self.np_random.uniform(
                -self.world_height/2 + self.CIRCULATION_ZONE + self.CAR_LENGTH,
                self.world_height/2 - self.CIRCULATION_ZONE - self.CAR_LENGTH
            )
            
            # Spawn in aisle (to the right of the column)
            # Use the same aisle distance as before
            x = self.AISLE_WIDTH / 2
            
            # Perpendicular to parked cars (±π/2) with small angle variation
            base_angle = self.np_random.choice([np.pi/2, -np.pi/2])
            angle_variation = self.np_random.uniform(-0.15, 0.15)  # ±8.6 degrees
            theta = base_angle + angle_variation
            
            self.state = np.array([x, y, theta, 0.0], dtype=np.float32)
            
            if not self._check_collision() and not self._check_out_of_bounds():
                break
        
        start_state_for_info = self.state.copy()
        
        target_center = self.target_pose[:2]
        car_center = self.state[:2]
        self.prev_dist_to_target = np.linalg.norm(target_center - car_center)
        self._prev_car_pos = car_center.copy()
        self.initial_distance = np.linalg.norm(target_center - car_center)
        self.immob_time = 0.0

        obs_dict = self._get_obs_dict()
        observation = self._obs_dict_to_array(obs_dict)
        info = {
            "state": self.state,
            "start_state": start_state_for_info,
            "target_x": self.target_pose[0],
            "target_y": self.target_pose[1]
        }
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
        info = {
            "state": self.state,
            "reward_total": total_reward,
            "target_x": self.target_pose[0],
            "target_y": self.target_pose[1],
            "angle_error": obs_dict["angle_error"],
            "lidar_distances": obs_dict["lidar_distances"],
            **reward_info
        }

        return observation, total_reward, terminated, truncated, info
    
    def render(self):
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill((240, 240, 240))

        def world_to_pixels(coords):
            px = (coords[0] + self.world_width / 2) * self.pixels_per_meter
            py = (-coords[1] + self.world_height / 2) * self.pixels_per_meter
            return (px, py)

        # Draw parking lines
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

        # Target
        target_poly_pixels = [world_to_pixels(p) for p in self.target_polygon.exterior.coords]
        pygame.draw.polygon(canvas, (144, 238, 144), target_poly_pixels)
        pygame.draw.polygon(canvas, (0, 128, 0), target_poly_pixels, 3)

        # Obstacles
        for obs_poly in self.obstacles:
            self._draw_car_top_view(canvas, obs_poly, (100, 100, 100), world_to_pixels)

        # Agent
        car_poly = Polygon(self._get_car_polygon(self.state))
        self._draw_car_top_view(canvas, car_poly, (70, 130, 180), world_to_pixels, is_agent=True)

        # Draw Lidar rays
        if self.render_mode == "human":
            obs_dict = self._get_obs_dict()
            lidar_distances = obs_dict["lidar_distances"]
            car_x, car_y, car_theta = self.state[:3]
            start_pixel = world_to_pixels((car_x, car_y))
            
            for i, dist in enumerate(lidar_distances):
                ray_global_angle = car_theta + self.LIDAR_ANGLES[i]
                end_x = car_x + dist * np.cos(ray_global_angle)
                end_y = car_y + dist * np.sin(ray_global_angle)
                end_pixel = world_to_pixels((end_x, end_y))
                color = (255, 0, 0) if dist < self.LIDAR_MAX_RANGE - 0.1 else (0, 150, 0)
                pygame.draw.line(canvas, color, start_pixel, end_pixel, 1)

        # Info overlay
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

    # Drawing helper
    def _draw_car_top_view(self, canvas, car_poly, color, world_to_pixels, is_agent=False):
        poly_pixels = [world_to_pixels(p) for p in car_poly.exterior.coords]
        pygame.draw.polygon(canvas, color, poly_pixels)
        pygame.draw.polygon(canvas, (50, 50, 50), poly_pixels, 2)
        
        coords = list(car_poly.exterior.coords)
        if len(coords) < 4:
            return
        
        rear_right_px = world_to_pixels(coords[0])
        rear_left_px = world_to_pixels(coords[1])
        front_left_px = world_to_pixels(coords[2])
        front_right_px = world_to_pixels(coords[3])
        
        if is_agent:
            pygame.draw.circle(canvas, (255, 220, 0), front_left_px, 4)
            pygame.draw.circle(canvas, (255, 220, 0), front_right_px, 4)
            pygame.draw.line(canvas, (200, 0, 0), rear_left_px, rear_right_px, 5)

            mirror_left_x = coords[1][0] * 0.25 + coords[2][0] * 0.75
            mirror_left_y = coords[1][1] * 0.25 + coords[2][1] * 0.75
            mid_left_pixel = world_to_pixels((mirror_left_x, mirror_left_y))
            pygame.draw.circle(canvas, (50, 50, 50), mid_left_pixel, 3)

            mirror_right_x = coords[0][0] * 0.25 + coords[3][0] * 0.75
            mirror_right_y = coords[0][1] * 0.25 + coords[3][1] * 0.75
            mid_right_pixel = world_to_pixels((mirror_right_x, mirror_right_y))
            pygame.draw.circle(canvas, (50, 50, 50), mid_right_pixel, 3)

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
    
    # Ray hit helper (kept for compatibility)
    def _ray_hits_polygon(self, ray_start_point, ray_end_point, polygon):
        ray_line = LineString([ray_start_point, ray_end_point])
        
        if not ray_line.intersects(polygon):
            return None
        
        intersection = ray_line.intersection(polygon)
        min_dist = float('inf')
        
        if intersection.geom_type == 'Point':
            min_dist = ray_start_point.distance(intersection)
        elif intersection.geom_type in ('MultiPoint', 'LineString'):
            for pt_coord in intersection.coords:
                dist = ray_start_point.distance(Point(pt_coord))
                min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else None

    def _get_obs_dict(self):
        target_center_x, target_center_y, target_angle = self.target_pose
        car_x, car_y, car_theta, car_v = self.state
        car_theta = (car_theta + np.pi) % (2 * np.pi) - np.pi
        
        # Relative vector to target (dx, dy)
        dx, dy = target_center_x - car_x, target_center_y - car_y
        dist_to_target = np.sqrt(dx**2 + dy**2)
        
        if dist_to_target > 0.01:
            target_dir_x = dx / dist_to_target
            target_dir_y = dy / dist_to_target
        else:
            target_dir_x = target_dir_y = 0.0
        
        angle_to_target = np.arctan2(dy, dx)
        angle_error = angle_to_target - car_theta
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        
        # Lidar
        lidar_distances = np.full(self.NUM_LIDAR_RAYS, self.LIDAR_MAX_RANGE, dtype=np.float32)
        car_pos_point = Point(car_x, car_y)
        all_obstacles = self.obstacles + self.world_boundaries
        
        for i, angle_offset in enumerate(self.LIDAR_ANGLES):
            ray_global_angle = car_theta + angle_offset
            ray_end_x = car_x + self.LIDAR_MAX_RANGE * np.cos(ray_global_angle)
            ray_end_y = car_y + self.LIDAR_MAX_RANGE * np.sin(ray_global_angle)
            ray_end_point = Point(ray_end_x, ray_end_y)
            ray_line = LineString([car_pos_point, ray_end_point])
            
            min_hit_dist = self.LIDAR_MAX_RANGE
            for obs in all_obstacles:
                if ray_line.intersects(obs):
                    intersection = ray_line.intersection(obs)
                    if intersection.geom_type == 'Point':
                        dist = car_pos_point.distance(intersection)
                        min_hit_dist = min(min_hit_dist, dist)
                    elif intersection.geom_type in ('MultiPoint', 'LineString'):
                        closest_pt = min(
                            (Point(c) for c in intersection.coords),
                            key=lambda p: car_pos_point.distance(p),
                            default=None
                        )
                        if closest_pt:
                            min_hit_dist = min(min_hit_dist, car_pos_point.distance(closest_pt))
            lidar_distances[i] = min_hit_dist

        return {
            "car_x": car_x, "car_y": car_y, "car_theta": car_theta, "car_v": car_v,
            # expose relative target coordinates (dx, dy)
            "dx": dx, "dy": dy,
            "target_dir_x": target_dir_x, "target_dir_y": target_dir_y,
            "angle_error": angle_error,
            "lidar_distances": lidar_distances
        }

    def _obs_dict_to_array(self, obs_dict):
        return np.array([
            # Voiture (4)
            obs_dict["car_x"], obs_dict["car_y"], obs_dict["car_theta"], obs_dict["car_v"],
            # Relative target (dx, dy)
            obs_dict["dx"], obs_dict["dy"],
            # Direction towards target (2)
            obs_dict["target_dir_x"], obs_dict["target_dir_y"],
            # Angle error (1)
            obs_dict["angle_error"],
            # Lidar distances (8)
            *obs_dict["lidar_distances"]
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
        is_parked = (intersection / car_area) > 0.92  # Relaxed from 0.95 to 0.92
        is_stopped = np.abs(self.state[3]) < 0.1
        target_angle = self.target_pose[2]
        car_angle = self.state[2]
        angle_diff = np.abs(target_angle - car_angle)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)
        is_aligned = angle_diff < 0.2
        return is_parked and is_stopped and is_aligned

    def _get_zone(self, distance):
        if distance > 20:
            return 1
        elif distance > 10:
            return 2
        elif distance > 5:
            return 3
        else:
            return 4

    def _calculate_reward(self, action, collision, out_of_bounds, success, obs_dict):
        reward_info = {
            "reward_progress": 0.0,
            "reward_alignment": 0.0,
            "reward_safety": 0.0,
            "reward_step_cost": 0.0,
            "reward_vel_near_goal": 0.0,
            "reward_collision": 0.0,
            "reward_out_of_bounds": 0.0,
            "reward_success": 0.0,
            "reward_in_spot": 0.0,  # NEW: Intermediate reward
            "raw_total": 0.0,
            "clipped_total": 0.0
        }

        # current quantities
        target_center = self.target_pose[:2]
        car_center = self.state[:2]
        current_dist = float(np.linalg.norm(target_center - car_center))

        lidar_distances = obs_dict["lidar_distances"].astype(np.float32)
        min_lidar_dist = float(np.min(lidar_distances))

        angle_error = float(obs_dict.get("angle_error", 0.0))
        speed = float(abs(self.state[3]))

        # progress reward (distance reduction)
        if self.prev_dist_to_target is not None:
            delta_dist = float(self.prev_dist_to_target - current_dist)
            reward_info["reward_progress"] = delta_dist * self.progress_scale

        # alignment reward only when inside activation distance
        if current_dist <= self.angle_activation_dist:
            reward_info["reward_alignment"] = self.alignment_weight * abs(angle_error)

        # safety penalty from lidar (quadratic), capped
        if min_lidar_dist < self.safety_threshold:
            raw_safety = -((self.safety_threshold - min_lidar_dist) ** 2) * self.safety_scale
            reward_info["reward_safety"] = max(raw_safety, self.safety_cap)

        # small per-step/time penalty
        reward_info["reward_step_cost"] = self.step_penalty

        # NEW: Intermediate reward for being in the parking spot (even if not perfect)
        # This helps the agent learn to enter the spot
        car_poly = Polygon(self._get_car_polygon(self.state))
        intersection = car_poly.intersection(self.target_polygon).area
        car_area = car_poly.area
        overlap_ratio = intersection / car_area if car_area > 0 else 0
        
        # Give bonus for being partially in the spot
        if overlap_ratio > 0.3:  # At least 30% in the spot
            reward_info["reward_in_spot"] = 1.0  # Good bonus for attempting to park
        else:
            reward_info["reward_in_spot"] = 0.0

        # penalize high speed near goal
        if current_dist < self.goal_slow_radius:
            reward_info["reward_vel_near_goal"] = self.velocity_near_goal_penalty * speed

        # terminal terms
        if collision:
            reward_info["reward_collision"] = self.collision_penalty

        if out_of_bounds:
            reward_info["reward_out_of_bounds"] = self.out_of_bounds_penalty

        if success:
            reward_info["reward_success"] = self.success_reward

        # update previous dist
        self.prev_dist_to_target = current_dist

        # sum raw and clip - but ONLY clip per-step rewards, NOT terminals
        # This is CRITICAL: terminal rewards (success, collision) must not be clipped
        # or the agent can't distinguish between success and small progress
        raw_step_reward = (
            reward_info["reward_progress"]
            + reward_info["reward_alignment"]
            + reward_info["reward_safety"]
            + reward_info["reward_step_cost"]
            + reward_info["reward_vel_near_goal"]
            + reward_info["reward_in_spot"]  # Intermediate reward
        )
        
        # Clip only the per-step continuous rewards
        clipped_step_reward = float(np.clip(raw_step_reward, -self.per_step_clip, self.per_step_clip))
        
        # Add terminal rewards WITHOUT clipping - this preserves the learning signal
        total_reward = clipped_step_reward
        if collision:
            total_reward += self.collision_penalty  # NOT clipped! Full -50 penalty
        if out_of_bounds:
            total_reward += self.out_of_bounds_penalty  # NOT clipped! Full -30 penalty
        if success:
            total_reward += self.success_reward  # NOT clipped! Full +500 reward
        
        # Update info dict for logging
        reward_info["raw_total"] = float(
            raw_step_reward + 
            reward_info["reward_collision"] + 
            reward_info["reward_out_of_bounds"] + 
            reward_info["reward_success"]
        )
        reward_info["clipped_total"] = total_reward
        
        return total_reward, reward_info
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()