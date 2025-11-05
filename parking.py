import gymnasium as gym
import numpy as np
from shapely.geometry import Polygon, LineString, Point
import pygame
from collections import deque

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
        self.VELOCITY_DECAY = 0.995

        # parking layout
        self.SPOT_WIDTH = self.CAR_WIDTH * 1.5
        self.SPOT_DEPTH = self.CAR_LENGTH * 1.2
        self.AISLE_WIDTH = 12.0
        self.CIRCULATION_ZONE = 15.0
        self.NUM_COLUMNS = 1
        self.CARS_PER_COLUMN = 10

        # lidar
        self.NUM_LIDAR_RAYS = 8
        self.LIDAR_MAX_RANGE = 20.0
        self.LIDAR_ANGLES = np.linspace(0, 2 * np.pi, self.NUM_LIDAR_RAYS, endpoint=False)

        # agent state
        self.state = np.zeros(4, dtype=np.float32)

        # actions
        self.action_space = gym.spaces.Box(
            low=np.array([-self.MAX_ACCEL, -self.MAX_STEER_ANGLE], dtype=np.float32),
            high=np.array([self.MAX_ACCEL, self.MAX_STEER_ANGLE], dtype=np.float32),
            shape=(2,)
        )

        # world size
        parking_width = (self.NUM_COLUMNS * self.SPOT_DEPTH) + ((self.NUM_COLUMNS - 1) * self.AISLE_WIDTH)
        parking_height = (self.CARS_PER_COLUMN * self.SPOT_WIDTH)
        self.world_width = parking_width + 2 * self.CIRCULATION_ZONE + self.AISLE_WIDTH
        self.world_height = parking_height + 2 * self.CIRCULATION_ZONE
        self.max_distance = max(self.world_width, self.world_height) + 10.0

        # world boundaries
        w, h = self.world_width / 2, self.world_height / 2
        self.world_boundaries = [
            LineString([(-w, -h), (w, -h)]),
            LineString([(w, -h), (w, h)]),
            LineString([(w, h), (-w, h)]),
            LineString([(-w, h), (-w, -h)])
        ]

        # observation space
        base_low = np.array(
            [-np.inf, -np.inf, -np.pi, -self.MAX_VELOCITY,
             -np.inf, -np.inf,
             -1.0, -1.0,
             -np.pi],
            dtype=np.float32
        )
        base_high = np.array(
            [np.inf, np.inf, np.pi, self.MAX_VELOCITY,
             np.inf, np.inf,
             1.0, 1.0,
             np.pi],
            dtype=np.float32
        )
        lidar_low = np.zeros(self.NUM_LIDAR_RAYS, dtype=np.float32)
        lidar_high = np.full(self.NUM_LIDAR_RAYS, self.LIDAR_MAX_RANGE, dtype=np.float32)
        extra_low = np.array(
            [-1.0, -1.0,
             0.0, 0.0, 0.0,
             -self.max_distance, -self.max_distance,
             -np.pi,
             0.0,
             -self.max_distance, -self.max_distance,
             -np.pi,
             0.0],
            dtype=np.float32
        )
        extra_high = np.array(
            [1.0, 1.0,
             self.max_distance, self.max_distance, self.max_distance,
             self.max_distance, self.max_distance,
             np.pi,
             1.0,
             self.max_distance, self.max_distance,
             np.pi,
             1.5],
            dtype=np.float32
        )
        low = np.concatenate([base_low, lidar_low, extra_low]).astype(np.float32)
        high = np.concatenate([base_high, lidar_high, extra_high]).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # variables
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
        self._last_canvas = None
        self._last_rgb_frame = None

        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Parking")
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

        # ===== Simple reward tuning =====
        self.progress_scale = 10.0
        self.heading_progress_scale = 5.0
        self.speed_penalty_scale = 0.1
        self.lateral_penalty_scale = 1.0
        self.step_penalty = -0.01
        self.success_reward = 200.0
        self.collision_penalty = -40.0
        self.out_of_bounds_penalty = -40.0

        # Spawn setup (kept simple)
        self.gate_offset = 1.0
        self.switch_radius = 2.0
        self.spawn_offset_range = (0.5, 2.0)
        self.spawn_lateral_span = self.SPOT_WIDTH * 1.5
        self.easy_spawn_prob = 1.0
        self.easy_spawn_jitter = 0.03
        # success criteria
        self.success_overlap = 0.75
        self.success_angle = 0.35
        self.success_speed = 0.2
        self.per_step_clip = 0.7

        # Internals for reward/episode tracking
        self._prev_spot_dist = None
        self._prev_heading_error = None
        self._best_distance = None
        self._last_improve_step = 0
        self._step_count = 0
        self.improve_epsilon = 0.1
        self.max_steps_no_improve = 1200
    # ---------------- core Gym API ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # build rows and target
        self.obstacles = []
        self.obstacle_poses = []
        self.empty_spot_index = 7

        parking_width = (self.NUM_COLUMNS * self.SPOT_DEPTH) + ((self.NUM_COLUMNS - 1) * self.AISLE_WIDTH)
        start_x = -parking_width / 2

        spot_idx = 0
        for col in range(self.NUM_COLUMNS):
            col_x = start_x + col * (self.SPOT_DEPTH + self.AISLE_WIDTH) + self.SPOT_DEPTH / 2
            for row in range(self.CARS_PER_COLUMN):
                spot_y = -self.world_height/2 + self.CIRCULATION_ZONE + row * self.SPOT_WIDTH + self.SPOT_WIDTH/2
                if spot_idx == self.empty_spot_index:
                    self.target_pose = np.array([col_x, spot_y, np.pi])
                    half_d, half_w = self.SPOT_DEPTH/2, self.SPOT_WIDTH/2
                    coords = [
                        (col_x - half_d, spot_y - half_w),
                        (col_x - half_d, spot_y + half_w),
                        (col_x + half_d, spot_y + half_w),
                        (col_x + half_d, spot_y - half_w)
                    ]
                    self.target_polygon = Polygon(coords)
                else:
                    car_state = np.array([col_x, spot_y, 0.0, 0.0])
                    self.obstacles.append(Polygon(self._get_car_polygon(car_state)))
                    self.obstacle_poses.append([col_x, spot_y])
                spot_idx += 1

        gate_xy_reset = self._gate_point()

        # spawn perpendicular in aisle, facing ±y (like before)
        corridor_min = -self.world_height/2 + self.CIRCULATION_ZONE + self.CAR_LENGTH
        corridor_max = self.world_height/2 - self.CIRCULATION_ZONE - self.CAR_LENGTH
        band_min = self.target_pose[1] - self.spawn_lateral_span
        band_max = self.target_pose[1] + self.spawn_lateral_span
        y_low = max(corridor_min, band_min)
        y_high = min(corridor_max, band_max)
        if y_high <= y_low:
            y_low, y_high = corridor_min, corridor_max

        for _ in range(100):
            y = self.np_random.uniform(y_low, y_high)
            offset = self.np_random.uniform(*self.spawn_offset_range)
            x = self.AISLE_WIDTH / 2 + offset
            easy_mode = self.np_random.random() < self.easy_spawn_prob
            if easy_mode:
                desired = np.arctan2(gate_xy_reset[1] - y, gate_xy_reset[0] - x)
                base_angle = desired
                angle_variation = self.np_random.uniform(-self.easy_spawn_jitter, self.easy_spawn_jitter)
            else:
                base_angle = self.np_random.choice([np.pi/2, -np.pi/2])
                angle_variation = self.np_random.uniform(-0.15, 0.15)
            theta = base_angle + angle_variation
            theta = np.arctan2(np.sin(theta), np.cos(theta))
            self.state = np.array([x, y, theta, 0.0], dtype=np.float32)
            if not self._check_collision() and not self._check_out_of_bounds():
                break

        # reset trackers
        self._prev_spot_dist = None
        self._prev_heading_error = None
        self._best_distance = None
        self._last_improve_step = 0
        self._step_count = 0
        # compute initial distance to SPOT center (for best-distance tracking)
        target_center = self.target_pose[:2]
        self._best_distance = float(np.linalg.norm(self.state[:2] - target_center))
        self._prev_spot_dist = self._best_distance
        self._last_improve_step = 0

        obs_dict = self._get_obs_dict()
        observation = self._obs_dict_to_array(obs_dict)
        info = {
            "state": self.state.copy(),
            "start_state": self.state.copy(),
            "target_x": self.target_pose[0],
            "target_y": self.target_pose[1]
        }
        return observation, info

    def step(self, action):
        self._dynamic_step(action)
        self._step_count += 1

        collision = self._check_collision()
        out_of_bounds = self._check_out_of_bounds()
        success = self._check_success()

        obs_dict = self._get_obs_dict()
        observation = self._obs_dict_to_array(obs_dict)

        total_reward, reward_info = self._calculate_reward(collision, out_of_bounds, success, obs_dict)

        # base terminals
        terminated = collision or success or out_of_bounds

        # anti-stall truncation
        truncated = False
        if not terminated:
            if (self._step_count - self._last_improve_step) >= self.max_steps_no_improve:
                truncated = True

        info = {
            "state": self.state.copy(),
            "target_x": self.target_pose[0],
            "target_y": self.target_pose[1],
            **reward_info
        }
        return observation, total_reward, terminated, truncated, info

    # ---------------- rendering ----------------
    def render(self):
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill((240, 240, 240))

        def world_to_pixels(coords):
            px = (coords[0] + self.world_width / 2) * self.pixels_per_meter
            py = (-coords[1] + self.world_height / 2) * self.pixels_per_meter
            return (px, py)

        # parking lines
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

        # target spot
        target_poly_pixels = [world_to_pixels(p) for p in self.target_polygon.exterior.coords]
        pygame.draw.polygon(canvas, (144, 238, 144), target_poly_pixels)
        pygame.draw.polygon(canvas, (0, 128, 0), target_poly_pixels, 3)

        # obstacles
        for obs_poly in self.obstacles:
            self._draw_car_top_view(canvas, obs_poly, (100, 100, 100), world_to_pixels)

        # agent
        car_poly = Polygon(self._get_car_polygon(self.state))
        self._draw_car_top_view(canvas, car_poly, (70, 130, 180), world_to_pixels, is_agent=True)

        # Gate marker (for debugging/visualization)
        gate = self._gate_point()
        gx, gy = world_to_pixels((gate[0], gate[1]))
        pygame.draw.circle(canvas, (255, 140, 0), (int(gx), int(gy)), 5)

        # Lidar rays (if human)
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

        # overlay
        if self.font:
            target_dist = np.linalg.norm(self.state[:2] - self.target_pose[:2])
            gate_dist = np.linalg.norm(self.state[:2] - self._gate_point())
            info_text = [
                f"Dist->spot: {target_dist:.1f} m",
                f"Dist->gate: {gate_dist:.1f} m",
                f"Best dist:  {self._best_distance:.1f} m",
                f"Speed: {abs(self.state[3]):.1f} m/s"
            ]
            for i, text in enumerate(info_text):
                text_surface = self.font.render(text, True, (0, 0, 0))
                canvas.blit(text_surface, (10, 10 + i * 25))

        # keep latest frame for diagnostics/saving
        self._last_canvas = canvas.copy()
        self._last_rgb_frame = np.transpose(pygame.surfarray.pixels3d(self._last_canvas), axes=(1, 0, 2)).copy()

        if self.render_mode == "human":
            self.screen.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:
            return self._last_rgb_frame.copy()

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

    # ---------------- physics ----------------
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

    # ---------------- observations ----------------
    def _get_obs_dict(self):
        car_x, car_y, car_theta, car_v = self.state
        cos_t, sin_t = np.cos(car_theta), np.sin(car_theta)

        target_center = self.target_pose[:2]
        dx = target_center[0] - car_x
        dy = target_center[1] - car_y
        dist_to_target = np.linalg.norm([dx, dy])

        if dist_to_target > 0.01:
            target_dir_x = dx / dist_to_target
            target_dir_y = dy / dist_to_target
        else:
            target_dir_x = target_dir_y = 0.0

        angle_to_target = np.arctan2(dy, dx)
        angle_error = angle_to_target - car_theta
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

        car_xy = np.array([car_x, car_y], dtype=float)
        subgoal_xy, use_gate, dist_gate, dist_spot = self._compute_subgoal(car_xy)
        rel_vec = subgoal_xy - car_xy
        forward_err = float(rel_vec[0] * cos_t + rel_vec[1] * sin_t)
        lateral_err = float(-rel_vec[0] * sin_t + rel_vec[1] * cos_t)
        dist_to_subgoal = float(np.linalg.norm(rel_vec))
        angle_to_subgoal = float(np.arctan2(lateral_err, forward_err))
        spot_xy = self.target_pose[:2].astype(float)
        spot_rel = car_xy - spot_xy
        cos_target = np.cos(self.target_pose[2])
        sin_target = np.sin(self.target_pose[2])
        spot_forward = float(spot_rel[0] * cos_target + spot_rel[1] * sin_target)
        spot_lateral = float(-spot_rel[0] * sin_target + spot_rel[1] * cos_target)
        spot_heading_error = float(np.arctan2(
            np.sin(car_theta - self.target_pose[2]),
            np.cos(car_theta - self.target_pose[2])
        ))
        car_poly = Polygon(self._get_car_polygon(self.state))
        overlap_area = car_poly.intersection(self.target_polygon).area
        car_area = car_poly.area if car_poly.area > 1e-9 else 1.0
        overlap_ratio = float(np.clip(overlap_area / car_area, 0.0, 1.5))

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
            "car_x": car_x,
            "car_y": car_y,
            "car_theta": car_theta,
            "car_v": car_v,
            "dx": dx,
            "dy": dy,
            "target_dir_x": target_dir_x,
            "target_dir_y": target_dir_y,
            "angle_error": angle_error,
            "lidar_distances": lidar_distances,
            "heading_sin": float(sin_t),
            "heading_cos": float(cos_t),
            "dist_to_gate": dist_gate,
            "dist_to_spot": dist_spot,
            "dist_to_subgoal": dist_to_subgoal,
            "subgoal_forward": forward_err,
            "subgoal_lateral": lateral_err,
            "angle_to_subgoal": angle_to_subgoal,
            "use_gate_flag": 1.0 if use_gate else 0.0,
            "spot_forward": spot_forward,
            "spot_lateral": spot_lateral,
            "spot_heading_error": spot_heading_error,
            "overlap_ratio": overlap_ratio
        }

    def _obs_dict_to_array(self, obs_dict):
        return np.array([
            obs_dict["car_x"], obs_dict["car_y"], obs_dict["car_theta"], obs_dict["car_v"],
            obs_dict["dx"], obs_dict["dy"],
            obs_dict["target_dir_x"], obs_dict["target_dir_y"],
            obs_dict["angle_error"],
            *obs_dict["lidar_distances"],
            obs_dict["heading_sin"], obs_dict["heading_cos"],
            obs_dict["dist_to_gate"], obs_dict["dist_to_spot"], obs_dict["dist_to_subgoal"],
            obs_dict["subgoal_forward"], obs_dict["subgoal_lateral"],
            obs_dict["angle_to_subgoal"],
            obs_dict["use_gate_flag"],
            obs_dict["spot_forward"], obs_dict["spot_lateral"],
            obs_dict["spot_heading_error"], obs_dict["overlap_ratio"]
        ], dtype=np.float32)

    # ---------------- geometry & checks ----------------
    def _get_car_polygon(self, state):
        x, y, theta, v = state
        half_len, half_wid = self.CAR_LENGTH / 2, self.CAR_WIDTH / 2
        corners = [(-half_len, -half_wid), (-half_len, +half_wid),
                   (+half_len, +half_wid), (+half_len, -half_wid)]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        transformed = []
        for (cx, cy) in corners:
            tx = x + (cx * cos_t - cy * sin_t)
            ty = y + (cx * sin_t + cy * cos_t)
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
        overlap = (intersection / car_area) if car_area > 0 else 0.0

        is_parked = overlap > self.success_overlap
        is_stopped = np.abs(self.state[3]) < self.success_speed
        target_angle = self.target_pose[2]
        car_angle = self.state[2]
        angle_diff = np.abs(target_angle - car_angle)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)
        is_aligned = angle_diff < self.success_angle
        return is_parked and is_stopped and is_aligned

    # ---------------- reward ----------------
    def _gate_point(self):
        """
        Compute a 'gate' point slightly outside the spot opening facing the aisle.
        With our layout (spawn at x > 0, spot near x ~ 0), the opening is at x = center_x + half_depth.
        Place gate further out by gate_offset.
        """
        cx, cy = self.target_pose[0], self.target_pose[1]
        half_d = self.SPOT_DEPTH / 2
        gate_x = cx + half_d + self.gate_offset
        gate_y = cy
        return np.array([gate_x, gate_y], dtype=float)

    def _compute_subgoal(self, car_xy):
        """
        Determine whether to target the gate (entry) or the parking spot center.
        Returns (subgoal_xy, use_gate, distance_to_gate, distance_to_spot).
        """
        spot_xy = self.target_pose[:2].astype(float)
        gate_xy = self._gate_point()
        dist_gate = float(np.linalg.norm(car_xy - gate_xy))
        dist_spot = float(np.linalg.norm(car_xy - spot_xy))
        use_gate = dist_gate > self.switch_radius
        subgoal_xy = gate_xy if use_gate else spot_xy
        return subgoal_xy.astype(float), use_gate, dist_gate, dist_spot

    def _calculate_reward(self, collision, out_of_bounds, success, obs_dict):
        reward_info = {
            "r_progress": 0.0,
            "r_heading": 0.0,
            "r_lateral": 0.0,
            "r_speed": 0.0,
            "r_step": self.step_penalty,
            "r_success": 0.0,
            "r_collision": 0.0,
            "r_oob": 0.0,
            "dist_to_spot": 0.0,
            "heading_error": 0.0,
            "lateral_offset": 0.0,
        }

        car_xy = self.state[:2].astype(float)
        spot_xy = self.target_pose[:2].astype(float)
        dist_spot = float(np.linalg.norm(car_xy - spot_xy))
        reward_info["dist_to_spot"] = dist_spot

        target_heading = self.target_pose[2]
        angle_error = target_heading - self.state[2]
        angle_error = float(np.arctan2(np.sin(angle_error), np.cos(angle_error)))
        heading_error = abs(angle_error)
        reward_info["heading_error"] = heading_error

        progress_delta = 0.0 if self._prev_spot_dist is None else self._prev_spot_dist - dist_spot
        r_progress = self.progress_scale * progress_delta
        heading_delta = 0.0 if self._prev_heading_error is None else self._prev_heading_error - heading_error
        r_heading = self.heading_progress_scale * heading_delta
        speed = abs(float(self.state[3]))
        r_speed = -self.speed_penalty_scale * speed
        target_heading = self.target_pose[2]
        right_vec = np.array([-np.sin(target_heading), np.cos(target_heading)], dtype=float)
        lateral_offset = abs(np.dot(car_xy - spot_xy, right_vec))
        r_lateral = -self.lateral_penalty_scale * lateral_offset

        reward_info["r_progress"] = r_progress
        reward_info["r_heading"] = r_heading
        reward_info["r_speed"] = r_speed
        reward_info["r_lateral"] = r_lateral
        reward_info["lateral_offset"] = lateral_offset

        total_reward = r_progress + r_heading + r_lateral + r_speed + self.step_penalty

        if progress_delta > 0.0:
            self._last_improve_step = self._step_count

        self._prev_spot_dist = dist_spot
        self._prev_heading_error = heading_error
        if self._best_distance is None or dist_spot < self._best_distance:
            self._best_distance = dist_spot

        if collision:
            reward_info["r_collision"] = self.collision_penalty
            total_reward += self.collision_penalty
        if out_of_bounds:
            reward_info["r_oob"] = self.out_of_bounds_penalty
            total_reward += self.out_of_bounds_penalty
        if success:
            reward_info["r_success"] = self.success_reward
            total_reward += self.success_reward

        return float(total_reward), reward_info

    # ---------------- utils ----------------
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def get_last_frame(self):
        if self._last_rgb_frame is None:
            return None
        return self._last_rgb_frame.copy()

    def save_last_frame(self, path):
        if self._last_canvas is None:
            raise RuntimeError("No frame available to save.")
        pygame.image.save(self._last_canvas, path)
