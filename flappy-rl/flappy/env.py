"""Gymnasium environment for the Flappy RL project."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .physics import (
    BIRD_X,
    GAP_HEIGHT,
    HEIGHT,
    MAX_VY,
    PIPE_VX,
    PhysicsConfig,
    WIDTH,
    bird_rect,
    collides,
    init_state,
    passed_pipe,
    pipe_rects,
    step_dynamics,
)
from .render import Renderer
from .utils import line_aabb_intersection, make_rng


class FlappyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        *,
        use_rays: bool = False,
        n_rays: int = 7,
        three_flaps: bool = False,
        wind: bool = False,
        moving_pipes: bool = False,
        energy: bool = False,
        gap_height_range: tuple[float, float] | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.use_rays = use_rays
        self.n_rays = max(1, n_rays)
        self.three_flaps = three_flaps
        self.wind = wind
        self.moving_pipes = moving_pipes
        self.energy = energy
        self._gap_height_range = gap_height_range
        self.render_mode = render_mode
        self.max_steps: int | None = None

        self._cfg = PhysicsConfig(
            three_flaps=three_flaps,
            wind=wind,
            moving_pipes=moving_pipes,
            energy=energy,
            pipe_vx=PIPE_VX,
        )
        if gap_height_range is not None:
            self._cfg.gap_height_range = gap_height_range
            self._gap_height_range = gap_height_range

        self._rng = make_rng(seed)
        self._renderer: Renderer | None = None
        self._state: dict[str, Any] | None = None
        self._steps = 0
        self._prev_dist = 0.0
        self._wind_est = 0.0

        self._obs_dim = self._compute_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        n_actions = 3 if three_flaps else 2
        self.action_space = spaces.Discrete(n_actions)

    def _compute_obs_dim(self) -> int:
        ray_features = self.n_rays if self.use_rays else 2
        obs_dim = 3 + ray_features + 1  # bird state + gap/rays + pipe vx
        obs_dim += 1  # slot for wind or energy
        if self.wind and self.energy:
            obs_dim += 1  # extra slot to keep both
        return obs_dim

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = make_rng(seed)
        self._state = init_state(self._rng)
        self._state["pipes_passed"] = 0
        self._steps = 0
        self._prev_dist = abs(self._state["bird_y"] - self._state["gap_center_y"])
        self._wind_est = 0.0
        obs = self._get_obs(self._state)
        info = {"passed": False, "dist": self._prev_dist, "pipes": 0}
        return obs, info

    def step(self, action: int):
        assert self._state is not None, "Environment must be reset before stepping"
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        self._steps += 1
        prev_state = self._state
        new_state = step_dynamics(prev_state, action, self._cfg, self._rng)

        pipe_cross = passed_pipe(prev_state, new_state)
        if pipe_cross:
            new_state["pipes_passed"] = prev_state.get("pipes_passed", 0) + 1
        else:
            new_state["pipes_passed"] = prev_state.get("pipes_passed", 0)

        bird_bb = bird_rect(new_state)
        pipe_bb = pipe_rects(new_state)
        new_state["pipe_rects"] = pipe_bb
        new_state["bird_pos"] = (BIRD_X, new_state["bird_y"])

        crash = collides(bird_bb, pipe_bb)
        out_of_bounds = new_state["bird_y"] <= 0.0 or new_state["bird_y"] >= HEIGHT
        terminated = crash or out_of_bounds
        truncated = False if self.max_steps is None else self._steps >= self.max_steps

        dist = abs(new_state["bird_y"] - new_state["gap_center_y"])
        prev_dist = self._prev_dist if self._steps > 1 else dist
        flap_used = new_state.get("last_flap", 0.0)
        reward = self._compute_reward(dist, prev_dist, flap_used, pipe_cross, crash)

        self._state = new_state
        self._prev_dist = dist
        obs = self._get_obs(new_state)
        info = {"passed": pipe_cross, "dist": dist, "pipes": new_state["pipes_passed"]}
        return obs, reward, terminated, truncated, info

    def _get_obs(self, state: dict) -> np.ndarray:
        bird_y = np.clip(state["bird_y"] / HEIGHT * 2.0 - 1.0, -1.0, 1.0)
        bird_vy = np.clip(state["bird_vy"] / MAX_VY, -1.0, 1.0)
        dx_norm = np.clip((state["x_pipe"] - BIRD_X) / WIDTH, -1.0, 1.0)

        features: list[float] = [bird_y, bird_vy, dx_norm]

        if self.use_rays:
            ray_vals = self._ray_observations(state)
            if len(ray_vals) < self.n_rays:
                ray_vals = ray_vals + [1.0] * (self.n_rays - len(ray_vals))
            features.extend(ray_vals)
        else:
            gap_center = (state["gap_center_y"] - HEIGHT / 2) / (HEIGHT / 2)
            gap_height = (state["gap_height"] - GAP_HEIGHT) / GAP_HEIGHT
            features.extend([gap_center, gap_height])

        pipe_vx_norm = np.clip(state["pipe_vx"] / abs(PIPE_VX), -1.0, 1.0)
        features.append(pipe_vx_norm)

        wind_slot = 0.0
        if self.wind:
            self._wind_est = 0.9 * self._wind_est + 0.1 * state.get("wind", 0.0)
            wind_slot = np.clip(self._wind_est, -1.0, 1.0)
        elif self.energy:
            wind_slot = np.clip(state.get("energy", 1.0) * 2.0 - 1.0, -1.0, 1.0)
        features.append(wind_slot)

        if self.wind and self.energy:
            features.append(np.clip(state.get("energy", 1.0) * 2.0 - 1.0, -1.0, 1.0))

        return np.array(features, dtype=np.float32)

    def _ray_observations(self, state: dict) -> list[float]:
        origin = np.array([BIRD_X, state["bird_y"]], dtype=float)
        max_d = WIDTH
        angles = np.linspace(-math.pi / 4, math.pi / 4, self.n_rays)
        readings: list[float] = []
        rects = pipe_rects(state)
        for angle in angles:
            direction = np.array([math.cos(angle), math.sin(angle)])
            end = origin + direction * max_d
            best = max_d
            for rect in rects:
                hit, dist = line_aabb_intersection(origin, end, rect)
                if hit and dist < best:
                    best = dist
            readings.append(best / max_d)
        return readings

    def render(self):
        if self.render_mode != "human" or self._state is None:
            return None
        if self._renderer is None:
            self._renderer = Renderer(WIDTH, HEIGHT)
        draw_state = {
            "pipe_rects": pipe_rects(self._state),
            "bird_pos": (BIRD_X, self._state["bird_y"]),
            "bird_angle": self._state.get("bird_angle", 0.0),
            "wing_phase": self._state.get("wing_phase", 0.0),
            "wind": self._state.get("wind", 0.0) if self.wind else 0.0,
            "pipes_passed": self._state.get("pipes_passed", 0),
            "t": self._state.get("t", 0),
        }
        return self._renderer.draw(draw_state)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def apply_settings(
        self,
        *,
        gap_height_range: tuple[float, float] | None = None,
        moving_pipes: bool | None = None,
        wind: bool | None = None,
    ) -> None:
        """Update runtime difficulty parameters without recreating the env."""
        if gap_height_range is not None:
            self._cfg.gap_height_range = gap_height_range
        if moving_pipes is not None:
            self.moving_pipes = moving_pipes
            self._cfg.moving_pipes = moving_pipes
        if wind is not None:
            self.wind = wind
            self._cfg.wind = wind

    def _compute_reward(
        self,
        dist: float,
        prev_dist: float,
        flap_used: float,
        pipe_cross: bool,
        crash: bool,
    ) -> float:
        r_step = 0.05 + 0.15 * (prev_dist - dist) - 0.002 * flap_used
        if self.energy:
            r_step -= 0.001 * flap_used
        r_pipe = 1.0 if pipe_cross else 0.0
        r_crash = -1.0 if crash else 0.0
        return float(np.clip(r_step + r_pipe + r_crash, -1.0, 1.0))

__all__ = ["FlappyEnv"]
