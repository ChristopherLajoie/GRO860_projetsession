"""Physics helpers for the Flappy RL environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .utils import clamp, ou_step

G = 0.45
FLAP_IMPULSE = 8.0
MAX_VY = 12.0
WIDTH = 400
HEIGHT = 600
PIPE_W = 60
PIPE_VX = -8.0
GAP_HEIGHT = 110
BIRD_X = 80
BIRD_SIZE = 24


@dataclass
class PhysicsConfig:
    three_flaps: bool = False
    wind: bool = False
    moving_pipes: bool = False
    energy: bool = False
    moving_amp: float = 35.0
    moving_omega: float = 0.05
    pipe_vx: float = PIPE_VX
    gap_height_range: tuple[float, float] = (GAP_HEIGHT - 15, GAP_HEIGHT + 15)
    gap_center_bounds: tuple[float, float] = (HEIGHT * 0.2, HEIGHT * 0.8)
    energy_cost: float = 0.12
    energy_regen: float = 0.01
    wind_mu: float = 0.25
    wind_theta: float = 0.008
    wind_sigma: float = 0.015
    wind_mag_clamp: float = 0.35
    wind_dir_flip_prob: float = 0.0
    wind_dir_min_steps: int = 9999
    pipe_speed_growth: float = 0.08


def init_state(rng: np.random.Generator) -> Dict[str, float]:
    """Return a freshly initialized simulation state."""
    gap_center = HEIGHT * 0.5
    gap_height = GAP_HEIGHT
    return {
        "bird_y": HEIGHT * 0.5,
        "bird_vy": 0.0,
        "x_pipe": WIDTH + 80.0,
        "gap_center_y": gap_center,
        "gap_base_y": gap_center,
        "gap_height": gap_height,
        "baseline_gap": GAP_HEIGHT,
        "pipe_vx": PIPE_VX,
        "wind": 0.0,
        "wind_mag": 0.0,
        "wind_dir": 1.0 if rng.random() < 0.5 else -1.0,
        "wind_dir_steps": 0,
        "energy": 1.0,
        "t": 0,
        "pipes_passed": 0,
        "last_flap": 0.0,
        "bird_angle": 0.0,
        "wing_phase": 0.0,
    }


def _sample_gap_height(cfg: PhysicsConfig, rng: np.random.Generator) -> float:
    lo, hi = cfg.gap_height_range
    return float(rng.uniform(lo, hi))


def _sample_gap_center(cfg: PhysicsConfig, rng: np.random.Generator) -> float:
    lo, hi = cfg.gap_center_bounds
    return float(rng.uniform(lo, hi))


def _flap_impulse(action: int, state: dict, cfg: PhysicsConfig) -> tuple[float, float]:
    if cfg.three_flaps:
        if action == 2:
            return 9.0, 1.0
        if action == 1:
            return 6.0, 0.6
        return 0.0, 0.0
    if action == 1:
        return FLAP_IMPULSE, 1.0
    return 0.0, 0.0


def update_moving_gap(state: dict, cfg: PhysicsConfig) -> None:
    base = state.get("gap_base_y", state["gap_center_y"])
    t = state.get("t", 0)
    offset = cfg.moving_amp * np.sin(cfg.moving_omega * t)
    new_center = clamp(base + offset, cfg.gap_center_bounds[0], cfg.gap_center_bounds[1])
    state["gap_center_y"] = new_center


def step_dynamics(state: dict, action: int, cfg: PhysicsConfig, rng: np.random.Generator) -> dict:
    """Advance the physics simulation by one step."""
    new_state = state.copy()
    new_state["t"] = state["t"] + 1

    impulse, flap_used = _flap_impulse(action, state, cfg)
    vy = state["bird_vy"] + G

    if impulse > 0.0:
        if cfg.energy:
            energy = clamp(state.get("energy", 1.0) - cfg.energy_cost * flap_used, 0.0, 1.0)
            new_state["energy"] = clamp(energy + cfg.energy_regen, 0.0, 1.0)
            impulse *= 0.5 + 0.5 * new_state["energy"]
        vy -= impulse
        new_state["last_flap"] = flap_used
    else:
        if cfg.energy:
            new_state["energy"] = clamp(state.get("energy", 1.0) + cfg.energy_regen, 0.0, 1.0)
        new_state["last_flap"] = 0.0

    if cfg.wind:
        wind_dir = state.get("wind_dir", 1.0)
        dir_steps = state.get("wind_dir_steps", 0) + 1
        if (
            cfg.wind_dir_flip_prob > 0.0
            and dir_steps >= cfg.wind_dir_min_steps
            and rng.random() < cfg.wind_dir_flip_prob
        ):
            wind_dir = 1.0 if rng.random() < 0.5 else -1.0
            dir_steps = 0
        wind_mag = ou_step(
            state.get("wind_mag", 0.0),
            rng,
            mu=cfg.wind_mu,
            sigma=cfg.wind_sigma,
            theta=cfg.wind_theta,
            clamp_value=cfg.wind_mag_clamp,
        )
        wind = wind_dir * wind_mag
    else:
        wind = 0.0
        wind_dir = state.get("wind_dir", 1.0)
        dir_steps = state.get("wind_dir_steps", 0)
        wind_mag = 0.0
    vy += wind
    vy = clamp(vy, -MAX_VY, MAX_VY)

    bird_y = clamp(state["bird_y"] + vy, 0.0, HEIGHT)

    progress = max(state.get("pipes_passed", 0), state.get("t", 0) / 800)
    speed_multiplier = 1.0 + cfg.pipe_speed_growth * progress
    pipe_vx = cfg.pipe_vx * speed_multiplier
    x_pipe = state["x_pipe"] + pipe_vx

    if x_pipe + PIPE_W < 0.0:
        x_pipe = WIDTH + rng.uniform(40.0, 120.0)
        pipe_vx = cfg.pipe_vx * speed_multiplier
        gap_height = _sample_gap_height(cfg, rng)
        gap_center = _sample_gap_center(cfg, rng)
        new_state["gap_height"] = gap_height
        new_state["gap_center_y"] = gap_center
        new_state["gap_base_y"] = gap_center
        new_state["baseline_gap"] = GAP_HEIGHT
    if cfg.moving_pipes:
        update_moving_gap(new_state, cfg)

    new_state["bird_y"] = bird_y
    new_state["bird_vy"] = vy
    new_state["wind"] = wind
    new_state["wind_mag"] = wind_mag
    new_state["wind_dir"] = wind_dir
    new_state["wind_dir_steps"] = dir_steps
    new_state["x_pipe"] = x_pipe
    new_state["pipe_vx"] = pipe_vx

    target_angle = clamp(-vy / MAX_VY, -1.0, 1.0) * 45.0
    prev_angle = state.get("bird_angle", 0.0)
    smoothing = 0.2
    new_state["bird_angle"] = (1 - smoothing) * prev_angle + smoothing * target_angle

    wing_phase = max(0.0, state.get("wing_phase", 0.0) - 0.05)
    if flap_used > 0.0:
        wing_phase = 1.0
    new_state["wing_phase"] = wing_phase

    return new_state


def passed_pipe(prev_state: dict, state: dict) -> bool:
    prev_front = prev_state["x_pipe"] + PIPE_W
    curr_front = state["x_pipe"] + PIPE_W
    return prev_front >= BIRD_X and curr_front < BIRD_X


def bird_rect(state: dict) -> tuple[float, float, float, float]:
    half = BIRD_SIZE / 2
    return (BIRD_X - half, state["bird_y"] - half, BIRD_SIZE, BIRD_SIZE)


def pipe_rects(state: dict) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    gap_center = state["gap_center_y"]
    gap_height = state["gap_height"]
    top_h = max(0.0, gap_center - gap_height / 2)
    bottom_y = gap_center + gap_height / 2
    bottom_h = max(0.0, HEIGHT - bottom_y)
    top_rect = (state["x_pipe"], 0.0, PIPE_W, top_h)
    bottom_rect = (state["x_pipe"], bottom_y, PIPE_W, bottom_h)
    return top_rect, bottom_rect


def collides(bird: tuple[float, float, float, float], pipes: tuple[tuple[float, float, float, float], tuple[float, float, float, float]]) -> bool:
    (bx, by, bw, bh) = bird
    for px, py, pw, ph in pipes:
        if ph <= 0:
            continue
        overlap_x = bx < px + pw and bx + bw > px
        overlap_y = by < py + ph and by + bh > py
        if overlap_x and overlap_y:
            return True
    return False


__all__ = [
    "G",
    "FLAP_IMPULSE",
    "MAX_VY",
    "WIDTH",
    "HEIGHT",
    "PIPE_W",
    "PIPE_VX",
    "GAP_HEIGHT",
    "PhysicsConfig",
    "init_state",
    "step_dynamics",
    "passed_pipe",
    "bird_rect",
    "pipe_rects",
    "collides",
    "update_moving_gap",
]
