"""Utility helpers for Flappy RL."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def make_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Return a numpy Generator seeded from ``seed`` when provided."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def ou_step(
    wind: float,
    rng: np.random.Generator,
    mu: float = 0.0,
    sigma: float = 0.3,
    theta: float = 0.05,
    clamp_value: float = 0.6,
) -> float:
    """Perform one Ornstein-Uhlenbeck step for the wind process."""
    noise = rng.normal(0.0, sigma)
    delta = theta * (mu - wind) + noise
    return clamp(wind + delta, -clamp_value, clamp_value)


def line_aabb_intersection(
    p0: Sequence[float],
    p1: Sequence[float],
    rect: Sequence[float],
) -> tuple[bool, float]:
    """Return (hit, distance) for segment-AABB intersection using Liang-Barsky."""
    x, y, w, h = rect
    min_corner = np.array([x, y], dtype=float)
    max_corner = np.array([x + w, y + h], dtype=float)
    p0_arr = np.array(p0, dtype=float)
    direction = np.array(p1, dtype=float) - p0_arr

    t_min, t_max = 0.0, 1.0
    for i in range(2):
        if np.isclose(direction[i], 0.0):
            if p0_arr[i] < min_corner[i] or p0_arr[i] > max_corner[i]:
                return False, float("inf")
        else:
            inv_d = 1.0 / direction[i]
            t1 = (min_corner[i] - p0_arr[i]) * inv_d
            t2 = (max_corner[i] - p0_arr[i]) * inv_d
            t_enter = min(t1, t2)
            t_exit = max(t1, t2)
            t_min = max(t_min, t_enter)
            t_max = min(t_max, t_exit)
            if t_min > t_max:
                return False, float("inf")

    hit_distance = np.linalg.norm(direction) * t_min
    return True, hit_distance


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` to the inclusive range ``[lo, hi]``."""
    return max(lo, min(hi, value))


__all__ = ["make_rng", "ou_step", "line_aabb_intersection", "clamp"]
