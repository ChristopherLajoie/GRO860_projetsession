
"""
Enhanced logger (generic) for reward components and performance metrics.

- Works with ANY env that returns `info` dicts.
- Automatically collects all per-step reward components whose keys start with
  "r_" or "reward_".
- Tracks basic kinematics and proximity metrics.

Outputs:
  - reward_logs/rewards_YYYYMMDD_HHMMSS.json   (all episodes & steps)
  - reward_logs/summary_YYYYMMDD_HHMMSS.txt    (human-readable summary)
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

@dataclass
class StepRecord:
    step: int
    total_reward: float
    velocity: float
    angle_error: Optional[float] = None
    min_lidar: Optional[float] = None
    distance_to_target: Optional[float] = None
    dist_to_subgoal: Optional[float] = None
    dist_to_gate: Optional[float] = None
    dist_to_spot: Optional[float] = None
    angle_to_subgoal: Optional[float] = None
    subgoal_forward: Optional[float] = None
    subgoal_lateral: Optional[float] = None
    heading_cos_to_goal: Optional[float] = None
    spot_forward: Optional[float] = None
    spot_lateral: Optional[float] = None
    spot_heading_error: Optional[float] = None
    overlap_ratio: Optional[float] = None
    use_gate: Optional[bool] = None
    heading_error: Optional[float] = None
    lateral_offset: Optional[float] = None
    reward_components: Dict[str, float] = field(default_factory=dict)

@dataclass
class EpisodeRecord:
    result: str
    total_reward: float
    steps: int
    min_distance: Optional[float]
    max_speed: float
    close_calls: int
    direction_changes: int
    time_in_reverse: int
    initial_distance: Optional[float]
    final_distance: Optional[float]
    steps_data: List[StepRecord] = field(default_factory=list)

class EnhancedRewardLogger:
    def __init__(self, log_dir: str = "reward_logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.ts = _now_tag()
        self.json_log = os.path.join(self.log_dir, f"rewards_{self.ts}.json")
        self.summary_log = os.path.join(self.log_dir, f"summary_{self.ts}.txt")

        # internal trackers
        self.episodes: List[EpisodeRecord] = []
        self.current_steps: List[StepRecord] = []
        self.current_target: Optional[np.ndarray] = None  # [tx, ty]
        self.prev_velocity: Optional[float] = None

        # episode metrics
        self._min_distance: Optional[float] = None
        self._max_speed: float = 0.0
        self._close_calls: int = 0
        self._direction_changes: int = 0
        self._time_in_reverse: int = 0
        self._initial_distance: Optional[float] = None
        self._final_distance: Optional[float] = None

    # ---- lifecycle ----
    def start_episode(self, initial_info: Dict[str, Any] | None):
        """Call once at the beginning of an episode (just after env.reset)."""
        self.current_steps = []
        self.current_target = None
        self.prev_velocity = None
        self._min_distance = None
        self._max_speed = 0.0
        self._close_calls = 0
        self._direction_changes = 0
        self._time_in_reverse = 0
        self._initial_distance = None
        self._final_distance = None

        # store target from reset info if provided
        if initial_info is not None:
            tx = initial_info.get("target_x", None)
            ty = initial_info.get("target_y", None)
            if tx is not None and ty is not None:
                self.current_target = np.array([float(tx), float(ty)], dtype=float)

            # also compute initial distance if state present
            st = initial_info.get("state", None)
            if st is not None and self.current_target is not None:
                car_xy = np.array(st[:2], dtype=float)
                self._initial_distance = float(np.linalg.norm(self.current_target - car_xy))

    def log_step(self, step_num: int, reward: float, info: Dict[str, Any] | None):
        """Record one step. `info` should be the env info dict from step()."""
        if info is None:
            info = {}

        # try to keep target if the env provides it every step
        if self.current_target is None:
            tx = info.get("target_x", None)
            ty = info.get("target_y", None)
            if tx is not None and ty is not None:
                self.current_target = np.array([float(tx), float(ty)], dtype=float)

        # kinematics
        angle_error = float(info.get("angle_error", np.nan)) if "angle_error" in info else None
        velocity = None
        state = info.get("state", None)
        if state is not None and len(state) >= 4:
            velocity = float(abs(state[3]))
        else:
            velocity = float(info.get("speed", 0.0))

        # lidar
        min_lidar = None
        if "lidar_distances" in info:
            try:
                ld = np.asarray(info["lidar_distances"], dtype=float)
                min_lidar = float(np.min(ld))
                if min_lidar < 1.0:
                    self._close_calls += 1
            except Exception:
                pass

        # distance to target (if we know target and position)
        distance = None
        if self.current_target is not None and state is not None and len(state) >= 2:
            car_xy = np.array(state[:2], dtype=float)
            distance = float(np.linalg.norm(self.current_target - car_xy))
            self._min_distance = distance if (self._min_distance is None) else min(self._min_distance, distance)

        dist_to_subgoal = info.get("dist_to_subgoal")
        dist_to_gate = info.get("dist_to_gate")
        dist_to_spot = info.get("dist_to_spot")
        angle_to_subgoal = info.get("angle_to_subgoal")
        subgoal_forward = info.get("subgoal_forward")
        subgoal_lateral = info.get("subgoal_lateral")
        heading_cos_to_goal = info.get("heading_cos_to_goal")
        spot_forward = info.get("spot_forward")
        spot_lateral = info.get("spot_lateral")
        spot_heading_error = info.get("spot_heading_error")
        overlap_ratio = info.get("overlap_ratio")
        use_gate = info.get("use_gate")
        heading_error_val = info.get("heading_error")
        lateral_offset = info.get("lateral_offset")

        # direction change / reverse time
        if velocity is not None:
            self._max_speed = max(self._max_speed, float(velocity))
            if state is not None and state[3] < -0.1:
                self._time_in_reverse += 1
            if self.prev_velocity is not None and (self.prev_velocity * state[3] < -0.1):
                self._direction_changes += 1
            self.prev_velocity = state[3] if state is not None else self.prev_velocity

        # gather reward components (generic)
        reward_components: Dict[str, float] = {}
        for k, v in info.items():
            if k.startswith("r_") or k.startswith("reward_"):
                try:
                    reward_components[k] = float(v)
                except Exception:
                    continue

        self.current_steps.append(
            StepRecord(
                step=int(step_num),
                total_reward=float(reward),
                velocity=float(velocity) if velocity is not None else 0.0,
                angle_error=angle_error,
                min_lidar=min_lidar,
                distance_to_target=distance,
                dist_to_subgoal=float(dist_to_subgoal) if dist_to_subgoal is not None else None,
                dist_to_gate=float(dist_to_gate) if dist_to_gate is not None else None,
                dist_to_spot=float(dist_to_spot) if dist_to_spot is not None else None,
                angle_to_subgoal=float(angle_to_subgoal) if angle_to_subgoal is not None else None,
                subgoal_forward=float(subgoal_forward) if subgoal_forward is not None else None,
                subgoal_lateral=float(subgoal_lateral) if subgoal_lateral is not None else None,
                heading_cos_to_goal=float(heading_cos_to_goal) if heading_cos_to_goal is not None else None,
                spot_forward=float(spot_forward) if spot_forward is not None else None,
                spot_lateral=float(spot_lateral) if spot_lateral is not None else None,
                spot_heading_error=float(spot_heading_error) if spot_heading_error is not None else None,
                overlap_ratio=float(overlap_ratio) if overlap_ratio is not None else None,
                use_gate=bool(use_gate) if use_gate is not None else None,
                heading_error=float(heading_error_val) if heading_error_val is not None else None,
                lateral_offset=float(lateral_offset) if lateral_offset is not None else None,
                reward_components=reward_components
            )
        )

    def end_episode(self, result: str, total_reward: float, steps: int, final_info: Dict[str, Any] | None = None):
        """Call once after the episode terminates or truncates."""
        if final_info is not None and self.current_target is not None:
            st = final_info.get("state", None)
            if st is not None and len(st) >= 2:
                car_xy = np.array(st[:2], dtype=float)
                self._final_distance = float(np.linalg.norm(self.current_target - car_xy))

        ep = EpisodeRecord(
            result=result,
            total_reward=float(total_reward),
            steps=int(steps),
            min_distance=self._min_distance,
            max_speed=self._max_speed,
            close_calls=self._close_calls,
            direction_changes=self._direction_changes,
            time_in_reverse=self._time_in_reverse,
            initial_distance=self._initial_distance,
            final_distance=self._final_distance,
            steps_data=self.current_steps.copy()
        )
        self.episodes.append(ep)
        self.current_steps = []

    # ---- persistence ----
    def save_json(self):
        payload = {
            "summary": {
                "episodes": len(self.episodes),
                "created_at": self.ts,
            },
            "episodes": [
                {
                    "result": ep.result,
                    "total_reward": ep.total_reward,
                    "steps": ep.steps,
                    "min_distance": ep.min_distance,
                    "max_speed": ep.max_speed,
                    "close_calls": ep.close_calls,
                    "direction_changes": ep.direction_changes,
                    "time_in_reverse": ep.time_in_reverse,
                    "initial_distance": ep.initial_distance,
                    "final_distance": ep.final_distance,
                    "steps": [
                        {
                            "step": st.step,
                            "total_reward": st.total_reward,
                            "velocity": st.velocity,
                            "angle_error": st.angle_error,
                            "min_lidar": st.min_lidar,
                            "distance_to_target": st.distance_to_target,
                            "dist_to_subgoal": st.dist_to_subgoal,
                            "dist_to_gate": st.dist_to_gate,
                            "dist_to_spot": st.dist_to_spot,
                            "angle_to_subgoal": st.angle_to_subgoal,
                            "subgoal_forward": st.subgoal_forward,
                            "subgoal_lateral": st.subgoal_lateral,
                            "heading_cos_to_goal": st.heading_cos_to_goal,
                            "spot_forward": st.spot_forward,
                            "spot_lateral": st.spot_lateral,
                            "spot_heading_error": st.spot_heading_error,
                            "overlap_ratio": st.overlap_ratio,
                            "use_gate": st.use_gate,
                            "heading_error": st.heading_error,
                            "lateral_offset": st.lateral_offset,
                            "reward_components": st.reward_components
                        }
                        for st in ep.steps_data
                    ]
                }
                for ep in self.episodes
            ]
        }
        with open(self.json_log, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def generate_final_summary(self):
        """Write a concise, human-readable text summary."""
        lines = []
        lines.append("="*80)
        lines.append(f"Reward Analysis Summary - {self.ts}")
        lines.append("="*80)
        lines.append(f"Episodes: {len(self.episodes)}")

        if not self.episodes:
            with open(self.summary_log, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return

        successes = sum(1 for e in self.episodes if e.result == "SUCCESS")
        timeouts  = sum(1 for e in self.episodes if e.result == "TIMEOUT")
        crashes   = sum(1 for e in self.episodes if e.result == "CRASH")
        oobs      = sum(1 for e in self.episodes if e.result == "OUT_OF_BOUNDS")

        lines.append(f"Success rate: {successes}/{len(self.episodes)} ({(successes/len(self.episodes))*100:.1f}%)")
        lines.append(f"Timeouts: {timeouts}, Crashes: {crashes}, OOB: {oobs}")

        avg_steps = np.mean([e.steps for e in self.episodes])
        avg_reward = np.mean([e.total_reward for e in self.episodes])
        lines.append(f"Avg steps: {avg_steps:.1f}")
        lines.append(f"Avg total reward: {avg_reward:.2f}")

        # distance and speed
        dists = [e.min_distance for e in self.episodes if e.min_distance is not None]
        if dists:
            lines.append(f"Min distance (avg across eps): {np.mean(dists):.2f} m")
        speeds = [e.max_speed for e in self.episodes]
        lines.append(f"Max speed (avg across eps): {np.mean(speeds):.2f} m/s")

        with open(self.summary_log, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def print_summary(self):
        """Convenience for console."""
        if not self.episodes:
            print("No episodes recorded.")
            return

        print(f"\n{'='*80}")
        print(f"Quick Summary: {len(self.episodes)} episodes")
        print(f"{'='*80}")
        successes = sum(1 for e in self.episodes if e.result == "SUCCESS")
        print(f"✅ Success Rate: {successes}/{len(self.episodes)} ({successes/len(self.episodes)*100:.1f}%)")
        avg_reward = np.mean([e.total_reward for e in self.episodes])
        print(f"📊 Average Reward: {avg_reward:.1f}")
        print(f"\n📁 Full analysis saved to: {self.summary_log}")
        print(f"{'='*80}\n")
