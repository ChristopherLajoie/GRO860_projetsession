
"""
Test an agent on the ParkingEnv with detailed, generic reward logging.
Includes command-line / env options for number of tests and rendered runs.
Actively calls env.render() each step to avoid a blank window.
"""

import os
import argparse
from pathlib import Path
from typing import Optional
import gymnasium as gym
from datetime import datetime
import numpy as np

from parking import ParkingEnv
from reward_logger import EnhancedRewardLogger

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

# Optional: handle pygame quit gracefully if available
try:
    import pygame
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False

if HAS_SB3:
    class ObservationNormalizer:
        """Applies saved VecNormalize statistics to raw observations."""

        def __init__(self, vecnorm_path: str):
            self._vecnorm = VecNormalize.load(
                vecnorm_path,
                DummyVecEnv([lambda: ParkingEnv(render_mode=None)])
            )
            self._vecnorm.training = False
            self._vecnorm.norm_reward = False

        def normalize(self, obs: np.ndarray) -> np.ndarray:
            obs_array = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            normalized = self._vecnorm.normalize_obs(obs_array)
            return normalized[0]

        def close(self):
            self._vecnorm.close()


def make_env(render=False):
    return ParkingEnv(render_mode="human" if render else None)


def _maybe_pump_pygame():
    if HAS_PYGAME:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _capture_frame(env, capture_dir: Path, episode_idx: int, step: int, label: Optional[str] = None):
    if capture_dir is None:
        return
    _ensure_dir(capture_dir)
    suffix = f"{step:04d}"
    if label:
        suffix = f"{suffix}_{label}"
    filename = capture_dir / f"ep{episode_idx:02d}_{suffix}.png"
    try:
        env.save_last_frame(str(filename))
    except Exception as exc:
        print(f"⚠️ Could not save frame {filename}: {exc}")


def run_episode(env, policy, logger, render=False, normalizer=None, capture_dir: Optional[Path] = None, episode_idx: int = 0, capture_interval: int = 50):
    obs, info = env.reset()
    obs = np.asarray(obs, dtype=np.float32)
    logger.start_episode(info)

    # show first frame to avoid black window
    should_render = render or capture_dir is not None
    if should_render:
        try:
            env.render()
            _maybe_pump_pygame()
        except Exception:
            pass
    if capture_dir is not None:
        _capture_frame(env, capture_dir, episode_idx, 0, "start")

    total = 0.0
    steps = 0
    result = "TIMEOUT"

    while True:
        if policy is None:
            action = env.action_space.sample()
        else:
            try:
                obs_for_policy = obs
                if normalizer is not None:
                    obs_for_policy = normalizer.normalize(obs_for_policy)
                action, _ = policy.predict(obs_for_policy, deterministic=True)
            except Exception:
                action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        total += float(reward)
        steps += 1
        logger.log_step(steps, reward, info)

        # actively render each step
        if should_render:
            try:
                env.render()
                _maybe_pump_pygame()
            except Exception:
                pass
        if capture_dir is not None and (steps % capture_interval == 0 or terminated or truncated):
            label = None
            if terminated:
                label = result.lower()
            elif truncated:
                label = "truncated"
            _capture_frame(env, capture_dir, episode_idx, steps, label)

        if terminated:
            if info.get("r_success", 0.0) > 0 or info.get("reward_success", 0.0) > 0:
                result = "SUCCESS"
            elif info.get("r_collision", 0.0) < 0 or info.get("reward_collision", 0.0) < 0:
                result = "CRASH"
            elif info.get("r_oob", 0.0) < 0 or info.get("reward_out_of_bounds", 0.0) < 0:
                result = "OUT_OF_BOUNDS"
            break
        if truncated:
            result = "TIMEOUT"
            break

    logger.end_episode(result, total, steps, info)
    if capture_dir is not None:
        _capture_frame(env, capture_dir, episode_idx, steps, result.lower())
    return result, total, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", "-n", type=int, default=int(os.environ.get("N_EPISODES", 20)),
                        help="Number of test episodes to run (default: 20).")
    parser.add_argument("--render", "-r", type=int, default=int(os.environ.get("RENDER_FIRST", 2)),
                        help="Number of episodes to render visually (default: 2).")
    parser.add_argument("--model", "-m", type=str, default=os.environ.get("LOAD_MODEL_PATH", ""),
                        help="Path to a PPO model (optional).")
    parser.add_argument("--vecnorm", "-v", type=str, default=os.environ.get("LOAD_VECNORM_PATH", ""),
                        help="Path to saved VecNormalize statistics (optional).")
    parser.add_argument("--no-capture", action="store_true",
                        help="Disable automatic diagnostic frame capture.")
    parser.add_argument("--capture-interval", type=int, default=50,
                        help="Steps between captured frames (default: 50).")
    args = parser.parse_args()

    # Register env (safe if already registered)
    try:
        gym.register(id="Parking-v0", entry_point="parking:ParkingEnv", max_episode_steps=600)
    except gym.error.Error:
        pass

    logger = EnhancedRewardLogger(log_dir="reward_logs")

    policy = None
    normalizer = None
    if HAS_SB3 and args.model:
        try:
            policy = PPO.load(args.model)
            print(f"✅ Loaded PPO model: {args.model}")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
        else:
            vecnorm_path = args.vecnorm
            if not vecnorm_path:
                base, _ = os.path.splitext(args.model)
                candidates = [
                    f"{args.model}_vecnormalize",
                    f"{args.model}_vecnormalize.npy",
                    f"{args.model}_vecnormalize.pkl",
                    f"{base}_vecnormalize",
                    f"{base}_vecnormalize.npy",
                    f"{base}_vecnormalize.pkl",
                ]
                for cand in candidates:
                    if os.path.exists(cand):
                        vecnorm_path = cand
                        break
            if vecnorm_path:
                try:
                    normalizer = ObservationNormalizer(vecnorm_path)
                    print(f"✅ Loaded VecNormalize stats: {vecnorm_path}")
                except Exception as e:
                    print(f"⚠️ Could not load VecNormalize stats: {e}")

    capture_root = None if args.no_capture else Path("reward_logs") / f"frames_{logger.ts}"
    if capture_root is not None:
        _ensure_dir(capture_root)

    for i in range(args.episodes):
        render = i < args.render
        env = make_env(render=render)
        episode_capture = None
        if capture_root is not None:
            episode_capture = capture_root / f"episode_{i+1:02d}"
        res = run_episode(
            env,
            policy,
            logger,
            render=render,
            normalizer=normalizer,
            capture_dir=episode_capture,
            episode_idx=i + 1,
            capture_interval=max(1, args.capture_interval)
        )
        print(f"Episode {i+1}/{args.episodes}: {res}")
        env.close()

    logger.save_json()
    logger.generate_final_summary()
    logger.print_summary()
    if normalizer is not None:
        normalizer.close()


if __name__ == "__main__":
    main()
