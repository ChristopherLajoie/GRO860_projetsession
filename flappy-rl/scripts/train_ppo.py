"""Train a PPO agent on the Flappy environment."""

from __future__ import annotations

import argparse
import subprocess
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from flappy.env import FlappyEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO training for Flappy RL")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--render-eval", action="store_true")
    parser.add_argument("--wind", action="store_true")
    parser.add_argument("--moving-pipes", action="store_true")
    parser.add_argument("--three-flaps", action="store_true")
    parser.add_argument("--use-rays", action="store_true")
    parser.add_argument("--n-rays", type=int, default=7)
    parser.add_argument("--energy", action="store_true")
    parser.add_argument("--gap-min", type=float)
    parser.add_argument("--gap-max", type=float)
    parser.add_argument("--logdir", default="runs/ppo")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _gap_range_from_args(args: argparse.Namespace) -> Optional[Tuple[float, float]]:
    if args.gap_min is not None and args.gap_max is not None:
        return (float(args.gap_min), float(args.gap_max))
    return None


def make_env(args: argparse.Namespace, seed_offset: int = 0) -> FlappyEnv:
    gap_range = _gap_range_from_args(args)
    env = FlappyEnv(
        use_rays=args.use_rays,
        n_rays=args.n_rays,
        three_flaps=args.three_flaps,
        wind=args.wind or args.curriculum,
        moving_pipes=args.moving_pipes,
        energy=args.energy,
        gap_height_range=gap_range,
        render_mode=None,
        seed=args.seed + seed_offset,
    )
    return env


def write_config(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass
    data = {**data, "git": git_hash}
    lines = ["# Auto-generated run config"]
    for key, value in data.items():
        lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines))


def evaluate_model(model: PPO, env: FlappyEnv, episodes: int, render: bool) -> Dict[str, float]:
    lengths: list[int] = []
    scores: list[int] = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = trunc = False
        steps = 0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, info = env.step(action)
            steps += 1
            if render:
                env.render()
        lengths.append(steps)
        scores.append(info.get("pipes", 0))
    return {
        "mean_pipes": float(np.mean(scores)),
        "median_pipes": float(np.median(scores)),
        "mean_length": float(np.mean(lengths)),
    }


def unwrap(env):
    while hasattr(env, "env"):
        env = env.env
    return env


class CurriculumCallback(BaseCallback):
    def __init__(self, stages: List[Dict[str, Any]], thresholds: List[float], enabled: bool, log_every: int = 2000) -> None:
        super().__init__()
        self.stages = stages
        self.thresholds = thresholds
        self.enabled = enabled
        self.stage_idx = 0
        self.buffer = deque(maxlen=50)
        self.log_every = log_every
        self._last_log_step = 0

    def _on_training_start(self) -> None:
        if self.enabled:
            self._apply_stage(initial=True)

    def _apply_stage(self, initial: bool = False) -> None:
        cfg = self.stages[self.stage_idx]
        for env in self.training_env.envs:
            base = unwrap(env)
            if isinstance(base, FlappyEnv):
                base.apply_settings(
                    gap_height_range=cfg["gap_range"],
                    moving_pipes=cfg["moving_pipes"],
                    wind=cfg["wind"],
                    pipe_speed=cfg.get("pipe_speed"),
                    pipe_speed_growth=cfg.get("pipe_speed_growth"),
                    wind_mu=cfg.get("wind_mu"),
                    moving_amp=cfg.get("moving_amp"),
                    moving_omega=cfg.get("moving_omega"),
                )
        self._record_curriculum(cfg, force=True)
        if not initial:
            print(f"[Curriculum] Stage {self.stage_idx} -> {cfg}")

    def _on_step(self) -> bool:
        if not self.enabled or self.stage_idx >= len(self.stages) - 1:
            return True
        cfg = self.stages[self.stage_idx]
        if self.model.num_timesteps - self._last_log_step >= self.log_every:
            self._record_curriculum(cfg)
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if done and "pipes" in info:
                self.buffer.append(info["pipes"])
        if len(self.buffer) == self.buffer.maxlen:
            avg = sum(self.buffer) / len(self.buffer)
            threshold = self.thresholds[self.stage_idx]
            if avg >= threshold:
                self.stage_idx += 1
                self._apply_stage()
                self.buffer.clear()
        return True

    def _record_curriculum(self, cfg: Dict[str, Any], force: bool = False) -> None:
        self._last_log_step = self.model.num_timesteps
        self.model.logger.record("curriculum/stage", float(self.stage_idx), exclude="stdout")
        self.model.logger.record("curriculum/gap_min", cfg["gap_range"][0], exclude="stdout")
        self.model.logger.record("curriculum/gap_max", cfg["gap_range"][1], exclude="stdout")
        self.model.logger.record("curriculum/moving_pipes", float(cfg["moving_pipes"]), exclude="stdout")
        self.model.logger.record("curriculum/wind", float(cfg["wind"]), exclude="stdout")
        if cfg.get("pipe_speed") is not None:
            self.model.logger.record("curriculum/pipe_speed", float(cfg["pipe_speed"]), exclude="stdout")
        if cfg.get("pipe_speed_growth") is not None:
            self.model.logger.record(
                "curriculum/pipe_speed_growth",
                float(cfg["pipe_speed_growth"]),
                exclude="stdout",
            )
        self.model.logger.dump(self.model.num_timesteps)


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int) -> None:
        super().__init__()
        self.total_timesteps = total_timesteps
        self._pbar: Optional[tqdm] = None
        self._last_update = 0

    def _on_training_start(self) -> None:
        if self._pbar is None:
            self._pbar = tqdm(total=self.total_timesteps, desc="Training", dynamic_ncols=True)

    def _on_step(self) -> bool:
        if self._pbar is not None:
            current = min(self.model.num_timesteps, self.total_timesteps)
            self._pbar.update(max(0, current - self._last_update))
            self._last_update = current
        return True

    def _on_training_end(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


def main() -> None:
    args = parse_args()
    gap_range = _gap_range_from_args(args)
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    vec_env = DummyVecEnv([lambda: Monitor(make_env(args, 0))])

    eval_env = Monitor(
        FlappyEnv(
            use_rays=args.use_rays,
            n_rays=args.n_rays,
            three_flaps=args.three_flaps,
            wind=args.wind or args.curriculum,
            moving_pipes=args.moving_pipes,
            energy=args.energy,
            gap_height_range=gap_range,
            render_mode="human" if args.render_eval else None,
            seed=args.seed + 101,
        )
    )
    eval_env.env.apply_settings(wind=args.wind)

    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=args.eval_episodes,
        eval_freq=20_000,
        best_model_save_path=str(logdir),
        log_path=str(logdir),
        deterministic=True,
        render=args.render_eval,
    )

    curriculum_stages = [
        {
            "gap_range": gap_range or (150, 165),
            "moving_pipes": False,
            "wind": False,
            "pipe_speed": -3.2,
            "pipe_speed_growth": 0.0,
        },
        {
            "gap_range": (135, 150),
            "moving_pipes": False,
            "wind": False,
            "pipe_speed": -3.2,
            "pipe_speed_growth": 0.0,
        },
        {
            "gap_range": (130, 145),
            "moving_pipes": True,
            "wind": False,
            "pipe_speed": -3.5,
            "pipe_speed_growth": 0.01,
            "moving_amp": 18.0,
            "moving_omega": 0.03,
        },
        {
            "gap_range": (120, 135),
            "moving_pipes": True,
            "wind": True,
            "pipe_speed": -4.0,
            "pipe_speed_growth": 0.015,
            "wind_mu": 0.18,
        },
        {
            "gap_range": (110, 125),
            "moving_pipes": True,
            "wind": True,
            "pipe_speed": -5.5,
            "pipe_speed_growth": 0.02,
            "wind_mu": 0.22,
            "moving_amp": 24.0,
        },
        {
            "gap_range": (100, 115),
            "moving_pipes": True,
            "wind": True,
            "pipe_speed": -7.0,
            "pipe_speed_growth": 0.03,
            "wind_mu": 0.25,
            "moving_amp": 30.0,
            "moving_omega": 0.05,
        },
    ]
    thresholds = [2, 4, 6, 9, 12, 15]
    curriculum_cb = CurriculumCallback(curriculum_stages, thresholds, args.curriculum)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1.5e-4,
        n_steps=4096,
        batch_size=512,
        clip_range=0.25,
        gae_lambda=0.95,
        ent_coef=0.005,
        vf_coef=0.6,
        max_grad_norm=0.5,
        tensorboard_log=str(logdir),
        seed=args.seed,
        verbose=1,
    )

    write_config(
        logdir / "config.yaml",
        {
            "algo": "PPO",
            "seed": args.seed,
            "total_steps": args.total_steps,
            "wind": args.wind,
            "moving_pipes": args.moving_pipes,
            "three_flaps": args.three_flaps,
            "use_rays": args.use_rays,
            "energy": args.energy,
            "gap_range": gap_range,
        },
    )

    progress_cb = ProgressCallback(args.total_steps)
    callbacks = [eval_callback, progress_cb]
    if args.curriculum:
        callbacks.append(curriculum_cb)
    model.learn(total_timesteps=args.total_steps, callback=callbacks)

    final_env = FlappyEnv(
        use_rays=args.use_rays,
        n_rays=args.n_rays,
        three_flaps=args.three_flaps,
        wind=args.wind,
        moving_pipes=args.moving_pipes,
        energy=args.energy,
        gap_height_range=gap_range,
        render_mode="human" if args.render_eval else None,
        seed=args.seed + 202,
    )

    stats = evaluate_model(model, final_env, args.eval_episodes, args.render_eval)
    print("Evaluation:", stats)
    model.save(logdir / "latest_ppo")


if __name__ == "__main__":
    main()
