"""Train a DQN agent on the Flappy environment with optional curriculum."""

from __future__ import annotations

import argparse
import subprocess
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from flappy.env import FlappyEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN training for Flappy RL")
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
    parser.add_argument("--logdir", default="runs/dqn")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def unwrap(env):
    while hasattr(env, "env"):
        env = env.env
    return env


class CurriculumCallback(BaseCallback):
    def __init__(self, stages: List[Dict[str, Any]], thresholds: List[float], enabled: bool) -> None:
        super().__init__()
        self.stages = stages
        self.thresholds = thresholds
        self.enabled = enabled
        self.stage_idx = 0
        self.buffer = deque(maxlen=50)

    def _on_training_start(self) -> None:
        if self.enabled:
            self._apply_stage()

    def _apply_stage(self) -> None:
        cfg = self.stages[self.stage_idx]
        for env in self.training_env.envs:
            base = unwrap(env)
            if isinstance(base, FlappyEnv):
                base.apply_settings(
                    gap_height_range=cfg["gap_range"],
                    moving_pipes=cfg["moving_pipes"],
                    wind=cfg["wind"],
                )
        print(f"[Curriculum] Stage {self.stage_idx} -> {cfg}")

    def _on_step(self) -> bool:
        if not self.enabled or self.stage_idx >= len(self.stages) - 1:
            return True
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
    if not (args.wind or args.curriculum):
        env.apply_settings(wind=False)
    else:
        env.apply_settings(wind=args.wind)
    env.apply_settings(moving_pipes=args.moving_pipes)
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


def evaluate_model(model: DQN, env: FlappyEnv, episodes: int, render: bool) -> Dict[str, float]:
    lengths: list[int] = []
    scores: list[int] = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
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


def main() -> None:
    args = parse_args()
    gap_range = _gap_range_from_args(args)
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    vec_env = DummyVecEnv([lambda: Monitor(make_env(args, i)) for i in range(1)])

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
            seed=args.seed + 42,
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
        {"gap_range": gap_range or (130, 140), "moving_pipes": False, "wind": False},
        {"gap_range": (115, 130), "moving_pipes": False, "wind": False},
        {"gap_range": (110, 125), "moving_pipes": True, "wind": False},
        {"gap_range": (105, 120), "moving_pipes": True, "wind": True},
    ]
    thresholds = [15, 20, 25]
    curriculum_cb = CurriculumCallback(curriculum_stages, thresholds, args.curriculum)

    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        train_freq=4,
        target_update_interval=5000,
        exploration_final_eps=0.02,
        exploration_fraction=0.4,
        tau=0.005,
        tensorboard_log=str(logdir),
        verbose=1,
        seed=args.seed,
    )

    write_config(
        logdir / "config.yaml",
        {
            "algo": "DQN",
            "seed": args.seed,
            "total_steps": args.total_steps,
            "wind": args.wind,
            "moving_pipes": args.moving_pipes,
            "three_flaps": args.three_flaps,
            "use_rays": args.use_rays,
            "energy": args.energy,
            "curriculum": args.curriculum,
            "gap_range": gap_range,
        },
    )

    progress_cb = ProgressCallback(args.total_steps)
    model.learn(total_timesteps=args.total_steps, callback=[eval_callback, curriculum_cb, progress_cb])

    final_env = FlappyEnv(
        use_rays=args.use_rays,
        n_rays=args.n_rays,
        three_flaps=args.three_flaps,
        wind=args.wind or args.curriculum,
        moving_pipes=args.moving_pipes,
        energy=args.energy,
        gap_height_range=gap_range,
        render_mode="human" if args.render_eval else None,
        seed=args.seed + 99,
    )

    stats = evaluate_model(model, final_env, args.eval_episodes, args.render_eval)
    print("Evaluation:", stats)
    model.save(logdir / "latest_dqn")


if __name__ == "__main__":
    main()
