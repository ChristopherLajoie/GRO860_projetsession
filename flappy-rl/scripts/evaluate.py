"""Evaluate a trained Flappy agent and optionally record video."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import DQN, PPO

from flappy.env import FlappyEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Flappy RL agents")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record", help="Path to MP4/ GIF for saving rollouts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wind", action="store_true")
    parser.add_argument("--moving-pipes", action="store_true")
    parser.add_argument("--three-flaps", action="store_true")
    parser.add_argument("--use-rays", action="store_true")
    parser.add_argument("--n-rays", type=int, default=7)
    parser.add_argument("--energy", action="store_true")
    parser.add_argument("--gap-min", type=float)
    parser.add_argument("--gap-max", type=float)
    parser.add_argument("--max-pipe-speed", type=float)
    return parser.parse_args()


def load_model(algo: str, path: str):
    if algo == "dqn":
        return DQN.load(path)
    return PPO.load(path)


def main() -> None:
    args = parse_args()
    gap_range = None
    if args.gap_min is not None and args.gap_max is not None:
        gap_range = (float(args.gap_min), float(args.gap_max))
    render_mode = "human" if (args.render or args.record) else None
    env = FlappyEnv(
        use_rays=args.use_rays,
        n_rays=args.n_rays,
        three_flaps=args.three_flaps,
        wind=args.wind,
        moving_pipes=args.moving_pipes,
        energy=args.energy,
        gap_height_range=gap_range,
        pipe_speed_cap=args.max_pipe_speed,
        render_mode=render_mode,
        seed=args.seed,
    )
    model = load_model(args.algo, args.model_path)

    lengths: List[int] = []
    scores: List[int] = []
    frames: List[np.ndarray] = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = trunc = False
        steps = 0
        episode_frames: List[np.ndarray] = []
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            steps += 1
            if render_mode:
                frame = env.render()
                if frame is not None:
                    episode_frames.append(frame)
        lengths.append(steps)
        scores.append(info.get("pipes", 0))
        if args.record:
            frames.extend(episode_frames)
    stats = {
        "mean_pipes": float(np.mean(scores)),
        "median_pipes": float(np.median(scores)),
        "max_pipes": float(np.max(scores)) if scores else 0.0,
        "mean_length": float(np.mean(lengths)),
    }
    print("Evaluation stats:", stats)

    if args.record and frames:
        out_path = Path(args.record)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out_path, frames, fps=60)
        print(f"Saved video to {out_path}")


if __name__ == "__main__":
    main()
