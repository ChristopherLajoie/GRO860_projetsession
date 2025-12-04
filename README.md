# Flappy RL

A Flappy Bird–style Gymnasium environment with optional complexity plus Stable-Baselines3 training scripts.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install tensorboard pytest
```

## Quick commands

```bash
# train
python scripts/train_rl.py --algo ppo --total-steps 300000

# train with curriculum
python scripts/train_rl.py --algo ppo --total-steps 1000000 --curriculum

# evaluate
python scripts/evaluate.py --algo ppo --model-path runs/PPO_23/PPO_23_latest.zip --episodes 50

# evaluate and render
python scripts/evaluate.py --algo ppo --model-path runs/PPO_23/PPO_23_latest.zip --episodes 50 --render

# benchmark (PPO vs PID)
python scripts/evaluate_multiple_seeds.py --algo ppo --model-path runs/PPO_23/PPO_23_latest.zip
python scripts/evaluate_multiple_seeds.py --algo pid --model-path none

# tests
pytest
```

## Usage highlights

- **Unified Trainer**: Use `scripts/train_rl.py` for training.
- **Features**: Toggle features with flags: `--wind`, `--moving-pipes`, `--three-flaps`.
- **Curriculum**: Enable/disable the built-in curriculum via `--curriculum` / `--no-curriculum`.

## Evaluation & Analysis

**Visualize Policy (Heatmap):**
```bash
python policy/plot_policy.py --model-path runs/PPO_23/PPO_23_latest.zip  --output policy.png --wind-value 0.0
```

**PID Tuning:**
Scripts for tuning the PID controller are located in `scripts/pid/`.
```bash
python scripts/pid/tune_pid.py
```
