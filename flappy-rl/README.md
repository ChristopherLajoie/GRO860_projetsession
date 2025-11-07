# Flappy RL

A Flappy Bird–style Gymnasium environment with optional complexity toggles (wind, moving pipes, ray sensors, energy budget) plus Stable-Baselines3 training scripts.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install tensorboard pytest
```

## Quick commands

```bash
# train (baseline DQN)
python scripts/train_dqn.py --seed 0 --total-steps 300000

# train with complexity
python scripts/train_dqn.py --seed 1 --total-steps 400000 --three-flaps --wind --moving-pipes

# evaluate
python scripts/evaluate.py --algo dqn --model-path runs/dqn/best_model.zip --episodes 50 --render

# tests
pytest -q

# tensorboard
tensorboard --logdir runs
```

## Usage highlights

- Toggle features with flags: `--wind`, `--moving-pipes`, `--three-flaps`, `--use-rays`, `--energy`.
- Clamp gaps with `--gap-min/--gap-max` to study generalisation (training scripts and evaluator).
- Enable/disable the built-in curriculum via `--curriculum` / `--no-curriculum` on `train_dqn.py`.
- TensorBoard logs, SB3 checkpoints, and `runs/config.yaml` capture seeds, flags, and git revision for reproducibility.

## Experiments

Run each experiment for ≥300k steps (adjust seeds as needed). Track **mean/median pipes passed**, **mean survival steps**, TensorBoard learning curves, and generalisation performance (train on `--gap-min 100 --gap-max 130`, then evaluate on `[85,95]` and `[135,150]`).

1. **Baseline DQN (static pipes, no wind)**
   ```bash
   python scripts/train_dqn.py --seed 0 --total-steps 300000 --no-curriculum --gap-min 130 --gap-max 140
   ```
2. **+Three flaps**
   ```bash
   python scripts/train_dqn.py --seed 1 --total-steps 350000 --three-flaps --no-curriculum --gap-min 130 --gap-max 140
   ```
3. **+Wind**
   ```bash
   python scripts/train_dqn.py --seed 2 --total-steps 350000 --wind --no-curriculum --gap-min 130 --gap-max 140
   ```
4. **+Moving pipes**
   ```bash
   python scripts/train_dqn.py --seed 3 --total-steps 400000 --wind --moving-pipes --gap-min 115 --gap-max 130
   ```
5. **Rays vs gap-info**
   ```bash
   # gap-based obs (reference)
   python scripts/train_dqn.py --seed 4 --total-steps 300000 --no-curriculum --gap-min 120 --gap-max 135
   # ray sensors
   python scripts/train_dqn.py --seed 4 --total-steps 300000 --use-rays --n-rays 7 --no-curriculum --gap-min 120 --gap-max 135
   ```
6. **PPO variant**
   ```bash
   python scripts/train_ppo.py --seed 5 --total-steps 400000 --wind --moving-pipes --three-flaps
   ```

### Evaluation & video

```bash
python scripts/evaluate.py --algo dqn --model-path runs/dqn/best_model.zip --episodes 50 --render --record runs/eval.mp4
```

Report success rate (pipes ≥ threshold), mean/median pipes, mean episode length, and supply qualitative rollouts when recording via `--record`.
