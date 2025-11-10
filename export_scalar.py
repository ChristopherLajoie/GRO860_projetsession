#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional, Set

from tensorboard.backend.event_processing import event_accumulator


def iter_event_files(logdir: Path, allowed_runs: Optional[Set[str]]) -> Iterable[Path]:
    for path in logdir.rglob("events.out.tfevents.*"):
        run = path.parent.name
        if allowed_runs is not None and run not in allowed_runs:
            continue
        yield path


def export_scalars(logdir: Path, out_csv: Path, allowed_runs: Optional[Set[str]] = None) -> None:
    rows = []
    for event_path in iter_event_files(logdir, allowed_runs):
        run = event_path.parent.name
        ea = event_accumulator.EventAccumulator(str(event_path))
        try:
            ea.Reload()
        except Exception as exc:
            print(f"Skipping {event_path}: {exc}")
            continue
        for tag in ea.Tags().get("scalars", []):
            for event in ea.Scalars(tag):
                rows.append(
                    {
                        "run": run,
                        "tag": tag,
                        "step": event.step,
                        "wall_time": event.wall_time,
                        "value": event.value,
                    }
                )

    rows.sort(key=lambda r: (r["run"], r["tag"], r["step"]))
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "tag", "step", "wall_time", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} scalar points to {out_csv}")


if __name__ == "__main__":
    logdir = Path("runs/ppo")  # change if needed
    out_csv = Path("tensorboard_scalars.csv")
    allowed = {"PPO_10"}  # e.g. {"DQN_3", "DQN_5"} to export only those runs
    export_scalars(logdir, out_csv, allowed)
