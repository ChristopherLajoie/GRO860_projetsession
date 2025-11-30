#!/usr/bin/env python3
"""
Evaluate a trained Flappy agent over multiple seeds and aggregate results.
Wrapper around scripts/evaluate.py.
"""

import argparse
import ast
import subprocess
import sys
import re
import numpy as np
from typing import List, Dict, Any

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Flappy RL agents over multiple seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="List of seeds to run")
    parser.add_argument("--model-path", required=True, help="Path to the model")
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True, help="RL Algorithm")
    
    # Capture all other arguments to pass to evaluate.py
    return parser.parse_known_args()

def run_evaluation(seed: int, known_args: argparse.Namespace, unknown_args: List[str]) -> Dict[str, float]:
    """Runs evaluate.py for a single seed and returns the stats."""
    cmd = [
        sys.executable,
        "flappy-rl/scripts/evaluate.py",
        "--model-path", known_args.model_path,
        "--algo", known_args.algo,
        "--seed", str(seed)
    ] + unknown_args

    print(f"Running evaluation for seed {seed}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Extract the stats dictionary from the output
        # We look for the line starting with "Evaluation stats:"
        for line in output.splitlines():
            if line.strip().startswith("Evaluation stats:"):
                stats_str = line.strip().replace("Evaluation stats:", "", 1).strip()
                try:
                    # Use ast.literal_eval for safe evaluation of the dictionary string
                    stats = ast.literal_eval(stats_str)
                    return stats
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing stats for seed {seed}: {e}")
                    print(f"Line was: {stats_str}")
                    return {}
        
        print(f"Could not find stats in output for seed {seed}")
        return {}

    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation for seed {seed}:")
        print(e.stderr)
        return {}

def main():
    known_args, unknown_args = parse_args()
    
    all_stats: Dict[str, List[float]] = {
        "mean_pipes": [],
        "median_pipes": [],
        "max_pipes": [],
        "mean_length": []
    }
    
    successful_seeds = 0
    
    for seed in known_args.seeds:
        stats = run_evaluation(seed, known_args, unknown_args)
        if stats:
            successful_seeds += 1
            for key, value in stats.items():
                if key in all_stats:
                    all_stats[key].append(value)
    
    print("\n" + "="*40)
    print(f"Aggregated Results ({successful_seeds}/{len(known_args.seeds)} seeds successful)")
    print("="*40)
    
    if successful_seeds > 0:
        for key, values in all_stats.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{key}: {mean_val:.2f} ± {std_val:.2f}")
            else:
                print(f"{key}: N/A")
    else:
        print("No successful evaluations.")

if __name__ == "__main__":
    main()
