
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from flappy.physics import HEIGHT, MAX_VY, PIPE_VX, GAP_HEIGHT, WIDTH, BIRD_X

def get_obs(bird_y, bird_vy, dx, gap_center_y, gap_height, pipe_vx, wind):
    # Normalize features as in env.py
    norm_bird_y = np.clip(bird_y / HEIGHT * 2.0 - 1.0, -1.0, 1.0)
    norm_bird_vy = np.clip(bird_vy / MAX_VY, -1.0, 1.0)
    norm_dx = np.clip((dx - BIRD_X) / WIDTH, -1.0, 1.0)
    
    norm_gap_center = (gap_center_y - HEIGHT / 2) / (HEIGHT / 2)
    norm_gap_height = (gap_height - GAP_HEIGHT) / GAP_HEIGHT
    
    norm_pipe_vx = np.clip(pipe_vx / abs(PIPE_VX), -1.0, 1.0)
    norm_wind = np.clip(wind, -1.0, 1.0)
    
    # [bird_y, bird_vy, dx, gap_center, gap_height, pipe_vx, wind]
    # Assuming no rays, no energy, wind=True
    features = [norm_bird_y, norm_bird_vy, norm_dx, norm_gap_center, norm_gap_height, norm_pipe_vx, norm_wind]
    return np.array(features, dtype=np.float32)

def plot_policy(model_path, output_file, wind_value=0.0):
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Grid resolution
    n_y = 50
    n_vy = 50
    
    ys = np.linspace(0, HEIGHT, n_y)
    vys = np.linspace(-MAX_VY, MAX_VY, n_vy)
    
    # Fixed parameters
    dx = 100.0 # Approaching pipe
    gap_center_y = HEIGHT / 2
    gap_height = GAP_HEIGHT
    pipe_vx = -8.0 # Standard speed
    wind = wind_value
    
    probs = np.zeros((n_vy, n_y))
    
    print("Generating heatmap...")
    for i, vy in enumerate(vys):
        for j, y in enumerate(ys):
            # Create single frame observation
            obs_frame = get_obs(y, vy, dx, gap_center_y, gap_height, pipe_vx, wind)
            
            # Stack 4 times (simulate steady state)
            obs_stacked = np.concatenate([obs_frame] * 4)
            
            # Predict
            # We need to convert to tensor for the policy
            obs_tensor = torch.as_tensor(obs_stacked).unsqueeze(0).to(model.device)
            
            with torch.no_grad():
                # Get distribution
                dist = model.policy.get_distribution(obs_tensor)
                # Probability of Action 1 (Flap)
                # Categorical distribution: probs returns [prob_0, prob_1, ...]
                prob_flap = dist.distribution.probs[0][1].item()
                
            probs[i, j] = prob_flap

    # Plot
    plt.figure(figsize=(10, 8))
    # Origin is usually bottom-left for plots, but matrix is (row, col).
    # Row i corresponds to vy[i]. Col j corresponds to ys[j].
    # We want Y-axis to be Bird Y (Height) and X-axis to be Velocity?
    # Or usually: X=Position, Y=Velocity (Phase Space).
    # Let's do: X = Vertical Velocity, Y = Bird Height (Y position).
    # Note: In game, Y=0 is TOP. Y=HEIGHT is BOTTOM.
    # Let's invert Y axis to match game visual (0 at top).
    
    plt.imshow(probs, extent=[0, HEIGHT, -MAX_VY, MAX_VY], origin='lower', aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
    
    # Wait, extent is [xmin, xmax, ymin, ymax].
    # If we want X=Height, Y=Velocity:
    # probs[i, j] -> i is vy index (Y-axis), j is y index (X-axis).
    
    plt.colorbar(label='Flap Probability')
    plt.xlabel('Bird Y Position (0=Top, 600=Bottom)')
    plt.ylabel('Vertical Velocity (-12=Up, +12=Down)')
    plt.title(f'PPO Policy: Flap Probability Heatmap\n(Pipe Distance=100, Wind={wind})')
    
    # Add target line (Gap Center)
    plt.axvline(x=gap_center_y, color='green', linestyle='--', label='Gap Center')
    plt.legend()
    
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="policy_heatmap.png")
    parser.add_argument("--wind-value", type=float, default=0.0, help="Wind value to visualize")
    args = parser.parse_args()
    
    plot_policy(args.model_path, args.output, args.wind_value)
