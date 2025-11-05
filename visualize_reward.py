
"""
Simple visualization of per-step reward components for each episode.
Works with the generic EnhancedRewardLogger JSON.
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt

PREFERRED_KEYS = [
    "r_progress","r_align","r_safety","r_step","r_speed_near_goal",
    "r_collision","r_oob","r_success"
]

def load_latest_log(log_dir="reward_logs"):
    log_path = Path(log_dir)
    json_files = sorted(log_path.glob("rewards_*.json"))
    if not json_files:
        print("No rewards_*.json files found.")
        return None
    return json_files[-1]

def plot_episode(ep, out_dir):
    steps = ep["steps"]
    if not steps:
        return
    t = [s["step"] for s in steps]
    total = [s["total_reward"] for s in steps]

    # Determine which components exist
    keys = set()
    for s in steps:
        keys.update(s.get("reward_components", {}).keys())
    # order by preferred keys first, then the rest alphabetically
    ordered = [k for k in PREFERRED_KEYS if k in keys] + sorted([k for k in keys if k not in PREFERRED_KEYS])

    # Plot total reward
    plt.figure()
    plt.plot(t, total, label="total_reward")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.title("Total reward per step")
    plt.legend()
    out = out_dir / "total_reward.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    # Plot components
    for k in ordered:
        vals = []
        for s in steps:
            vals.append(s.get("reward_components", {}).get(k, 0.0))
        plt.figure()
        plt.plot(t, vals, label=k)
        plt.xlabel("step")
        plt.ylabel("value")
        plt.title(k)
        plt.legend()
        out = out_dir / f"{k}.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()

def main():
    latest = load_latest_log()
    if latest is None:
        return
    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)
    out_root = Path(str(latest).replace("rewards_", "figs_")).with_suffix("")
    out_root.mkdir(parents=True, exist_ok=True)

    episodes = data.get("episodes", [])
    for idx, ep in enumerate(episodes, start=1):
        ep_dir = out_root / f"episode_{idx:02d}_{ep.get('result','NA')}"
        ep_dir.mkdir(exist_ok=True, parents=True)
        plot_episode(ep, ep_dir)
        print(f"Saved plots to {ep_dir}")

if __name__ == "__main__":
    main()
