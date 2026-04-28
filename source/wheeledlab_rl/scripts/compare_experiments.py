"""
Compare slip angle and speed across multiple experiment rollouts.

Usage:
    python compare_experiments.py rollouts1.pt rollouts2.pt ... --labels exp1 exp2 ...

Example:
    python compare_experiments.py entropy0.005.pt entropy0.01.pt entropy0.02.pt \
        --labels entropy=0.005 entropy=0.01 entropy=0.02
"""

import argparse
from pathlib import Path
import torch
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install matplotlib")


def load_metrics(filepath, env_idx=0):
    data = torch.load(filepath, map_location="cpu", weights_only=True)
    obs = data["observations"]  # [T, N, D]
    acts = data["actions"]      # [T, N, 2]

    cur = obs[:, env_idx, :]
    act = acts[:, env_idx, :]

    lin_vel = cur[:, 6:9]
    vx = lin_vel[:, 0].numpy()
    vy = lin_vel[:, 1].numpy()

    speed = np.sqrt(vx**2 + vy**2)
    slip_deg = np.abs(np.arctan2(vy, vx)) * 180.0 / np.pi

    moving = np.abs(vx) > 0.5
    drift = (slip_deg > 15.0) & moving
    spinout = (slip_deg > 60.0) & moving

    return {
        "speed": speed,
        "slip_deg": slip_deg,
        "mean_speed": speed.mean(),
        "mean_slip_deg": slip_deg[moving].mean() if moving.any() else 0.0,
        "drift_ratio": drift.mean(),
        "spinout_rate": spinout.mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", nargs="+", help="Paths to rollouts .pt files")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--env-idx", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    labels = args.labels or [Path(p).stem for p in args.rollouts]

    all_metrics = []
    for path, label in zip(args.rollouts, labels):
        print(f"Loading {label}...")
        m = load_metrics(path, env_idx=args.env_idx)
        m["label"] = label
        all_metrics.append(m)

    # Print summary table
    print(f"\n{'Label':<25} {'Mean Speed':>12} {'Mean Slip°':>12} {'Drift Ratio':>13} {'Spinout Rate':>13}")
    print("-" * 75)
    for m in all_metrics:
        print(f"{m['label']:<25} {m['mean_speed']:>12.3f} {m['mean_slip_deg']:>12.1f} {m['drift_ratio']:>13.3f} {m['spinout_rate']:>13.3f}")

    # Plot timeseries comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    T_max = max(len(m["slip_deg"]) for m in all_metrics)
    t = np.arange(T_max) * args.dt

    for m in all_metrics:
        n = len(m["slip_deg"])
        axes[0].plot(t[:n], m["slip_deg"], alpha=0.8, label=m["label"])
        axes[1].plot(t[:n], m["speed"], alpha=0.8, label=m["label"])

    axes[0].axhline(15, color="k", linestyle="--", linewidth=1, label="drift threshold (15°)")
    axes[0].axhline(60, color="r", linestyle="--", linewidth=1, label="spinout threshold (60°)")
    axes[0].set_ylabel("Slip Angle (°)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_ylabel("Speed (m/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Experiment Comparison: Slip Angle & Speed")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200)
        print(f"\nSaved to {args.save}")

    plt.show()


if __name__ == "__main__":
    main()