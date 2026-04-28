"""
Compare performance across different friction values.

Usage:
    python compare_friction_performance.py <playback_dir>

    This will find all rollouts files in the directory and compare them.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import re

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting disabled.")


def extract_friction_from_filename(filename):
    """Extract friction value from filename like 'play-name_friction0.5-rollouts.pt'"""
    match = re.search(r'friction([\d.]+)', str(filename))
    if match:
        return float(match.group(1))
    return None


def load_rollouts(filepath):
    """Load rollouts data from .pt file."""
    return torch.load(filepath, map_location='cpu')


def compute_performance_metrics(data):
    """Compute drift-relevant performance metrics from rollouts.
    Obs layout (RecurrentDriftObsCfg): root_pos(3), root_euler(3), base_lin_vel(3), base_ang_vel(3), last_action(2)
    """
    metrics = {}

    if 'observations' in data:
        obs = data['observations']  # [T, N, D]
        lin_vel = obs[:, :, 6:9]   # vx, vy, vz in base frame
        vx = lin_vel[:, :, 0]
        vy = lin_vel[:, :, 1]

        speed = torch.sqrt(vx**2 + vy**2)
        slip_angle = torch.abs(torch.atan2(vy, vx))  # radians

        # only count when actually moving
        moving_mask = torch.abs(vx) > 0.5

        # mean speed
        metrics['mean_speed_ms'] = speed.mean().item()

        # mean slip angle when moving (degrees)
        if moving_mask.any():
            metrics['mean_slip_angle_deg'] = (slip_angle[moving_mask].mean() * 180.0 / 3.14159).item()
        else:
            metrics['mean_slip_angle_deg'] = 0.0

        # drift ratio: fraction of time slip > 15 deg (0.25 rad)
        drifting = (slip_angle > 0.25) & moving_mask
        metrics['drift_ratio'] = drifting.float().mean().item()

        # spinout rate: slip > 60 deg (1.05 rad) = out of control
        spinout = (slip_angle > 1.05) & moving_mask
        metrics['spinout_rate'] = spinout.float().mean().item()

        # cross track distance (using x position as proxy on straights)
        pos = obs[:, :, 0:3]
        cross_track = torch.abs(pos[:, :, 0] - 0.8)  # 0.8 = LINE_RADIUS
        metrics['mean_cross_track_m'] = cross_track.mean().item()

    if 'actions' in data:
        actions = data['actions']
        throttle = actions[:, :, 0]
        steer = actions[:, :, 1]
        metrics['mean_throttle'] = throttle.mean().item()
        metrics['mean_steer_abs'] = steer.abs().mean().item()

    return metrics


def compare_friction_performance(playback_dir):
    """Compare performance across different friction values."""
    playback_path = Path(playback_dir)

    rollouts_files = list(playback_path.glob("*rollouts.pt"))

    if not rollouts_files:
        print(f"No rollouts files found in {playback_dir}")
        return

    print(f"Found {len(rollouts_files)} rollouts files")

    results = []
    for filepath in sorted(rollouts_files):
        friction = extract_friction_from_filename(filepath.name)
        if friction is None:
            friction = "default"

        print(f"\nLoading: {filepath.name}")
        data = load_rollouts(filepath)
        metrics = compute_performance_metrics(data)

        results.append({
            'file': filepath.name,
            'friction': friction,
            'metrics': metrics,
            'data': data
        })

    print(f"\n{'='*80}")
    print("Drift Performance Comparison Across Friction Values")
    print(f"{'='*80}")

    numeric_results = [(r['friction'], r) for r in results if isinstance(r['friction'], (int, float))]
    default_results = [(r['friction'], r) for r in results if not isinstance(r['friction'], (int, float))]
    numeric_results.sort(key=lambda x: x[0])
    all_results = numeric_results + default_results

    if all_results:
        metric_keys = list(all_results[0][1]['metrics'].keys())
        print(f"\n{'Friction':<15}", end="")
        for key in metric_keys:
            print(f"{key:<25}", end="")
        print()
        print("-" * (15 + 25 * len(metric_keys)))

        for friction, result in all_results:
            print(f"{str(friction):<15}", end="")
            for key in metric_keys:
                value = result['metrics'].get(key, 0.0)
                print(f"{value:<25.4f}", end="")
            print()

    if HAS_MATPLOTLIB and len(numeric_results) > 1:
        plot_friction_comparison(all_results, playback_path)

    return results


def plot_friction_comparison(results, save_dir):
    """Plot drift metrics vs friction."""
    if not HAS_MATPLOTLIB:
        return

    numeric_results = [(r['friction'], r) for r in results if isinstance(r['friction'], (int, float))]
    if len(numeric_results) < 2:
        print("\nNeed at least 2 numeric friction values to plot")
        return

    numeric_results.sort(key=lambda x: x[0])
    frictions = [r[0] for r in numeric_results]

    metric_keys = list(numeric_results[0][1]['metrics'].keys())

    num_metrics = len(metric_keys)
    cols = min(3, num_metrics)
    rows = (num_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if num_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, metric_key in enumerate(metric_keys):
        ax = axes[idx]
        values = [r[1]['metrics'].get(metric_key, 0.0) for r in numeric_results]

        ax.plot(frictions, values, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Friction')
        ax.set_ylabel(metric_key.replace('_', ' ').title())
        ax.set_title(f'{metric_key.replace("_", " ").title()} vs Friction')
        ax.grid(True, alpha=0.3)

    for idx in range(num_metrics, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    save_path = save_dir / "drift_friction_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved comparison plot to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare drift performance across friction values")
    parser.add_argument("playback_dir", type=str, help="Directory containing rollouts files")
    parser.add_argument("--show", action="store_true", help="Display plots")

    args = parser.parse_args()

    compare_friction_performance(args.playback_dir)


if __name__ == "__main__":
    main()


