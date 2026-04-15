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
    """Compute performance metrics from rollouts."""
    metrics = {}
    
    if 'actions' in data:
        actions = data['actions']
        num_steps, num_envs, action_dim = actions.shape
        
        if action_dim == 2:
            throttle = actions[:, :, 0]
            steer = actions[:, :, 1]
            
            metrics['mean_throttle'] = throttle.mean().item()
            metrics['mean_steer'] = steer.mean().item()
            metrics['throttle_std'] = throttle.std().item()
            metrics['steer_std'] = steer.std().item()
            
            # Action smoothness (lower std = smoother)
            metrics['action_smoothness'] = 1.0 / (1.0 + throttle.std().item() + steer.std().item())
            
            # Action consistency (how consistent actions are over time)
            throttle_temporal_std = throttle.std(dim=0).mean().item()
            steer_temporal_std = steer.std(dim=0).mean().item()
            metrics['temporal_consistency'] = 1.0 / (1.0 + throttle_temporal_std + steer_temporal_std)
    
    if 'observations' in data:
        obs = data['observations']
        # Observation stability
        metrics['obs_stability'] = 1.0 / (1.0 + obs.std().item())
        
        # Observation trend
        obs_per_timestep = obs.mean(dim=2)
        obs_trend = (obs_per_timestep[-1] - obs_per_timestep[0]).mean().item()
        metrics['obs_trend'] = obs_trend
    
    return metrics


def compare_friction_performance(playback_dir):
    """Compare performance across different friction values."""
    playback_path = Path(playback_dir)
    
    # Find all rollouts files
    rollouts_files = list(playback_path.glob("*rollouts.pt"))
    
    if not rollouts_files:
        print(f"No rollouts files found in {playback_dir}")
        return
    
    print(f"Found {len(rollouts_files)} rollouts files")
    
    # Load and analyze each file
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
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("Performance Comparison Across Friction Values")
    print(f"{'='*80}")
    
    # Sort by friction value
    numeric_results = [(r['friction'], r) for r in results if isinstance(r['friction'], (int, float))]
    default_results = [(r['friction'], r) for r in results if not isinstance(r['friction'], (int, float))]
    numeric_results.sort(key=lambda x: x[0])
    all_results = numeric_results + default_results
    
    # Print header
    if all_results:
        metric_keys = list(all_results[0][1]['metrics'].keys())
        print(f"\n{'Friction':<15}", end="")
        for key in metric_keys:
            print(f"{key:<20}", end="")
        print()
        print("-" * (15 + 20 * len(metric_keys)))
        
        # Print data
        for friction, result in all_results:
            print(f"{str(friction):<15}", end="")
            for key in metric_keys:
                value = result['metrics'].get(key, 0.0)
                print(f"{value:<20.4f}", end="")
            print()
    
    # Plot comparison if matplotlib available
    if HAS_MATPLOTLIB and len(numeric_results) > 1:
        plot_friction_comparison(all_results, playback_path)
    
    return results


def plot_friction_comparison(results, save_dir):
    """Plot performance metrics vs friction."""
    if not HAS_MATPLOTLIB:
        return
    
    numeric_results = [(r['friction'], r) for r in results if isinstance(r['friction'], (int, float))]
    if len(numeric_results) < 2:
        print("\nNeed at least 2 numeric friction values to plot")
        return
    
    numeric_results.sort(key=lambda x: x[0])
    frictions = [r[0] for r in numeric_results]
    
    # Get all metric keys
    metric_keys = list(numeric_results[0][1]['metrics'].keys())
    
    # Create subplots
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
    
    # Hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = save_dir / "friction_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved comparison plot to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare performance across friction values")
    parser.add_argument("playback_dir", type=str, help="Directory containing rollouts files")
    parser.add_argument("--show", action="store_true", help="Display plots")
    
    args = parser.parse_args()
    
    compare_friction_performance(args.playback_dir)


if __name__ == "__main__":
    main()





