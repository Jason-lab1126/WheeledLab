import argparse
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import LineCollection
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it (e.g., `pip install matplotlib`)."
    ) from exc


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding. window<=1 returns x."""
    window = int(window)
    if window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(xp, kernel, mode="valid")


def _segments_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive-exclusive segments where mask is True."""
    if mask.size == 0:
        return []
    m = mask.astype(bool)
    dm = np.diff(m.astype(np.int8), prepend=0, append=0)
    starts = np.where(dm == 1)[0]
    ends = np.where(dm == -1)[0]
    return list(zip(starts, ends))


def _pick_top_events(beta: np.ndarray, drift_mask: np.ndarray, k: int = 2) -> list[tuple[int, int]]:
    """Pick up to k drift segments ranked by peak beta."""
    segs = _segments_from_mask(drift_mask)
    if not segs:
        return []
    scored = []
    for a, b in segs:
        if b <= a:
            continue
        scored.append((float(beta[a:b].max()), a, b))
    scored.sort(reverse=True, key=lambda t: t[0])
    out = []
    for _, a, b in scored[: max(1, int(k))]:
        out.append((a, b))
    return out


def _plot_track(
    x: np.ndarray,
    y: np.ndarray,
    speed: np.ndarray,
    drift_mask: np.ndarray,
    *,
    title: str,
    rectangles: list[tuple[float, float, float, float]] | None = None,
    arrow_stride: int = 18,
):
    """Plot XY trajectory colored by speed and highlight drift region."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # Color the trajectory by speed using a LineCollection.
    pts = np.stack([x, y], axis=1).astype(np.float32)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, array=speed[:-1], cmap="plasma", linewidths=2.2)
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("speed (m/s)")

    # Direction arrows (downsampled).
    stride = max(2, int(arrow_stride))
    dx = np.diff(x)
    dy = np.diff(y)
    ax.quiver(
        x[:-1:stride],
        y[:-1:stride],
        dx[::stride],
        dy[::stride],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0035,
        color="k",
        alpha=0.8,
    )

    # Highlight drift points (optional visual cue).
    if drift_mask.any():
        ax.scatter(x[drift_mask], y[drift_mask], s=10, c="k", alpha=0.75, label="drift")

    # Draw rectangles (e.g., event-local drift regions).
    if rectangles:
        for (rx, ry, rw, rh) in rectangles:
            ax.add_patch(Rectangle((rx, ry), rw, rh, fill=False, lw=3, edgecolor="k"))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    return fig, ax


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot drift signals from rollouts.pt (play_policy.py output)")
    parser.add_argument("rollouts_path", type=str, help="Path to *-rollouts.pt saved by play_policy.py")
    parser.add_argument("--env-idx", type=int, default=0, help="Which env index to plot from the batch")
    parser.add_argument("--start-step", type=int, default=0, help="Start step index (inclusive)")
    parser.add_argument(
        "--end-step",
        type=int,
        default=None,
        help="End step index (exclusive). Defaults to end of rollout.",
    )
    parser.add_argument("--beta-min", type=float, default=0.25, help="Slip angle threshold (rad) for drift window")
    parser.add_argument("--min-vx", type=float, default=1.0, help="Minimum |v_x| (m/s) gating for drift window")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=11,
        help="Centered moving-average window (steps) for slip/speed/commands. Use 1 to disable.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Timestep (seconds) used for time axis (default: 0.02).",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=2,
        help="How many drift events to plot as stacked panels (default: 2).",
    )
    parser.add_argument(
        "--event-window-s",
        type=float,
        default=1.2,
        help="Length (seconds) of each event panel window (default: 1.2).",
    )
    parser.add_argument(
        "--event-anchor",
        type=str,
        default="peak",
        choices=["peak", "start"],
        help="How to align the event window: at segment start or at peak slip (default: peak).",
    )
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the timeseries figure")
    parser.add_argument("--save-track", type=str, default=None, help="Optional path to save the track/XY figure")
    parser.add_argument(
        "--track-smooth-window",
        type=int,
        default=None,
        help="Optional moving-average window for x/y on the track plot. Defaults to --smooth-window.",
    )
    parser.add_argument("--track-arrow-stride", type=int, default=18, help="Arrow spacing on track plot (steps)")
    args = parser.parse_args()

    rollouts_path = Path(args.rollouts_path)
    # rollouts.pt is created by our own script (play_policy.py). Use weights_only=True
    # to avoid unpickling arbitrary objects and to silence PyTorch's security warning.
    data = torch.load(rollouts_path, map_location="cpu", weights_only=True)

    obs = data["observations"]  # expected: [T, N, 14]
    acts = data["actions"]      # expected: [T, N, 2]

    if not isinstance(obs, torch.Tensor) or obs.ndim != 3:
        raise ValueError(f"Unexpected observations type/shape: {type(obs)} {getattr(obs, 'shape', None)}")
    if not isinstance(acts, torch.Tensor) or acts.ndim != 3:
        raise ValueError(f"Unexpected actions type/shape: {type(acts)} {getattr(acts, 'shape', None)}")

    T, N, D = obs.shape
    if args.env_idx < 0 or args.env_idx >= N:
        raise ValueError(f"--env-idx out of range: {args.env_idx} (num_envs={N})")
    if D < 12:
        raise ValueError(f"Observation dim too small ({D}); expected at least pose+vel terms.")

    start = int(args.start_step)
    end = T if args.end_step is None else int(args.end_step)
    if start < 0 or start >= T:
        raise ValueError(f"--start-step out of range: {start} (T={T})")
    if end <= start or end > T:
        raise ValueError(f"--end-step out of range: {end} (start={start}, T={T})")

    # For your RecurrentDriftObsCfg PolicyCfg, the saved observation layout is:
    # root_pos(3), root_euler_xyz(3), base_lin_vel(3), base_ang_vel(3), last_action(2) = 14
    cur = obs[start:end, args.env_idx, :]   # [Ts, 14]
    act = acts[start:end, args.env_idx, :]  # [Ts, 2]

    root_pos = cur[:, 0:3]
    lin_vel = cur[:, 6:9]

    x = _to_numpy(root_pos[:, 0])
    y = _to_numpy(root_pos[:, 1])

    vx = _to_numpy(lin_vel[:, 0])
    vy = _to_numpy(lin_vel[:, 1])

    speed = np.sqrt(vx**2 + vy**2)
    beta = np.abs(np.arctan2(vy, vx))  # rad

    throttle = _to_numpy(act[:, 0])
    steer = _to_numpy(act[:, 1])

    # Smooth to reduce observation-noise visual clutter (does not change underlying rollout).
    w = int(args.smooth_window)
    beta_s = _moving_average(beta, w)
    speed_s = _moving_average(speed, w)
    throttle_s = _moving_average(throttle, w)
    steer_s = _moving_average(steer, w)

    drift_mask = (beta_s > float(args.beta_min)) & (np.abs(vx) > float(args.min_vx))
    drift_idx = np.where(drift_mask)[0]

    rect = None
    if drift_idx.size > 0:
        xs = x[drift_idx]
        ys = y[drift_idx]
        pad = 0.15
        rect = (xs.min() - pad, ys.min() - pad, (xs.max() - xs.min()) + 2 * pad, (ys.max() - ys.min()) + 2 * pad)

    dt = float(args.dt)

    # Pick top-K drift events and plot them like the reference: (Slip, Speed, Command) x K
    events = _pick_top_events(beta_s, drift_mask, k=int(args.events))
    if not events:
        # Fallback: plot the whole window as a single event
        events = [(0, len(beta_s))]

    steps_per_event = max(1, int(round(float(args.event_window_s) / dt)))

    fig = plt.figure(figsize=(6.0, 2.2 * 3 * len(events)))
    gs = fig.add_gridspec(nrows=3 * len(events), ncols=1, hspace=0.35)

    for i, (a, b) in enumerate(events):
        if args.event_anchor == "start":
            anchor = a
        else:
            # peak beta within the segment
            anchor = int(a + np.argmax(beta_s[a:b]))

        s0 = max(0, anchor)
        s1 = min(len(beta_s), s0 + steps_per_event)
        # if close to end, shift window left to keep length
        if s1 - s0 < steps_per_event:
            s0 = max(0, s1 - steps_per_event)

        tt = (np.arange(s0, s1, dtype=np.float32) - s0) * dt

        # Convert slip to degrees to match your reference plots.
        beta_deg = beta_s[s0:s1] * (180.0 / np.pi)
        speed_w = speed_s[s0:s1]
        thr_w = throttle_s[s0:s1]
        ste_w = steer_s[s0:s1]

        ax_slip = fig.add_subplot(gs[i * 3 + 0, 0])
        ax_spd = fig.add_subplot(gs[i * 3 + 1, 0], sharex=ax_slip)
        ax_cmd = fig.add_subplot(gs[i * 3 + 2, 0], sharex=ax_slip)

        ax_slip.plot(tt, beta_deg, lw=1.5)
        ax_slip.set_ylabel("Slip Angle (°)")
        ax_slip.grid(alpha=0.35)

        ax_spd.plot(tt, speed_w, lw=1.5)
        ax_spd.set_ylabel("Speed (m/s)")
        ax_spd.grid(alpha=0.35)

        ax_cmd.plot(tt, thr_w, color="tab:green", lw=1.5, label="Throttle")
        ax_cmd.plot(tt, ste_w, color="tab:red", lw=1.5, label="Steering")
        ax_cmd.set_ylabel("Command")
        ax_cmd.set_xlabel("Seconds")
        ax_cmd.set_ylim(-1.05, 1.05)
        ax_cmd.grid(alpha=0.35)
        ax_cmd.legend(loc="upper left", frameon=False, ncols=2)

        # reduce clutter like the reference figure
        plt.setp(ax_slip.get_xticklabels(), visible=False)
        plt.setp(ax_spd.get_xticklabels(), visible=False)

    fig.suptitle(f"{rollouts_path}  (env {args.env_idx})", fontsize=9)
    # Constrained layout plays nicer with lots of shared axes than tight_layout.
    try:
        fig.set_constrained_layout(True)
    except Exception:
        fig.tight_layout(rect=(0, 0, 1, 0.98))

    if args.save is not None:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"[INFO] Saved figure to: {out_path}")

    # Optional: track plot (XY) highlighting drift.
    if args.save_track is not None:
        track_out = Path(args.save_track)
        track_out.parent.mkdir(parents=True, exist_ok=True)
        # Smooth x/y for visualization (policy obs contains injected noise).
        tw = int(args.track_smooth_window) if args.track_smooth_window is not None else int(args.smooth_window)
        x_s = _moving_average(x, tw)
        y_s = _moving_average(y, tw)

        # Build event-specific rectangles based on drift points inside each event window.
        rects: list[tuple[float, float, float, float]] = []
        pad = 0.15
        for (a, b) in events:
            m = drift_mask[a:b]
            if not m.any():
                continue
            xs = x_s[a:b][m]
            ys = y_s[a:b][m]
            rx, ry = xs.min() - pad, ys.min() - pad
            rw, rh = (xs.max() - xs.min()) + 2 * pad, (ys.max() - ys.min()) + 2 * pad
            rects.append((float(rx), float(ry), float(rw), float(rh)))

        fig2, _ = _plot_track(
            x_s,
            y_s,
            speed_s,
            drift_mask,
            title=f"Trajectory + drift highlight (env {args.env_idx})",
            rectangles=rects,
            arrow_stride=int(args.track_arrow_stride),
        )
        fig2.savefig(track_out, dpi=250, bbox_inches="tight")
        print(f"[INFO] Saved track figure to: {track_out}")

    plt.show()


if __name__ == "__main__":
    main()