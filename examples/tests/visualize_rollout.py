#!/usr/bin/env python3
"""Visualize diagnostic data from a pi0 evaluation rollout.

Reads a .npz file produced by DiagnosticLogger and generates six figures:
  Fig 1: Joint trajectory — planned (commanded) vs actual (robot state)
  Fig 2: Action chunk utilization — how many actions are new/stale/scheduled
  Fig 3: Timing analysis — tick durations, deadline overruns, actual frequency
  Fig 4: t_obs drift and state history buffer coverage
  Fig 5: Action continuity across chunk boundaries
  Fig 6: Gripper command timeline

Usage:
    cd ~/yingxi/dsrl_pi0
    conda activate dsrl_pi0   # needs matplotlib, numpy
    python3 examples/tests/visualize_rollout.py \\
        --npz ./logs/diagnostics/episode_000.npz \\
        --output ./logs/diagnostics/episode_000_analysis.pdf
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend, works on headless servers
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages
except ImportError:
    print("ERROR: matplotlib is required.  pip install matplotlib")
    sys.exit(1)

from examples.tests.diagnostic_logger import DiagnosticLogger


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rel(t: np.ndarray, t0: float) -> np.ndarray:
    """Convert absolute wall-clock times to seconds from episode start."""
    return t - t0


def _joint_snapshot_matrix(data: dict, t0: float) -> tuple[np.ndarray, np.ndarray]:
    """(time_vec, joint_matrix (N×7)) from snapshot observations."""
    t_ticks  = _rel(data["t_tick"], t0)
    snapshots = data["joint_pos_snapshot"]   # object array of (7,) arrays
    if len(snapshots) == 0:
        return np.array([]), np.empty((0, 7))
    mat = np.stack([s for s in snapshots])   # (N, 7)
    return t_ticks, mat


def _joint_interp_matrix(data: dict, t0: float) -> tuple[np.ndarray, np.ndarray]:
    """(time_vec, joint_matrix (N×7)) from interpolated observations.
    Rows with all-NaN joints (no interpolation available) are kept as-is.
    """
    t_ticks = _rel(data["t_tick"], t0)
    interps = data["joint_pos_interp"]
    mat = np.stack([s for s in interps])   # (N, 7), may contain NaN
    return t_ticks, mat


def _planned_waypoints(data: dict, t0: float) -> tuple[np.ndarray, np.ndarray]:
    """Collect all arm waypoints sent across all ticks into flat arrays.

    Returns (times (M,), positions (M, 7)) sorted by time.
    """
    all_times = []
    all_pos   = []
    arm_times_arr     = data["arm_times_sent"]
    arm_positions_arr = data["arm_positions_sent"]
    for at, ap in zip(arm_times_arr, arm_positions_arr):
        if len(at) > 0:
            all_times.append(at)
            all_pos.append(ap)
    if not all_times:
        return np.array([]), np.empty((0, 7))
    times = np.concatenate(all_times)
    pos   = np.concatenate(all_pos, axis=0)
    order = np.argsort(times)
    return _rel(times[order], t0), pos[order]


def _chunk_boundaries(data: dict, t0: float) -> np.ndarray:
    """Times when add_waypoints was called (chunk boundary markers)."""
    arm_times_arr = data["arm_times_sent"]
    boundaries = []
    t_ticks = data["t_tick"]
    for i, at in enumerate(arm_times_arr):
        if len(at) > 0:
            boundaries.append(t_ticks[i] - t0)
    return np.array(boundaries)


# ─────────────────────────────────────────────────────────────────────────────
# Figure functions
# ─────────────────────────────────────────────────────────────────────────────

def fig_joint_trajectory(data: dict, t0: float) -> plt.Figure:
    """Fig 1: Planned vs actual joint positions for all 7 DOF."""
    fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)
    fig.suptitle("Fig 1: Joint Trajectory — Planned vs Actual", fontsize=13, fontweight="bold")

    t_snap, snap_mat  = _joint_snapshot_matrix(data, t0)
    t_interp, interp_mat = _joint_interp_matrix(data, t0)
    t_plan, plan_mat  = _planned_waypoints(data, t0)
    boundaries        = _chunk_boundaries(data, t0)

    joint_names = [f"j{i}" for i in range(7)]

    for j, ax in enumerate(axes):
        # Actual (robot state snapshot)
        ax.plot(t_snap, snap_mat[:, j], color="tab:red",  lw=1.2, label="actual (snapshot)", zorder=3)
        # Interpolated (sent to inference)
        valid_interp = ~np.isnan(interp_mat[:, j])
        if valid_interp.any():
            ax.plot(t_interp[valid_interp], interp_mat[valid_interp, j],
                    color="tab:green", lw=1.0, ls="--", label="interp (to inference)", zorder=2)
        # Planned (commanded to HighFreqController)
        if len(t_plan) > 0:
            ax.scatter(t_plan, plan_mat[:, j], color="tab:blue", s=12, zorder=4, label="planned waypoints")
        # Chunk boundaries
        for bnd in boundaries:
            ax.axvline(bnd, color="gray", lw=0.5, alpha=0.5, ls=":")

        ax.set_ylabel(f"{joint_names[j]} (rad)", fontsize=8)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time from episode start (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def fig_action_chunk_utilization(data: dict) -> plt.Figure:
    """Fig 2: Per-inference-cycle action chunk utilization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle("Fig 2: Action Chunk Utilization", fontsize=13, fontweight="bold")

    recv_mask  = data["infer_recv"].astype(bool)
    n_returned = data["n_returned"][recv_mask].astype(float)
    n_is_new   = data["n_is_new"][recv_mask].astype(float)
    n_sched    = data["n_scheduled"][recv_mask].astype(float)
    n_cycles   = len(n_returned)

    if n_cycles == 0:
        ax1.text(0.5, 0.5, "No inference results received", ha="center", va="center")
        ax2.text(0.5, 0.5, "No inference results received", ha="center", va="center")
        fig.tight_layout()
        return fig

    x = np.arange(n_cycles)
    n_stale    = n_returned - n_is_new
    n_truncated = np.maximum(n_is_new - n_sched, 0)

    ax1.bar(x, n_sched,                          label="scheduled (executed)",       color="tab:green")
    ax1.bar(x, n_truncated, bottom=n_sched,       label="is_new but exec_steps limit", color="tab:orange")
    ax1.bar(x, n_stale, bottom=n_is_new,          label="stale (filtered by is_new)", color="tab:red", alpha=0.7)
    ax1.set_ylabel("# actions")
    ax1.set_xlabel("Inference cycle")
    ax1.set_title("Per-cycle chunk breakdown (stacked)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, n_sched, color="tab:green", marker="o", ms=4, label="n_scheduled")
    ax2.plot(x, n_is_new, color="tab:blue",  marker="s", ms=3, ls="--", label="n_is_new")
    ax2.axhline(data["n_scheduled"].max() if len(data["n_scheduled"]) else 6,
                color="gray", ls=":", lw=1, label="execution_steps ceiling")
    ax2.set_xlabel("Inference cycle")
    ax2.set_ylabel("# actions")
    ax2.set_title("Scheduled vs is_new over time")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def fig_timing(data: dict, t0: float, target_hz: float = 10.0) -> plt.Figure:
    """Fig 3: Tick duration histogram, overrun timeline, instantaneous frequency."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle("Fig 3: Timing Analysis", fontsize=13, fontweight="bold")

    t_ticks = data["t_tick"]
    if len(t_ticks) < 2:
        for ax in (ax1, ax2, ax3):
            ax.text(0.5, 0.5, "Not enough ticks", ha="center", va="center")
        fig.tight_layout()
        return fig

    tick_dts_ms = np.diff(t_ticks) * 1000.0
    overrun_ms  = data["tick_overrun_ms"]
    t_rel       = _rel(t_ticks, t0)
    inst_hz     = 1000.0 / np.maximum(tick_dts_ms, 1.0)   # avoid div-by-zero

    target_dt_ms = 1000.0 / target_hz

    # Histogram of tick durations
    ax1.hist(tick_dts_ms, bins=50, color="tab:blue", edgecolor="white", lw=0.3)
    ax1.axvline(target_dt_ms, color="red", ls="--", label=f"target {target_hz} Hz = {target_dt_ms:.0f} ms")
    ax1.set_xlabel("Tick duration (ms)")
    ax1.set_ylabel("Count")
    ax1.set_title("Tick duration distribution")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Overrun timeline
    ax2.plot(t_rel, overrun_ms, color="tab:orange", lw=0.8)
    ax2.fill_between(t_rel, 0, overrun_ms, alpha=0.3, color="tab:orange")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Overrun (ms)")
    ax2.set_title("Tick deadline overrun over time (0 = on time)")
    ax2.grid(True, alpha=0.3)

    # Instantaneous frequency
    ax3.plot(t_rel[1:], inst_hz, color="tab:green", lw=0.8)
    ax3.axhline(target_hz, color="red", ls="--", lw=1, label=f"target {target_hz} Hz")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Frequency (Hz)")
    ax3.set_title("Instantaneous control frequency")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def fig_tobs_and_state_buffer(data: dict, t0: float) -> plt.Figure:
    """Fig 4: t_obs drift and state history coverage."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle("Fig 4: t_obs Drift & State History Buffer", fontsize=13, fontweight="bold")

    t_rel   = _rel(data["t_tick"], t0)
    t_obs   = data["t_obs"]
    t_ticks = data["t_tick"]
    drift_ms = (t_ticks - t_obs) * 1000.0   # how far in the past t_obs was

    ax1.plot(t_rel, drift_ms, color="tab:purple", lw=1.0)
    ax1.axhline(np.mean(drift_ms), color="red", ls="--", lw=1,
                label=f"mean = {np.mean(drift_ms):.1f} ms")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("t_obs age (ms)")
    ax1.set_title("t_obs drift (camera frame age from now) — should be ≈ camera_obs_latency")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_rel, data["state_history_n"], color="tab:blue", lw=1.0)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Entries")
    ax2.set_title("State history buffer size requested = 100; should stay at 100")
    ax2.set_ylim(0, 120)
    ax2.grid(True, alpha=0.3)

    ax3.plot(t_rel, data["state_history_span"] * 1000.0, color="tab:cyan", lw=1.0)
    ax3.axhline(500, color="red", ls="--", lw=1, label="expected ~500ms @ 200Hz×100")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Span (ms)")
    ax3.set_title("State history time span covered — should be ≈ 500 ms")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def fig_action_continuity(data: dict, t0: float, n_joints: int = 3) -> plt.Figure:
    """Fig 5: Arm waypoints across chunk boundaries — check for discontinuities."""
    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 3 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]
    fig.suptitle("Fig 5: Action Continuity Across Chunk Boundaries", fontsize=13, fontweight="bold")

    arm_times_arr     = data["arm_times_sent"]
    arm_positions_arr = data["arm_positions_sent"]
    t_ticks           = data["t_tick"]

    # Assign a colour per chunk
    chunk_colors = plt.cm.tab10.colors
    chunk_idx = 0

    for i, (at, ap) in enumerate(zip(arm_times_arr, arm_positions_arr)):
        if len(at) == 0:
            continue
        chunk_t = _rel(np.asarray(at), t0)
        chunk_p = np.asarray(ap)   # (N, 7)
        color = chunk_colors[chunk_idx % len(chunk_colors)]
        for j, ax in enumerate(axes):
            ax.scatter(chunk_t, chunk_p[:, j], color=color, s=18, zorder=4)
            ax.plot(chunk_t, chunk_p[:, j], color=color, lw=0.8, alpha=0.7)
        chunk_idx += 1

    boundaries = _chunk_boundaries(data, t0)
    for bnd in boundaries:
        for ax in axes:
            ax.axvline(bnd, color="gray", lw=0.7, alpha=0.4, ls="--")

    for j, ax in enumerate(axes):
        ax.set_ylabel(f"j{j} (rad)", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time from episode start (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def fig_gripper_timeline(data: dict, t0: float) -> plt.Figure:
    """Fig 6: Gripper command timeline."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    fig.suptitle("Fig 6: Gripper Command Timeline", fontsize=13, fontweight="bold")

    t_rel  = _rel(data["t_tick"], t0)
    g_cmds = data["gripper_cmd"]

    sent_mask = ~np.isnan(g_cmds)
    if sent_mask.any():
        ax.step(t_rel[sent_mask], g_cmds[sent_mask], where="post",
                color="tab:brown", lw=1.5, label="gripper command")
        ax.scatter(t_rel[sent_mask], g_cmds[sent_mask], color="tab:brown", s=20, zorder=5)

    ax.set_xlim(t_rel[0], t_rel[-1])
    ax.set_ylim(-0.1, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0 (open)", "1 (closed)"])
    ax.set_xlabel("Time from episode start (s)")
    ax.set_ylabel("Gripper position")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize pi0 eval rollout diagnostic data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python3 examples/tests/visualize_rollout.py \\
      --npz ./logs/diagnostics/episode_000.npz \\
      --output ./logs/diagnostics/episode_000_analysis.pdf
        """,
    )
    parser.add_argument("--npz",    required=True, help="Path to diagnostic .npz file.")
    parser.add_argument("--output", default=None,  help="Output PDF path. Default: <npz>.pdf")
    parser.add_argument("--hz",     default=10.0,  type=float, help="Target control frequency.")
    parser.add_argument("--joints", default=3,     type=int,
                        help="How many joints to show in Fig 5 (continuity). Default: 3.")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found")
        sys.exit(1)

    out_path = Path(args.output) if args.output else npz_path.with_suffix(".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {npz_path} ...")
    data = DiagnosticLogger.load(npz_path)

    n_ticks = len(data["t_tick"])
    t0      = float(data["t_tick"][0])
    dur_s   = float(data["t_tick"][-1]) - t0

    print(f"  {n_ticks} ticks over {dur_s:.1f}s")

    figures = [
        ("Joint Trajectory",         fig_joint_trajectory(data, t0)),
        ("Action Chunk Utilization", fig_action_chunk_utilization(data)),
        ("Timing",                   fig_timing(data, t0, target_hz=args.hz)),
        ("t_obs & State Buffer",     fig_tobs_and_state_buffer(data, t0)),
        ("Action Continuity",        fig_action_continuity(data, t0, n_joints=args.joints)),
        ("Gripper Timeline",         fig_gripper_timeline(data, t0)),
    ]

    print(f"Saving {len(figures)} figures to {out_path} ...")
    with PdfPages(str(out_path)) as pdf:
        for title, fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"  [✓] {title}")

    print(f"\nDone: {out_path}")
    print("Open with: evince / Preview / any PDF viewer")


if __name__ == "__main__":
    main()
