"""visualize_eval_dino.py — Visualize a diagnostic .npz from evaluate_policy_real.py.

Usage:
    python3 examples/tests/visualize_eval_dino.py \
        --npz ./logs/diagnostics/episode_000.npz \
        --output ./logs/diagnostics/episode_000_analysis.pdf
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


JOINT_NAMES = [f"J{i+1}" for i in range(7)]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def load(npz_path: str) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _safe(arr, fallback=None):
    """Return arr if non-empty, else fallback."""
    if arr is None or arr.size == 0:
        return fallback
    return arr


def plot_episode(data: dict, output_pdf: str) -> None:
    joint_pos   = _safe(data.get("joint_positions"))    # (T, 7)
    gripper_pos = _safe(data.get("gripper_positions"))  # (T, 1)
    obs_ts      = _safe(data.get("obs_timestamps"))     # (T,)

    rl_noises   = _safe(data.get("rl_noises"))          # (N_infer, horizon, noise_dim)
    pi0_chunks  = _safe(data.get("pi0_chunks"))         # (N_infer, horizon, 32)
    infer_steps = _safe(data.get("infer_steps"))        # (N_infer,)

    exec_pos    = _safe(data.get("exec_positions"))     # (N_exec, 7)
    exec_ts     = _safe(data.get("exec_timestamps"))    # (N_exec,)

    success     = bool(data.get("success", False))
    duration_s  = float(data.get("duration_s", 0))
    episode_id  = int(data.get("episode_id", 0))
    action_scale = float(data.get("action_scale", 0.5))
    max_jd      = float(data.get("max_joint_delta", 0.1))
    dt_step     = float(data.get("dt_step", 1 / 10))

    T = len(joint_pos) if joint_pos is not None else 0
    t_obs = (obs_ts - obs_ts[0]) if obs_ts is not None and T > 0 else np.arange(T) * dt_step

    # ── Commanded joint-velocity from pi0 (first chunk per inference) ──────────
    # pi0_chunks[i] has shape (horizon, 32); first 7 dims are joint velocities
    cmd_vels = None
    cmd_t    = None
    if pi0_chunks is not None and infer_steps is not None:
        cmd_vels = pi0_chunks[:, 0, :7]          # (N_infer, 7) — first step of chunk
        cmd_t    = infer_steps.astype(float) * dt_step

    # ── Actual joint displacement (finite difference of observed positions) ──────
    obs_vel = None
    if joint_pos is not None and T > 1:
        obs_vel = np.diff(joint_pos, axis=0) / dt_step   # (T-1, 7) rad/s

    with PdfPages(output_pdf) as pdf:

        # ── Page 1: Joint positions ──────────────────────────────────────────────
        fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)
        fig.suptitle(
            f"Episode {episode_id}  |  {'SUCCESS ✓' if success else 'FAILURE ✗'}  "
            f"|  {duration_s:.1f}s  |  action_scale={action_scale}  max_Δ={max_jd:.3f}",
            fontsize=12,
        )
        for j, ax in enumerate(axes):
            if joint_pos is not None:
                ax.plot(t_obs, joint_pos[:, j], color=COLORS[j], lw=1.5,
                        label="observed")
            if exec_pos is not None and exec_ts is not None:
                t_exec_rel = exec_ts - obs_ts[0] if obs_ts is not None else exec_ts
                ax.scatter(t_exec_rel, exec_pos[:, j], color="red", s=8, zorder=5,
                           label="commanded" if j == 0 else None)
            ax.set_ylabel(JOINT_NAMES[j], fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[0].legend(fontsize=8, loc="upper right")
        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: Policy action vs actual joint velocity ───────────────────────
        fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)
        fig.suptitle("Joint velocity: pi0 commanded (×max_Δ) vs observed", fontsize=12)
        for j, ax in enumerate(axes):
            if cmd_vels is not None:
                cmd_rad = cmd_vels[:, j] * max_jd / dt_step   # convert to rad/s
                ax.step(cmd_t, cmd_rad, where="post", color=COLORS[j], lw=1.5,
                        label="pi0 cmd (rad/s)")
            if obs_vel is not None:
                t_vel = t_obs[1:]
                ax.plot(t_vel, obs_vel[:, j], color="gray", lw=0.8, alpha=0.7,
                        label="actual Δpos/dt")
            ax.set_ylabel(JOINT_NAMES[j], fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[0].legend(fontsize=8, loc="upper right")
        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 3: RL noise norms + gripper + inference timeline ─────────────────
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 1, hspace=0.45)

        # 3a: RL noise L2 norm per inference step
        ax0 = fig.add_subplot(gs[0])
        if rl_noises is not None and infer_steps is not None:
            noise_norms = np.linalg.norm(
                rl_noises.reshape(len(rl_noises), -1), axis=1)
            ax0.bar(infer_steps * dt_step, noise_norms, width=dt_step * 0.8,
                    color="steelblue", alpha=0.8)
            ax0.axhline(noise_norms.mean(), color="red", lw=1, ls="--",
                        label=f"mean={noise_norms.mean():.2f}")
            ax0.legend(fontsize=8)
        ax0.set_ylabel("‖RL noise‖₂")
        ax0.set_title("RL noise magnitude per inference")
        ax0.grid(True, alpha=0.3)

        # 3b: Gripper position
        ax1 = fig.add_subplot(gs[1])
        if gripper_pos is not None and T > 0:
            g = gripper_pos[:, 0] if gripper_pos.ndim == 2 else gripper_pos
            ax1.plot(t_obs, g, color="darkorange", lw=1.5)
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_ylabel("Gripper (0=closed, 1=open)")
        ax1.set_title("Gripper position")
        ax1.grid(True, alpha=0.3)

        # 3c: pi0 first-step gripper command
        ax2 = fig.add_subplot(gs[2])
        if pi0_chunks is not None and infer_steps is not None:
            gripper_cmds = pi0_chunks[:, 0, -1]    # last dim = gripper
            ax2.step(infer_steps * dt_step, gripper_cmds, where="post",
                     color="purple", lw=1.5)
            ax2.set_ylim(-1.1, 1.1)
            ax2.set_ylabel("Gripper cmd (raw)")
        ax2.set_title("pi0 gripper command (raw, before binarize)")
        ax2.set_xlabel("Time (s)")
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"Episode {episode_id} — noise & gripper", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved: {output_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="Path to episode_NNN.npz")
    parser.add_argument("--output", default=None,
        help="Output PDF path. Defaults to <npz_stem>_analysis.pdf")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    out_path = args.output or str(npz_path.with_suffix("").with_name(
        npz_path.stem + "_analysis.pdf"))

    data = load(str(npz_path))
    plot_episode(data, out_path)


if __name__ == "__main__":
    main()
