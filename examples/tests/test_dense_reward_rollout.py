#!/usr/bin/env python3
"""Integration test: one rollout (random actions) + Robometer reward calibration.

Tests the complete "image buffer → Robometer → visualization" pipeline
WITHOUT loading pi0 / JAX / DINOv2 (which can segfault in certain GPU setups).

Outputs:
  <output-dir>/reward_frames.mp4         — exterior frames at each query step
  <output-dir>/reward_progress_viz.png   — frames (top) + progress line chart (bottom)
  console                                — per-step reward breakdown

Usage:
  python3 examples/tests/test_dense_reward_rollout.py --gpu 4 --output-dir /tmp/dense_test
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = str(Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dense reward rollout test: random actions + Robometer progress"
    )
    p.add_argument("--output-dir",    default="/tmp/dense_rollout_test")
    p.add_argument("--gpu",           default="0",
                   help="GPU id for Robometer server (CUDA_VISIBLE_DEVICES)")
    p.add_argument("--instruction",   default="pick up the peg and insert it vertically")
    p.add_argument("--max-timesteps", type=int, default=600)
    p.add_argument("--query-freq",    type=int, default=8)
    p.add_argument("--robometer-python",
                   default="/opt/yingxi/envs/robometer/bin/python3")
    p.add_argument("--robometer-checkpoint",
                   default="/opt/yingxi/checkpoint-400")
    p.add_argument("--robometer-base-model",
                   default="/opt/caoyuhang/Pretrained_models/Qwen3-VL-4B-Instruct")
    p.add_argument("--robofac-python",
                   default="/opt/yingxi/envs/robofac/bin/python3")
    p.add_argument("--workspace-bounds",
                   default="/home/gpu4/yingxi/dsrl_pi0/workspace_bounds.json")
    p.add_argument("--video-fps",     type=float, default=4.0)
    p.add_argument("--max-display-frames", type=int, default=30)
    p.add_argument("--progress-reward-scale", type=float, default=1.0)
    p.add_argument("--dpi",           type=int, default=180)
    return p.parse_args()


# ── Video / viz helpers ───────────────────────────────────────────────────────

def _save_video(frames: list[np.ndarray], path: str, fps: float) -> None:
    try:
        import cv2
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for fr in frames:
            writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        writer.release()
    except Exception:
        import imageio
        imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8)


def _visualize(
    frames: list[np.ndarray],
    progress: np.ndarray,
    binary_rewards: np.ndarray,
    dense_rewards: np.ndarray,
    title: str,
    output_path: str,
    max_display: int,
    dpi: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(frames)
    if T == 0:
        print("No frames to visualize"); return

    disp_idx = (
        np.linspace(0, T - 1, min(max_display, T), dtype=int)
        if T > max_display
        else np.arange(T)
    )
    disp_frames   = [frames[i] for i in disp_idx]
    disp_progress = progress[disp_idx]

    ncols = len(disp_frames)
    fig_width = max(14.0, ncols * 0.8)
    fig = plt.figure(figsize=(fig_width, 8))
    grid = fig.add_gridspec(2, ncols, height_ratios=[1.0, 1.5], hspace=0.02, wspace=0.02)

    # ── Top: frames ───────────────────────────────────────────────────────────
    for col, (frame, fi, pv) in enumerate(zip(disp_frames, disp_idx, disp_progress)):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(frame)
        ax.set_title(f"f{int(fi)}\n{pv:.2f}", fontsize=6, pad=1)
        ax.axis("off")

    # ── Bottom: progress line chart ────────────────────────────────────────────
    ax_c = fig.add_subplot(grid[1, :])
    x = np.arange(T)

    ax_c.plot(x, progress, color="#1f77b4", linewidth=2.0, label="Robometer progress")
    ax_c.scatter(x, progress, color="#1f77b4", s=25, zorder=4)

    # Colour vertical bands by binary reward sign
    for xi, br in zip(x, binary_rewards):
        ax_c.axvline(xi, color="#2ca02c" if br >= 0 else "#d62728", alpha=0.12, lw=1.0)

    # Mark displayed frames
    for fi in disp_idx:
        ax_c.axvline(fi, color="grey", alpha=0.4, lw=0.8, linestyle="--")

    ax_c.set_xlim(-0.5, T - 0.5)
    ax_c.set_ylim(-0.05, 1.10)
    ax_c.set_xlabel("query step index", fontsize=9, labelpad=2)
    ax_c.set_ylabel("Robometer progress  [0, 1]", fontsize=9, labelpad=3)
    ax_c.grid(True, alpha=0.3)
    ax_c.tick_params(axis="both", labelsize=8)
    ax_c.legend(fontsize=8, loc="upper left")

    fig.suptitle(title, fontsize=11, y=0.97)
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.08, top=0.91)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Render/display env vars (same as run_sim_dino.sh)
    os.environ.setdefault("DISPLAY",                  ":0")
    os.environ.setdefault("MUJOCO_GL",                "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM",        "egl")
    os.environ.setdefault("TRANSFORMERS_OFFLINE",     "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE",      "1")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ── Imports (lightweight — no JAX / pi0 / DINOv2) ────────────────────────
    from examples.envs.mani_skill_client import ManiSkillRemoteEnv
    from examples.robometer_reward_client import RobometerRewardClient

    # ── Robometer client ──────────────────────────────────────────────────────
    print(f"Starting Robometer server on GPU {args.gpu} (loading Qwen3-4B, ~20s) …")
    robometer_client = RobometerRewardClient(
        robometer_python=args.robometer_python,
        checkpoint_path=args.robometer_checkpoint,
        base_model_id=args.robometer_base_model,
        gpu=args.gpu,
    )
    print("Robometer ready")

    # ── ManiSkill env ─────────────────────────────────────────────────────────
    print("Starting ManiSkill env …")
    workspace_bounds = (
        args.workspace_bounds
        if os.path.exists(args.workspace_bounds)
        else None
    )
    env = ManiSkillRemoteEnv(
        robofac_python=args.robofac_python,
        workspace_bounds_path=workspace_bounds,
    )
    obs, _ = env.reset()
    print("ManiSkill ready — running rollout with random actions …\n")

    # ── Rollout with random actions ───────────────────────────────────────────
    query_freq   = args.query_freq
    max_steps    = args.max_timesteps
    image_buffer = []        # exterior frames at each query step, (224,224,3) uint8
    action_list  = []        # list of steps that were query steps

    is_success     = False
    failure_reason = "timeout"
    env_steps      = 0

    for t in range(max_steps):
        qpos = obs["qpos"]      # (9,)
        ext  = obs["ext"]       # (128,128,3) or (224,224,3)
        # ext comes from ManiSkill at 224×224 (server crops to 224 in _extract_obs)
        # Robometer processes 224×224 internally; we pass ext as-is

        if t % query_freq == 0:
            image_buffer.append(ext.copy())
            action_list.append(t)

        # Random joint action in Franka workspace
        # First 7: joint positions near reset pose; last 1: gripper open (1.0)
        from examples.envs.mani_skill_client import DROID_RESET_QPOS
        noise = np.random.uniform(-0.05, 0.05, 7).astype(np.float32)
        action = np.concatenate([DROID_RESET_QPOS[:7] + noise, [1.0]]).astype(np.float32)

        obs, _, terminated, truncated, info = env.step(action)
        env_steps = t + 1

        if info.get("workspace_violated", False):
            failure_reason = "workspace_" + info.get("phase", "unknown")
            break
        if info["success"]:
            is_success     = True
            failure_reason = ""
            break
        if terminated or truncated:
            break

    query_steps = len(image_buffer)
    print(f"\nRollout done  —  env_steps={env_steps}, query_steps={query_steps}")
    print(f"  result: {'SUCCESS' if is_success else 'FAIL (' + failure_reason + ')'}")

    if query_steps == 0:
        print("No query steps recorded — rollout terminated at step 0"); return

    # ── Robometer progress calibration ────────────────────────────────────────
    frames_arr = np.stack(image_buffer, axis=0)       # (T, H, W, 3) uint8
    print(f"\nCalling Robometer on {frames_arr.shape} frames …")
    t0 = time.time()
    progress_scores, success_probs = robometer_client.compute_progress(
        frames_arr, args.instruction
    )
    elapsed = time.time() - t0
    print(f"Robometer inference: {elapsed:.2f}s")

    scale = args.progress_reward_scale
    progress_reward = scale * progress_scores.astype(np.float32)

    # Binary sparse reward (same formula as training)
    if is_success:
        binary_rewards = np.concatenate(
            [-np.ones(query_steps - 1, dtype=np.float32), [0.0]]
        )
    else:
        binary_rewards = -np.ones(query_steps, dtype=np.float32)
    dense_rewards = binary_rewards + progress_reward

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"{'step':>4}  {'binary':>7}  {'progress':>9}  {'dense':>7}  env_t")
    print(f"{'─'*68}")
    for i, (br, pr, dr, et) in enumerate(
        zip(binary_rewards, progress_scores, dense_rewards, action_list)
    ):
        print(f"  {i:>3}  {br:>7.3f}  {pr:>9.4f}  {dr:>7.4f}  t={et}")
    print(f"{'─'*68}")
    print(f"       progress  min={progress_scores.min():.4f}  "
          f"max={progress_scores.max():.4f}  mean={progress_scores.mean():.4f}")
    print(f"{'='*68}\n")

    # ── Save video ────────────────────────────────────────────────────────────
    video_path = str(out_dir / "reward_frames.mp4")
    print(f"Saving reward-frame video ({query_steps} frames @ {args.video_fps} fps) …")
    _save_video(image_buffer, video_path, fps=args.video_fps)
    print(f"  → {video_path}")

    # ── Save visualization ────────────────────────────────────────────────────
    result_tag = "SUCCESS" if is_success else f"FAIL_{failure_reason}"
    title = (
        f"Dense Reward Rollout  —  {result_tag}\n"
        f"env_steps={env_steps}  query_steps={query_steps}  "
        f"progress_mean={progress_scores.mean():.3f}  "
        f"inference={elapsed:.2f}s"
    )
    png_path = str(out_dir / "reward_progress_viz.png")
    print("Rendering visualization …")
    _visualize(
        frames=[fr for fr in image_buffer],
        progress=progress_scores,
        binary_rewards=binary_rewards,
        dense_rewards=dense_rewards,
        title=title,
        output_path=png_path,
        max_display=args.max_display_frames,
        dpi=args.dpi,
    )

    print(f"\nAll outputs in:  {out_dir}")
    print("  reward_frames.mp4        — exterior cam, one frame per query step")
    print("  reward_progress_viz.png  — frames (top) + progress chart (bottom)")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    robometer_client.close()
    env.close()


if __name__ == "__main__":
    main()
