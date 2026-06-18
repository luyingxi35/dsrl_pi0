#!/usr/bin/env python3
"""Smoke test: verify Robometer inference on a full timeout rollout (~75 frames) does not OOM.

Usage:
  # Use existing eval videos as frame source (recommended, fastest):
  python3 examples/tests/test_smoke_robometer.py \\
      --video-dir logs/pi0_eval_sim_20260618_221150

  # Use all wrist videos from the directory (75-frame simulation):
  python3 examples/tests/test_smoke_robometer.py \\
      --video-dir logs/pi0_eval_sim_20260618_221150 --max-frames 75

  # Collect from live ManiSkill rollout instead:
  python3 examples/tests/test_smoke_robometer.py --rollout
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = str(Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from examples.robometer_reward_client import (
    RobometerRewardClient,
    _DEFAULT_BASE_MODEL_ID,
    _DEFAULT_CHECKPOINT_PATH,
    _DEFAULT_ROBOMETER_PYTHON,
)


# ── Frame loading helpers ─────────────────────────────────────────────────────

def _load_mp4_frames(mp4_path: str, target_h: int = 224, target_w: int = 224) -> np.ndarray:
    """Extract frames from an mp4 file and resize to (H, W, 3) uint8."""
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python required: pip install opencv-python-headless")
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(frame.astype(np.uint8))
    cap.release()
    return np.stack(frames, axis=0) if frames else np.zeros((0, target_h, target_w, 3), dtype=np.uint8)


def load_frames_from_video_dir(
    video_dir: str,
    max_frames: int = 75,
    pattern: str = "*_side.mp4",
) -> np.ndarray:
    """Load frames from all matching videos in a directory, up to max_frames."""
    mp4s = sorted(glob.glob(os.path.join(video_dir, pattern)))
    if not mp4s:
        # Fall back to any mp4
        mp4s = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not mp4s:
        raise FileNotFoundError(f"No mp4 files found in {video_dir}")

    frames_list = []
    for mp4 in mp4s:
        f = _load_mp4_frames(mp4)
        frames_list.append(f)
        if sum(len(x) for x in frames_list) >= max_frames:
            break

    all_frames = np.concatenate(frames_list, axis=0)[:max_frames]
    print(f"Loaded {len(all_frames)} frames from {len(mp4s)} video(s) in {video_dir}")
    return all_frames


def collect_rollout_frames(max_frames: int = 75) -> np.ndarray:
    """Run one ManiSkill rollout and collect exterior frames."""
    from examples.envs.mani_skill_client import ManiSkillRemoteEnv
    env = ManiSkillRemoteEnv()
    obs, _ = env.reset()
    frames = []
    for _ in range(max_frames):
        ext = obs["ext"]  # (224,224,3) uint8
        frames.append(ext.copy())
        action = np.zeros(8, dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()
    result = np.stack(frames, axis=0)
    print(f"Collected {len(result)} frames from live rollout")
    return result


# ── GPU memory helpers ────────────────────────────────────────────────────────

def _gpu_mem_mb() -> Optional[float]:
    """Return GPU memory allocated in MB, or None if CUDA not available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test: 75-frame Robometer inference OOM check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video-dir", default=None,
                        help="Directory with eval .mp4 files (preferred test source)")
    parser.add_argument("--rollout", action="store_true",
                        help="Collect frames from a live ManiSkill rollout instead")
    parser.add_argument("--max-frames", type=int, default=75,
                        help="Maximum frames to send to Robometer (default: 75 = full timeout)")
    parser.add_argument("--task",
                        default="pick up the peg and insert it vertically",
                        help="Task instruction for Robometer")
    parser.add_argument("--robometer-python",  default=_DEFAULT_ROBOMETER_PYTHON)
    parser.add_argument("--checkpoint-path",   default=_DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--base-model-id",     default=_DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--gpu",             default=None,
                        help="GPU id for Robometer server (default: inherit CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--video-pattern",     default="*_side.mp4",
                        help="Glob pattern for video files (default: *_side.mp4)")
    args = parser.parse_args()

    if args.video_dir is None and not args.rollout:
        parser.error("Specify --video-dir or --rollout")

    # ── Load frames ───────────────────────────────────────────────────────────
    if args.video_dir:
        frames = load_frames_from_video_dir(
            args.video_dir,
            max_frames=args.max_frames,
            pattern=args.video_pattern,
        )
    else:
        frames = collect_rollout_frames(args.max_frames)

    if frames.shape[0] == 0:
        print("ERROR: no frames loaded")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Smoke test: {frames.shape[0]} frames {frames.shape[1:]} -> Robometer")
    print(f"Task: {args.task}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Base model: {args.base_model_id}")
    print(f"{'='*60}\n")

    mem_before = _gpu_mem_mb()

    # ── Start client (loads model) ────────────────────────────────────────────
    client = RobometerRewardClient(
        robometer_python=args.robometer_python,
        checkpoint_path=args.checkpoint_path,
        base_model_id=args.base_model_id,
        gpu=args.gpu,
    )

    mem_after_load = _gpu_mem_mb()
    if mem_before is not None and mem_after_load is not None:
        print(f"GPU memory after model load: {mem_after_load:.0f} MB "
              f"(+{mem_after_load - mem_before:.0f} MB from subprocess)")

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f"\nRunning inference on {frames.shape[0]} frames …")
    t0 = time.time()
    try:
        progress, success_probs = client.compute_progress(frames, args.task)
    except Exception as exc:
        print(f"\nFAIL: inference error: {exc}")
        client.close()
        sys.exit(1)
    elapsed = time.time() - t0

    mem_after_infer = _gpu_mem_mb()

    # ── Report results ────────────────────────────────────────────────────────
    print(f"\nInference time: {elapsed:.2f}s")
    if mem_after_load is not None and mem_after_infer is not None:
        print(f"GPU memory during inference: {mem_after_infer:.0f} MB")
    print(f"\nProgress scores ({len(progress)} values):")
    for i, p in enumerate(progress):
        print(f"  frame {i+1:3d}/{frames.shape[0]}: progress={p:.4f}"
              + (f", success_prob={success_probs[i]:.4f}" if i < len(success_probs) else ""))

    # ── Assertions ────────────────────────────────────────────────────────────
    assert len(progress) == frames.shape[0], (
        f"FAIL: output length {len(progress)} != input length {frames.shape[0]}"
    )
    assert np.all((progress >= 0.0) & (progress <= 1.0)), (
        f"FAIL: progress values out of [0,1] range: {progress}"
    )

    print(f"\n{'='*60}")
    print("PASS: Robometer inference successful")
    print(f"  frames={frames.shape[0]}, output_len={len(progress)}")
    print(f"  progress min={progress.min():.4f} max={progress.max():.4f} mean={progress.mean():.4f}")
    print(f"  time={elapsed:.2f}s, no OOM")
    print(f"{'='*60}\n")

    client.close()


if __name__ == "__main__":
    main()
