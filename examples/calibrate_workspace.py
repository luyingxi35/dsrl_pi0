#!/usr/bin/env python3
"""Offline workspace calibration for PegInsertionVertical-v1.

Uniformly samples robot joint configurations, checks whether the peg (pre-grasp)
and hole (post-grasp) are visible in the wrist camera (hand_camera) via camera
projection, and produces per-joint min/max bounding boxes.

Run with robofac env (direct ManiSkill access):
    /opt/yingxi/envs/robofac/bin/python3 examples/calibrate_workspace.py \\
        --n_qpos 3000 \\
        --n_peg_per_qpos 15 \\
        --slack 0.10 \\
        --output ./workspace_bounds.json

Outputs:
    workspace_bounds.json          — bounds (calibrated + effective)
    workspace_bounds_vis/          — 8 PNG renders at bounding-box extremes
"""
import argparse
import json
import os
import sys
import importlib.util
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# Lavapipe / sapien patches  (identical to mani_skill_server.py, must precede
# all sapien / mani_skill imports)
# ══════════════════════════════════════════════════════════════════════════════
_LAVAPIPE_ICD = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
_OIDN_LIB = (
    "/opt/yingxi/envs/robofac/lib/python3.10/site-packages/sapien/oidn_library"
)
os.environ["VK_ICD_FILENAMES"] = _LAVAPIPE_ICD
os.environ["LP_NUM_THREADS"]   = os.environ.get("LP_NUM_THREADS",  "4")
os.environ["OMP_NUM_THREADS"]  = os.environ.get("OMP_NUM_THREADS", "1")
os.environ["MKL_NUM_THREADS"]  = os.environ.get("MKL_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"]   = ""
os.environ["NVIDIA_VISIBLE_DEVICES"] = ""
os.environ["LD_LIBRARY_PATH"] = (
    _OIDN_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")
)

import sapien  # noqa: E402
_orig_RS = sapien.render.RenderSystem
sapien.render.RenderSystem = lambda device=None: _orig_RS()

import mani_skill.envs.utils.system.backend as _be   # noqa: E402
import mani_skill.envs.sapien_env as _se              # noqa: E402
_orig_parse = _be.parse_sim_and_render_backend
def _patched_parse(sim_backend, render_backend):
    r = _orig_parse(sim_backend, render_backend)
    r.render_device  = sapien.Device("cpu")
    r.render_backend = "sapien_cpu"
    return r
_be.parse_sim_and_render_backend = _patched_parse
_se.parse_sim_and_render_backend = _patched_parse

# ── Load task ─────────────────────────────────────────────────────────────────
_TASK_FILE = "/home/gpu4/yingxi/RoboFPE/mani_envs/tasks/task_PegInsertionVertical.py"
_spec = importlib.util.spec_from_file_location("peg_task", _TASK_FILE)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["peg_task"] = _mod
_spec.loader.exec_module(_mod)

import gymnasium as gym   # noqa: E402
import numpy as np        # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
# Physical joint limits (from qlimits probe; 7 arm joints only)
PHYS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], np.float32)
PHYS_UPPER = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], np.float32)
PHYS_MARGIN = 0.12   # keep slightly inside limits (same as robot_qpos_limit_margin in task)

SAMPLE_LOWER = PHYS_LOWER + PHYS_MARGIN
SAMPLE_UPPER = PHYS_UPPER - PHYS_MARGIN

# Fixed hole and peg vertical quaternion (from mani_skill_client.py)
PEG_VERTICAL_QUAT = [0.70710678, 0.0, 0.70710678, 0.0]
FIXED_HOLE_POSE   = {"p": [-0.04, 0.02, 0.04], "q": PEG_VERTICAL_QUAT}
HOLE_POS          = np.array([-0.04, 0.02, 0.04], dtype=np.float32)  # center of hole top
HOLE_XY           = HOLE_POS[:2]

PEG_RADIUS_RANGE  = (0.12, 0.18)   # from mani_skill_client.py DROID_PEG_RADIUS_RANGE
PEG_TABLE_Z       = 0.105          # peg center height when resting on table (peg_half_length)

# A fixed reference peg start pose for the calibration resets
# (peg position doesn't affect camera params; this just avoids env validation errors)
_REF_PEG_POSE = {
    "p": [HOLE_POS[0] + 0.15, float(HOLE_POS[1]), PEG_TABLE_Z],
    "q": PEG_VERTICAL_QUAT,
}

IMG_W = IMG_H = 224


# ══════════════════════════════════════════════════════════════════════════════
# Camera projection
# ══════════════════════════════════════════════════════════════════════════════

def project_check(world_pos: np.ndarray, ext: np.ndarray, intr: np.ndarray,
                  W: int = IMG_W, H: int = IMG_H, margin: int = 0) -> bool:
    """Return True if world_pos projects inside the camera image.

    Args:
        world_pos: (3,) point in world frame.
        ext:  (3, 4) extrinsic_cv  [R | t]  (world→camera, OpenCV convention).
        intr: (3, 3) intrinsic_cv.
        margin: pixel border to exclude (0 = full FOV).
    """
    p_hom = np.append(world_pos, 1.0)   # (4,)
    p_cam = ext @ p_hom                  # (3,)
    if p_cam[2] <= 0:                    # behind camera
        return False
    p_img = intr @ p_cam                 # (3,) homogeneous
    u = float(p_img[0] / p_img[2])
    v = float(p_img[1] / p_img[2])
    return margin <= u <= W - margin and margin <= v <= H - margin


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_np(x) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _get_cam_params(obs: dict, cam: str = "hand_camera"):
    """Return (ext (3,4), intr (3,3)) from obs['sensor_param']."""
    sp = obs["sensor_param"][cam]
    ext  = _to_np(sp["extrinsic_cv"]).reshape(3, 4).astype(np.float32)
    intr = _to_np(sp["intrinsic_cv"]).reshape(3, 3).astype(np.float32)
    return ext, intr


def _save_image(arr: np.ndarray, path: str) -> None:
    """Save uint8 RGB array as PNG.  Falls back to raw npy if PIL not available."""
    arr = np.asarray(arr, dtype=np.uint8)
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except ImportError:
        np.save(path.replace(".png", ".npy"), arr)
        print(f"  [PIL not found] saved raw npy: {path.replace('.png', '.npy')}")


# ══════════════════════════════════════════════════════════════════════════════
# Main calibration
# ══════════════════════════════════════════════════════════════════════════════

def calibrate(args) -> None:
    rng = np.random.default_rng(args.seed)

    # ── create env ────────────────────────────────────────────────────────────
    env = gym.make(
        "PegInsertionVertical-v1",
        obs_mode="rgb+state",
        render_mode="rgb_array",
        num_envs=1,
        robot_uids="panda_wristcam",
        sensor_configs=dict(width=IMG_W, height=IMG_H),
        max_episode_steps=600,
    )

    peg_visible_qpos:  list[np.ndarray] = []
    hole_visible_qpos: list[np.ndarray] = []
    n_crash = 0

    print(f"Sampling {args.n_qpos} configurations "
          f"({args.n_peg_per_qpos} peg positions each)…")

    for i in range(args.n_qpos):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{args.n_qpos}  "
                  f"peg_vis={len(peg_visible_qpos)}  "
                  f"hole_vis={len(hole_visible_qpos)}  "
                  f"crashes={n_crash}")

        # Sample 7-joint arm configuration uniformly within safe limits
        qpos_7 = rng.uniform(SAMPLE_LOWER, SAMPLE_UPPER).astype(np.float32)
        qpos_9 = np.concatenate([qpos_7, [0.04, 0.04]])   # open gripper

        try:
            obs, _ = env.reset(
                seed=int(rng.integers(0, 2**31)),
                options={
                    "robot_qpos": qpos_9.tolist(),
                    "hole_pose":  FIXED_HOLE_POSE,
                    "peg_pose":   _REF_PEG_POSE,
                },
            )
        except Exception as exc:
            n_crash += 1
            if n_crash <= 5:
                print(f"  [WARN] env.reset crashed at sample {i}: {exc}")
            continue

        # Camera params depend only on qpos (robot pose), not peg placement
        ext, intr = _get_cam_params(obs, "hand_camera")

        # ── hole visibility (fixed position, single check) ────────────────────
        if project_check(HOLE_POS, ext, intr):
            hole_visible_qpos.append(qpos_7.copy())

        # ── peg visibility (multiple random positions) ─────────────────────────
        vis_count = 0
        for _ in range(args.n_peg_per_qpos):
            theta  = rng.uniform(-np.pi, np.pi)
            radius = rng.uniform(*PEG_RADIUS_RANGE)
            peg_xy = HOLE_XY + radius * np.array([np.cos(theta), np.sin(theta)], np.float32)
            peg_pos = np.array([peg_xy[0], peg_xy[1], PEG_TABLE_Z], np.float32)
            if project_check(peg_pos, ext, intr):
                vis_count += 1
        if vis_count >= args.n_peg_per_qpos // 2:   # ≥50% peg positions visible
            peg_visible_qpos.append(qpos_7.copy())

    env_original = env  # keep for visualisation below

    if not peg_visible_qpos:
        raise RuntimeError("No peg-visible configurations found. Check task / hole pose.")
    if not hole_visible_qpos:
        raise RuntimeError("No hole-visible configurations found. Check task / hole pose.")

    peg_arr  = np.stack(peg_visible_qpos,  axis=0)   # (N_peg, 7)
    hole_arr = np.stack(hole_visible_qpos, axis=0)   # (N_hole, 7)

    # ── calibrated bounds (tight) ─────────────────────────────────────────────
    cal_peg_lower  = peg_arr.min(axis=0)
    cal_peg_upper  = peg_arr.max(axis=0)
    cal_hole_lower = hole_arr.min(axis=0)
    cal_hole_upper = hole_arr.max(axis=0)

    # ── effective bounds (calibrated + slack, clamped to physical limits) ──────
    slack = float(args.slack)
    eff_peg_lower  = np.maximum(cal_peg_lower  - slack, SAMPLE_LOWER)
    eff_peg_upper  = np.minimum(cal_peg_upper  + slack, SAMPLE_UPPER)
    eff_hole_lower = np.maximum(cal_hole_lower - slack, SAMPLE_LOWER)
    eff_hole_upper = np.minimum(cal_hole_upper + slack, SAMPLE_UPPER)

    # ── print summary ─────────────────────────────────────────────────────────
    print("\n─── Calibrated bounds (tight) ───")
    for j in range(7):
        w_peg  = cal_peg_upper[j]  - cal_peg_lower[j]
        w_hole = cal_hole_upper[j] - cal_hole_lower[j]
        print(f"  J{j}  peg  [{cal_peg_lower[j]:+.3f}, {cal_peg_upper[j]:+.3f}]  width={w_peg:.3f}")
        print(f"       hole [{cal_hole_lower[j]:+.3f}, {cal_hole_upper[j]:+.3f}]  width={w_hole:.3f}")
    print(f"\npeg-visible  configs: {len(peg_visible_qpos)} / {args.n_qpos}")
    print(f"hole-visible configs: {len(hole_visible_qpos)} / {args.n_qpos}")
    print(f"crashes:              {n_crash} / {args.n_qpos}")

    # ── save JSON ─────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "calibrated": {
            "peg_lower":  cal_peg_lower.tolist(),
            "peg_upper":  cal_peg_upper.tolist(),
            "hole_lower": cal_hole_lower.tolist(),
            "hole_upper": cal_hole_upper.tolist(),
        },
        "effective": {
            "peg_lower":  eff_peg_lower.tolist(),
            "peg_upper":  eff_peg_upper.tolist(),
            "hole_lower": eff_hole_lower.tolist(),
            "hole_upper": eff_hole_upper.tolist(),
        },
        "slack_rad": slack,
        "meta": {
            "n_qpos_sampled":     args.n_qpos,
            "n_peg_per_qpos":     args.n_peg_per_qpos,
            "peg_visible_count":  len(peg_visible_qpos),
            "hole_visible_count": len(hole_visible_qpos),
            "crashes":            n_crash,
            "hole_pos":           HOLE_POS.tolist(),
            "peg_radius_range":   list(PEG_RADIUS_RANGE),
            "phys_lower":         PHYS_LOWER.tolist(),
            "phys_upper":         PHYS_UPPER.tolist(),
            "phys_margin":        PHYS_MARGIN,
        },
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved bounds → {output_path}")

    # ── visualise 4 extreme configurations (lower/upper corner per box) ───────
    vis_dir = output_path.parent / "workspace_bounds_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    render_configs = [
        ("peg",  "lower", cal_peg_lower),
        ("peg",  "upper", cal_peg_upper),
        ("hole", "lower", cal_hole_lower),
        ("hole", "upper", cal_hole_upper),
    ]
    print("\nRendering boundary configurations…")
    for label, direction, qpos_7 in render_configs:
        qpos_9 = np.concatenate([qpos_7, [0.04, 0.04]])
        try:
            obs, _ = env_original.reset(
                options={
                    "robot_qpos": qpos_9.tolist(),
                    "hole_pose":  FIXED_HOLE_POSE,
                    "peg_pose":   _REF_PEG_POSE,
                }
            )
            wrist_img = _to_np(obs["sensor_data"]["hand_camera"]["rgb"]).reshape(IMG_H, IMG_W, 3)
            ext_img   = _to_np(obs["sensor_data"]["base_camera"]["rgb"]).reshape(IMG_H, IMG_W, 3)
        except Exception as exc:
            print(f"  [WARN] render failed for {label}/{direction}: {exc}")
            continue

        wrist_path = str(vis_dir / f"vis_{label}_{direction}_wrist.png")
        ext_path   = str(vis_dir / f"vis_{label}_{direction}_ext.png")
        _save_image(wrist_img, wrist_path)
        _save_image(ext_img,   ext_path)
        print(f"  {label}/{direction}: wrist → {wrist_path}")
        print(f"  {label}/{direction}: ext   → {ext_path}")

    env_original.close()
    print("\nDone.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_qpos",         default=3000, type=int,
                   help="Number of joint configurations to sample (default 3000)")
    p.add_argument("--n_peg_per_qpos", default=15,   type=int,
                   help="Peg position randomizations per configuration (default 15)")
    p.add_argument("--slack",          default=0.10, type=float,
                   help="Extra rad added to each joint range in effective bounds (default 0.10)")
    p.add_argument("--output",         default="./workspace_bounds.json",
                   help="Path for output JSON file")
    p.add_argument("--seed",           default=42,   type=int)
    return p


if __name__ == "__main__":
    calibrate(_build_parser().parse_args())
