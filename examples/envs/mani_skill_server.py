#!/usr/bin/env python3
"""ManiSkill environment server — runs under robofac conda env.

Serves PegInsertionVertical-v1 (panda_wristcam) via length-prefixed pickle
protocol over stdin/stdout. Mirrors the real-robot DROID setup:
  base_camera  → exterior image (128×128)
  hand_camera  → wrist image   (128×128, mounted on gripper)

Protocol:
  Client → Server: {"cmd": "reset"} or {"cmd": "step", "action": list(9)} or {"cmd": "close"}
  Server → Client: {"ok": bool, "qpos": ndarray(9), "ext": ndarray(128,128,3),
                    "wrist": ndarray(128,128,3), [reward, done, success]}

Usage (from dsrl_pi0 training side):
    Spawned automatically by ManiSkillRemoteEnv in mani_skill_client.py.

Rendering note:
    H100 compute GPUs hang in sapien's GPU Vulkan renderer (host-specific driver
    issue, SAPIEN #290).  We follow the workaround from allenai/vla-evaluation-
    harness PR #51: redirect to Mesa lavapipe (CPU Vulkan) + force the ManiSkill
    sapien_cpu render backend.  Three monkey-patches are applied in strict order
    BEFORE any mani_skill / gymnasium imports:
      1. VK_ICD_FILENAMES  → lavapipe ICD
      2. sapien.render.RenderSystem wrapped to drop the device arg
      3. parse_sim_and_render_backend patched to return sapien_cpu backend
"""
import os
import sys
import struct
import pickle
import importlib.util

# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — env vars (must precede ALL sapien / mani_skill imports)
# ══════════════════════════════════════════════════════════════════════════════
_LAVAPIPE_ICD = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
_OIDN_LIB = (
    "/opt/yingxi/envs/robofac/lib/python3.10/site-packages/sapien/oidn_library"
)

os.environ["VK_ICD_FILENAMES"]  = _LAVAPIPE_ICD
os.environ["LP_NUM_THREADS"]    = os.environ.get("LP_NUM_THREADS",  "4")
os.environ["OMP_NUM_THREADS"]   = os.environ.get("OMP_NUM_THREADS", "1")
os.environ["MKL_NUM_THREADS"]   = os.environ.get("MKL_NUM_THREADS", "1")
# prepend OIDN library dir so nightly sapien finds libOpenImageDenoise.so.2
os.environ["LD_LIBRARY_PATH"] = (
    _OIDN_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")
)

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — import sapien, then wrap RenderSystem
#           lavapipe has no CUDA backend → drop the device positional arg
# ══════════════════════════════════════════════════════════════════════════════
import sapien  # noqa: E402

_orig_RenderSystem = sapien.render.RenderSystem

def _lavapipe_RenderSystem(device=None):  # noqa: N802
    """Wrap RenderSystem to ignore device arg when using lavapipe."""
    return _orig_RenderSystem()

sapien.render.RenderSystem = _lavapipe_RenderSystem

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — patch parse_sim_and_render_backend in both import locations
#           forces render_device → sapien.Device("cpu"), render_backend → sapien_cpu
# ══════════════════════════════════════════════════════════════════════════════
import mani_skill.envs.utils.system.backend as _be   # noqa: E402
import mani_skill.envs.sapien_env as _se              # noqa: E402

_orig_parse = _be.parse_sim_and_render_backend

def _patched_parse(sim_backend, render_backend):
    result = _orig_parse(sim_backend, render_backend)
    result.render_device  = sapien.Device("cpu")
    result.render_backend = "sapien_cpu"
    return result

_be.parse_sim_and_render_backend = _patched_parse
_se.parse_sim_and_render_backend = _patched_parse   # update already-imported ref

# ══════════════════════════════════════════════════════════════════════════════
# Now safe to load the task and import gymnasium
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np  # noqa: E402

# ── Load the task directly, bypassing mani_envs/tasks/__init__.py ─────────────
# This avoids the noTableSceneBuilder / other mani_skill version issues that
# occur when importing the entire mani_envs.tasks package.
_TASK_FILE = "/home/gpu4/yingxi/RoboFPE/mani_envs/tasks/task_PegInsertionVertical.py"
_spec = importlib.util.spec_from_file_location("peg_task", _TASK_FILE)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["peg_task"] = _mod
_spec.loader.exec_module(_mod)   # registers PegInsertionVertical-v1 with gymnasium

import gymnasium as gym   # noqa: E402 – must come after env registration


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _send(obj: object) -> None:
    """Pickle and length-prefix an object to stdout."""
    data = pickle.dumps(obj, protocol=4)
    sys.stdout.buffer.write(struct.pack(">I", len(data)) + data)
    sys.stdout.buffer.flush()


def _recv() -> object:
    """Read a length-prefixed pickle object from stdin."""
    raw = sys.stdin.buffer.read(4)
    if not raw:
        sys.exit(0)   # stdin closed → parent process died
    n = struct.unpack(">I", raw)[0]
    return pickle.loads(sys.stdin.buffer.read(n))


def _to_numpy(x):
    """Convert torch tensor or other array-like to numpy."""
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _extract_obs(obs: dict):
    """Extract qpos, exterior RGB, and wrist RGB from a ManiSkill obs dict.

    With sapien_cpu (lavapipe) backend, obs has keys: sensor_param, sensor_data, state.
    qpos is the first 9 elements of obs['state'] (shape [1, 43]):
      state[:, :9]  = qpos  (7 arm joints + 2 gripper fingers, matches qpos proprioception)
      state[:, 9:]  = qvel + extra

    Returns:
        qpos  (9,)   float32 — 7 arm joints + 2 gripper fingers
        ext   (H,W,3) uint8  — base_camera (exterior, 128×128)
        wrist (H,W,3) uint8  — hand_camera (wrist, 128×128)
    """
    # sapien_cpu backend: qpos is in obs['state'][:, :9], no obs['agent'] key
    if "agent" in obs:
        qpos = _to_numpy(obs["agent"]["qpos"])[0].astype(np.float32)      # (9,) GPU backend
    else:
        qpos = _to_numpy(obs["state"])[0, :9].astype(np.float32)             # (9,) CPU backend
    ext   = _to_numpy(obs["sensor_data"]["base_camera"]["rgb"])[0].astype(np.uint8)   # (128,128,3)
    wrist = _to_numpy(obs["sensor_data"]["hand_camera"]["rgb"])[0].astype(np.uint8)   # (128,128,3)
    return qpos, ext, wrist


# ── Environment ───────────────────────────────────────────────────────────────

# panda_wristcam adds hand_camera sensor; SUPPORTED_ROBOTS only warns, not raises.
env = gym.make(
    "PegInsertionVertical-v1",
    obs_mode="rgb+state",
    render_mode="rgb_array",
    num_envs=1,
    robot_uids="panda_wristcam",
    sensor_configs=dict(width=224, height=224),
)


# ── Main server loop ──────────────────────────────────────────────────────────

while True:
    msg = _recv()
    cmd = msg.get("cmd")

    if cmd == "reset":
        obs, _ = env.reset()
        qpos, ext, wrist = _extract_obs(obs)
        _send({"ok": True, "qpos": qpos, "ext": ext, "wrist": wrist})

    elif cmd == "step":
        action = np.asarray(msg["action"], dtype=np.float32).reshape(1, -1)  # (1,8): batched for ManiSkill
        obs, reward, terminated, truncated, info = env.step(action)
        qpos, ext, wrist = _extract_obs(obs)

        # Scalar extraction (ManiSkill may return tensors)
        r_val = reward.item() if hasattr(reward, "item") else float(reward)
        done  = bool(terminated or truncated)
        succ  = bool(info["success"].item() if hasattr(info["success"], "item")
                     else info["success"])

        _send({"ok":      True,
               "qpos":    qpos,
               "ext":     ext,
               "wrist":   wrist,
               "reward":  r_val,
               "done":    done,
               "success": succ})

    elif cmd == "close":
        env.close()
        sys.exit(0)

    else:
        _send({"ok": False, "error": f"Unknown command: {cmd}"})
