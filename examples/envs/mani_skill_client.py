"""ManiSkill remote environment client (runs in dsrl_pi0 conda env).

Spawns mani_skill_server.py as a subprocess under the robofac conda env,
then wraps it as a standard gym.Env.  Mirrors the real-robot PolicyService
pattern: the heavy ManiSkill/sapien dependencies stay isolated in a separate
Python process and communicate via length-prefixed pickle over stdin/stdout.

Observation space (matches server _extract_obs):
    qpos  : (9,)       float32 — 7 arm joints + 2 gripper fingers
    ext   : (128,128,3) uint8  — base_camera  (exterior)
    wrist : (128,128,3) uint8  — hand_camera  (wrist, panda_wristcam)

Action space:
    (8,) float32 in [-1, 1] — [arm_7d_vel, gripper] (panda_wristcam: 1 gripper cmd)
"""

import pathlib
import pickle
import os
import signal
import struct
import subprocess

import gymnasium as gym
import numpy as np
from gym.spaces import Box, Dict

_SERVER_SCRIPT = str(pathlib.Path(__file__).parent / "mani_skill_server.py")
_CAM_SHAPE     = (224, 224, 3)
PEG_VERTICAL_QUAT = [0.70710678, 0.0, 0.70710678, 0.0]
FIXED_HOLE_POSE = {
    "p": [-0.04, 0.02, 0.04],
    "q": PEG_VERTICAL_QUAT,
}
DROID_PEG_RADIUS_RANGE = (0.12, 0.18)
DROID_RESET_QPOS = np.array(
    [
        0.0,
        -np.pi / 5,
        0.0,
        -4 * np.pi / 5,
        0.0,
        3 * np.pi / 5,
        0.0,
        0.04,
        0.04,
    ],
    dtype=np.float32,
)


class ManiSkillRemoteEnv(gym.Env):
    """gym.Env wrapper around a ManiSkill subprocess server.

    Args:
        robofac_python: Path to the robofac conda Python interpreter.
        server_script:  Path to mani_skill_server.py (auto-detected by default).
    """

    def __init__(
        self,
        robofac_python: str = "/opt/yingxi/envs/robofac/bin/python3",
        server_script:  str = _SERVER_SCRIPT,
        reset_qpos: np.ndarray | None = DROID_RESET_QPOS,
        fixed_hole_pose: dict | None = FIXED_HOLE_POSE,
        randomize_peg_pose: bool = True,
        peg_radius_range: tuple[float, float] = DROID_PEG_RADIUS_RANGE,
    ):
        self._reset_options = {}
        if reset_qpos is not None:
            self._reset_options["robot_qpos"] = np.asarray(reset_qpos, dtype=np.float32).tolist()
        if fixed_hole_pose is not None:
            self._reset_options["hole_pose"] = fixed_hole_pose
        self._randomize_peg_pose = randomize_peg_pose
        self._peg_radius_range = tuple(float(v) for v in peg_radius_range)
        if len(self._peg_radius_range) != 2 or self._peg_radius_range[0] < 0:
            raise ValueError("peg_radius_range must be a non-negative (min, max) pair.")
        if self._peg_radius_range[1] < self._peg_radius_range[0]:
            raise ValueError("peg_radius_range max must be >= min.")
        self._rng = np.random.default_rng()

        server_env = os.environ.copy()
        # The server renders through lavapipe/sapien_cpu; hiding CUDA prevents
        # torch/ManiSkill from initializing every GPU in the robofac process.
        server_env["CUDA_VISIBLE_DEVICES"] = ""
        server_env["NVIDIA_VISIBLE_DEVICES"] = ""
        self._proc = subprocess.Popen(
            [robofac_python, server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            start_new_session=True,
            env=server_env,
        )

        self.observation_space = Dict({
            "qpos":  Box(-np.inf, np.inf, (9,),       dtype=np.float32),
            "ext":   Box(0, 255,          _CAM_SHAPE,  dtype=np.uint8),
            "wrist": Box(0, 255,          _CAM_SHAPE,  dtype=np.uint8),
        })
        self.action_space = Box(-1.0, 1.0, (8,), dtype=np.float32)   # panda_wristcam: 7 arm joints + 1 gripper

    # ── gym interface ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        reset_options = self._build_reset_options(seed, options)
        self._send({"cmd": "reset", "seed": seed, "options": reset_options})
        r = self._recv()
        obs = {"qpos": r["qpos"], "ext": r["ext"], "wrist": r["wrist"]}
        return obs, {}

    def step(self, action):
        self._send({"cmd": "step", "action": np.asarray(action).tolist()})
        r = self._recv()
        obs  = {"qpos": r["qpos"], "ext": r["ext"], "wrist": r["wrist"]}
        info = {"success": r["success"]}
        return obs, r["reward"], r["done"], False, info

    def seed(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _build_reset_options(self, seed=None, options=None):
        reset_options = dict(self._reset_options)
        if options:
            reset_options.update(options)
        if self._randomize_peg_pose and reset_options.get("peg_pose") is None:
            rng = np.random.default_rng(seed) if seed is not None else self._rng
            hole_pose = reset_options.get("hole_pose", FIXED_HOLE_POSE)
            hole_xy = np.asarray(hole_pose["p"][:2], dtype=np.float32)
            theta = rng.uniform(-np.pi, np.pi)
            radius = rng.uniform(*self._peg_radius_range)
            peg_xy = hole_xy + radius * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
            reset_options["peg_pose"] = {
                "p": [float(peg_xy[0]), float(peg_xy[1]), 0.105],
                "q": PEG_VERTICAL_QUAT,
            }
        return reset_options

    def close(self):
        if self._proc.poll() is not None:
            return
        try:
            self._send({"cmd": "close"})
            self._proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(self._proc.pid, signal.SIGTERM)
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    os.killpg(self._proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._proc.wait(timeout=3)
        finally:
            try:
                self._proc.stdin.close()
                self._proc.stdout.close()
            except Exception:
                pass

    # ── internal I/O ──────────────────────────────────────────────────────────

    def _send(self, obj: object) -> None:
        data = pickle.dumps(obj, protocol=4)
        self._proc.stdin.write(struct.pack(">I", len(data)) + data)
        self._proc.stdin.flush()

    def _recv(self) -> object:
        raw = self._proc.stdout.read(4)
        if not raw:
            raise RuntimeError("ManiSkill server process ended unexpectedly.")
        n = struct.unpack(">I", raw)[0]
        return pickle.loads(self._proc.stdout.read(n))
