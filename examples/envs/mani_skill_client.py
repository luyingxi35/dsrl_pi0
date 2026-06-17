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
import struct
import subprocess

import gymnasium as gym
import numpy as np
from gym.spaces import Box, Dict

_SERVER_SCRIPT = str(pathlib.Path(__file__).parent / "mani_skill_server.py")
_CAM_SHAPE     = (224, 224, 3)


class ManiSkillRemoteEnv(gym.Env):
    """gym.Env wrapper around a ManiSkill subprocess server.

    Args:
        robofac_python: Path to the robofac conda Python interpreter.
        server_script:  Path to mani_skill_server.py (auto-detected by default).
    """

    def __init__(
        self,
        robofac_python: str = "/home/gpu4/miniconda3/envs/robofac/bin/python3",
        server_script:  str = _SERVER_SCRIPT,
    ):
        self._proc = subprocess.Popen(
            [robofac_python, server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        self.observation_space = Dict({
            "qpos":  Box(-np.inf, np.inf, (9,),       dtype=np.float32),
            "ext":   Box(0, 255,          _CAM_SHAPE,  dtype=np.uint8),
            "wrist": Box(0, 255,          _CAM_SHAPE,  dtype=np.uint8),
        })
        self.action_space = Box(-1.0, 1.0, (8,), dtype=np.float32)   # panda_wristcam: 7 arm joints + 1 gripper

    # ── gym interface ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self._send({"cmd": "reset"})
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
        pass   # env randomisation handled inside the subprocess

    def close(self):
        try:
            self._send({"cmd": "close"})
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
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
