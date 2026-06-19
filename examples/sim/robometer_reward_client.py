"""Robometer reward client — spawns robometer_reward_server.py as subprocess.

Mirrors the ManiSkillRemoteEnv pattern: heavy inference is isolated in a
separate process (robometer conda env, GPU PyTorch) and communicates with the
JAX training process via length-prefixed pickle over stdin/stdout.
"""
from __future__ import annotations

import os
import pathlib
import pickle
import struct
import subprocess
import sys
import time
from typing import Tuple

import numpy as np

_SERVER_SCRIPT = str(pathlib.Path(__file__).parent / "robometer_reward_server.py")

# Default paths (can be overridden via constructor args)
_DEFAULT_ROBOMETER_PYTHON   = "/opt/yingxi/envs/robometer/bin/python3"
_DEFAULT_CHECKPOINT_PATH    = "/opt/yingxi/checkpoint-400"
_DEFAULT_BASE_MODEL_ID      = "/opt/caoyuhang/Pretrained_models/Qwen3-VL-4B-Instruct"
_ROBOMETER_CONFIG_YAML_PATH = (
    "/home/gpu4/yingxi/RoboFPE/robometer/robometer/configs/config.yaml"
)


class RobometerRewardClient:
    """Client that wraps a long-running Robometer server subprocess.

    The server process loads the Qwen3-VL-4B model once at startup (~30 s) and
    then serves ``compute_progress()`` calls for the lifetime of training.

    Args:
        robometer_python:   Path to the robometer conda Python interpreter.
        checkpoint_path:    Path to fine-tuned Robometer checkpoint directory.
        base_model_id:      Path to Qwen3-VL-4B-Instruct base model (for tokenizer).
    """

    def __init__(
        self,
        robometer_python:  str = _DEFAULT_ROBOMETER_PYTHON,
        checkpoint_path:   str = _DEFAULT_CHECKPOINT_PATH,
        base_model_id:     str = _DEFAULT_BASE_MODEL_ID,
        gpu: str | None = None,
    ) -> None:
        server_env = os.environ.copy()
        server_env["ROBOMETER_CONFIG_YAML_PATH"] = _ROBOMETER_CONFIG_YAML_PATH
        # If gpu is explicitly specified, override CUDA_VISIBLE_DEVICES for the server.
        # Otherwise inherit from the parent process (set by run_sim_dino_dense.sh).
        if gpu is not None:
            server_env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        cmd = [
            robometer_python, _SERVER_SCRIPT,
            "--checkpoint_path", checkpoint_path,
            "--base_model_id",   base_model_id,
        ]
        print(f"[RobometerRewardClient] launching server: {' '.join(cmd)}", flush=True)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,   # pass server stderr through to our stderr
            env=server_env,
        )
        self._closed = False

        # Wait for ready signal (model loading takes ~30 s).
        print("[RobometerRewardClient] waiting for model to load …", flush=True)
        t0 = time.time()
        resp = self._recv()
        if not resp.get("ready"):
            raise RuntimeError(f"Robometer server failed to start: {resp}")
        print(f"[RobometerRewardClient] ready in {time.time()-t0:.1f}s", flush=True)

    # ── Pipe helpers ─────────────────────────────────────────────────────────

    def _send(self, data: dict) -> None:
        raw = pickle.dumps(data, protocol=4)
        self._proc.stdin.write(struct.pack(">I", len(raw)))
        self._proc.stdin.write(raw)
        self._proc.stdin.flush()

    def _recv(self) -> dict:
        header = self._proc.stdout.read(4)
        if len(header) < 4:
            raise RuntimeError("Robometer server closed unexpectedly")
        n = struct.unpack(">I", header)[0]
        return pickle.loads(self._proc.stdout.read(n))

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_progress(
        self,
        frames: np.ndarray,
        task: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run progress estimation on a sequence of frames.

        Args:
            frames: uint8 array of shape (T, H, W, 3) — exterior camera frames,
                    one per query step.
            task:   Natural-language task description.

        Returns:
            progress_array:  float32 (T,), per-frame progress ∈ [0, 1].
            success_probs:   float32 (T,), per-frame success probability (may be empty).
        """
        if self._closed:
            raise RuntimeError("RobometerRewardClient has been closed")
        self._send({"cmd": "infer", "frames": frames, "task": task})
        resp = self._recv()
        if not resp["ok"]:
            raise RuntimeError(f"Robometer inference error: {resp.get('error')}")
        return (
            np.asarray(resp["progress"],      dtype=np.float32),
            np.asarray(resp["success_probs"], dtype=np.float32),
        )

    def close(self) -> None:
        """Gracefully shut down the server subprocess."""
        if self._closed:
            return
        self._closed = True
        try:
            self._send({"cmd": "close"})
        except Exception:
            pass
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
