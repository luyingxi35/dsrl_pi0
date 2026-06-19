#!/usr/bin/env python3
"""Standalone pi0-only evaluation in ManiSkill simulation using pd_joint_delta_pos.

Compared to evaluate_pi0.py (which uses pd_joint_pos + velocity integration),
this script feeds the pi0 velocity output DIRECTLY to ManiSkill as per-step
normalised joint deltas (pd_joint_delta_pos controller).

DROID convention: pi0 action[7] ∈ [0,1], >0.5 → open, <0.5 → close.
Semantics: action_scale=0.5 → normalised_delta = pi0_vel (identity mapping).
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import pickle
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.sim.envs.mani_skill_client import ManiSkillRemoteEnv
from examples.sim.action_utils import pi0_vel_to_delta_actions
from examples.real.utils.real_robot_common import (
    RolloutResult,
    append_result,
    resolve_outputdir,
    save_rollout_video,
)


DEFAULT_CHECKPOINT_PATH = "/opt/yingxi/pi0_droid"
DEFAULT_ROBOFAC_PYTHON = "/opt/yingxi/envs/robofac/bin/python3"

RESULT_FIELDS = [
    "episode_id",
    "success",
    "failure_reason",
    "env_steps",
    "duration_s",
    "video_path",
    "wrist_video_path",
    "timestamp",
    "instruction",
    "checkpoint_path",
    "execution_steps",
    "max_rollout_steps",
    "action_scale",
    "robofac_python",
]


def _extract_sim_obs(env_obs: dict):
    """Return (qpos, exterior RGB, wrist RGB) from ManiSkillRemoteEnv obs."""
    return (
        np.asarray(env_obs["qpos"], dtype=np.float32),
        np.asarray(env_obs["ext"], dtype=np.uint8),
        np.asarray(env_obs["wrist"], dtype=np.uint8),
    )


def _obs_to_pi0_input(
    qpos: np.ndarray,
    ext_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    instruction: str,
    use_exterior: bool = False,
) -> dict:
    """Build pi0 DroidInputs without importing JAX-heavy sim training utilities.

    Args:
        use_exterior: If True, pass exterior (base_camera) image to pi0.
                      Default False — only wrist camera, matching real eval default
                      (evaluate_pi0_real.py --use_exterior_camera=0).
                      DroidInputs automatically sets exterior to zeros + mask=False
                      when the key is absent.
    """
    from openpi_client import image_tools

    data = {
        "observation/wrist_image_left": image_tools.convert_to_uint8(wrist_rgb),
        "observation/joint_position": qpos[:7].astype(np.float32),
        "observation/gripper_position": np.array(
            [np.clip(np.mean(qpos[7:9]) / 0.04, 0.0, 1.0)],
            dtype=np.float32,
        ),
        "prompt": instruction,
    }
    if use_exterior:
        data["observation/exterior_image_1_left"] = image_tools.convert_to_uint8(ext_rgb)
    return data


def _send_msg(stream, obj: object) -> None:
    data = pickle.dumps(obj, protocol=4)
    stream.write(struct.pack(">I", len(data)) + data)
    stream.flush()


def _recv_msg(stream) -> object:
    raw = stream.read(4)
    if not raw:
        raise RuntimeError("Local pi0 policy worker exited unexpectedly.")
    n = struct.unpack(">I", raw)[0]
    return pickle.loads(stream.read(n))


def _load_pi0_policy_in_worker(checkpoint_path: str, config_name: str = "pi0_droid"):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"pi0 checkpoint path does not exist: {checkpoint_path}")

    from openpi.policies import policy_config as openpi_policy_config
    from openpi.training import config as openpi_config

    pi0_cfg = openpi_config.get_config(config_name)
    policy = openpi_policy_config.create_trained_policy(pi0_cfg, checkpoint_path)
    logging.info("Loaded %s from %s", config_name, checkpoint_path)
    return policy


def run_policy_worker(checkpoint_path: str, config_name: str = "pi0_droid") -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, force=True)
    policy = _load_pi0_policy_in_worker(checkpoint_path, config_name)
    while True:
        try:
            msg = _recv_msg(sys.stdin.buffer)
            cmd = msg.get("cmd")
            if cmd == "infer":
                response = policy.infer(msg["obs"], noise=msg.get("noise"))
                _send_msg(sys.stdout.buffer, {"ok": True, "response": response})
            elif cmd == "close":
                _send_msg(sys.stdout.buffer, {"ok": True})
                return
            else:
                raise ValueError(f"Unknown policy worker command: {cmd}")
        except Exception:
            _send_msg(
                sys.stdout.buffer,
                {"ok": False, "error": traceback.format_exc()},
            )


class LocalPi0Policy:
    """Small subprocess wrapper around local OpenPI to isolate JAX/CUDA runtime."""

    def __init__(self, checkpoint_path: str, config_name: str = "pi0_droid"):
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"pi0 checkpoint path does not exist: {checkpoint_path}")
        self._config_name = config_name
        self._proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--_policy_worker", checkpoint_path, self._config_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            start_new_session=True,
        )

    def infer(self, obs, noise=None):
        _send_msg(self._proc.stdin, {"cmd": "infer", "obs": obs, "noise": noise})
        try:
            result = _recv_msg(self._proc.stdout)
        except RuntimeError as exc:
            code = self._proc.poll()
            raise RuntimeError(
                f"Local pi0 policy worker exited unexpectedly "
                f"(returncode={code})."
            ) from exc
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Local pi0 policy worker failed."))
        return result["response"]

    def close(self):
        if self._proc.poll() is not None:
            return
        try:
            _send_msg(self._proc.stdin, {"cmd": "close"})
            _recv_msg(self._proc.stdout)
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


def load_pi0_policy(checkpoint_path: str, config_name: str = "pi0_droid"):
    policy = LocalPi0Policy(checkpoint_path, config_name)
    logging.info("Started local %s worker for %s", config_name, checkpoint_path)
    return policy


def run_rollout(
    args: argparse.Namespace,
    env: ManiSkillRemoteEnv,
    agent_dp: Any,
    episode_id: int,
    outputdir: Path,
) -> tuple[RolloutResult, str]:
    from tqdm import tqdm

    env_obs, _ = env.reset()
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    start_time = time.time()
    side_image_list: list[np.ndarray] = []
    wrist_image_list: list[np.ndarray] = []
    success = False
    failure_reason = "timeout"
    env_steps = 0

    pbar = tqdm(total=args.max_rollout_steps, desc=f"pi0 sim episode {episode_id}", unit="step")
    try:
        while env_steps < args.max_rollout_steps:
            # ── Observe + infer ───────────────────────────────────────────────
            qpos, ext_rgb, wrist_rgb = _extract_sim_obs(env_obs)
            side_image_list.append(ext_rgb)
            wrist_image_list.append(wrist_rgb)

            pi0_obs = _obs_to_pi0_input(qpos, ext_rgb, wrist_rgb, args.instruction,
                                             use_exterior=bool(args.use_exterior_camera))
            actions = np.asarray(agent_dp.infer(pi0_obs)["actions"])
            # Debug: log raw gripper values from the action chunk.
            # Run with LOG_LEVEL=DEBUG to see: helps verify gripper convention.
            gripper_chunk = actions[:args.execution_steps, 7]
            logging.debug(
                "step=%d gripper_chunk[:%d]: min=%.3f max=%.3f mean=%.3f → %s (delta mode)",
                env_steps, args.execution_steps,
                float(gripper_chunk.min()), float(gripper_chunk.max()),
                float(gripper_chunk.mean()),
                "OPEN" if gripper_chunk.mean() > 0.5 else "CLOSED",  # DROID: >0.5=open
            )
            if actions.ndim != 2 or actions.shape[-1] < 8:
                raise RuntimeError(
                    f"Expected pi0 actions shape (H, >=8), got {actions.shape}"
                )

            # ── Convert velocity chunk → normalised per-step deltas ───────────
            # pi0 vel ±1 → normalised delta ±1 → actual ±0.1 rad/step (pd_joint_delta_pos)
            # With action_scale=0.5: normalised_delta = pi0_vel (identity)
            joint_pos_actions = pi0_vel_to_delta_actions(
                actions,
                action_scale=args.action_scale,
                execution_steps=args.execution_steps,
            )

            # ── Execute waypoints ─────────────────────────────────────────────
            done = False
            for action_8d in joint_pos_actions:
                if env_steps >= args.max_rollout_steps:
                    break
                env_obs, _reward, terminated, truncated, info = env.step(action_8d)
                env_steps += 1
                pbar.update(1)

                # Collect image after each step for smooth video
                _, ext_step, wrist_step = _extract_sim_obs(env_obs)
                side_image_list.append(ext_step)
                wrist_image_list.append(wrist_step)

                if bool(info.get("success", False)):
                    success = True
                    failure_reason = ""
                    done = True
                    break
                if bool(terminated) or bool(truncated):
                    done = True
                    break

            if done:
                break
    finally:
        pbar.close()

    duration_s = time.time() - start_time
    video_path = save_rollout_video(outputdir, episode_id, side_image_list, camera_name="side")
    wrist_video_path = save_rollout_video(
        outputdir, episode_id, wrist_image_list, camera_name="wrist"
    )
    return RolloutResult(
        episode_id=episode_id,
        success=success,
        failure_reason=failure_reason,
        env_steps=env_steps,
        duration_s=duration_s,
        video_path=video_path,
        timestamp=timestamp,
    ), wrist_video_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate pi0_droid in ManiSkill simulation.")
    parser.add_argument("--instruction", default="pick up the blue peg and insert it into the hole")
    # parser.add_argument("--instruction", default="pick up the blue peg")
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--max_rollout_steps", default=600, type=int)
    parser.add_argument(
        "--execution_steps",
        default=6,
        type=int,
        help=(
            "Number of joint-position waypoints to execute between pi0 inferences. "
            "Matches evaluate_pi0_real.py default. Default: 6."
        ),
    )
    parser.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--robofac_python", default=DEFAULT_ROBOFAC_PYTHON)
    parser.add_argument(
        "--action_scale",
        default=0.5,
        type=float,
        help=(
            "Velocity scale for delta mode. Default 0.5. "
            "normalised_delta = clip(pi0_vel, ±1) × action_scale × 2.0 (then clipped to ±1). "
            "With action_scale=0.5: normalised_delta = pi0_vel (identity mapping)."
        ),
    )
    parser.add_argument("--use_exterior_camera", default=0, type=int, choices=(0, 1),
                        help=("Pass exterior (base_camera) image to pi0. "
                              "Default 0 = wrist-only, matches real eval --use_exterior_camera=0."))
    parser.add_argument("--workspace_bounds_path",
                        default=None,  # workspace constraint is for SAC training; disable for pi0-only eval
                        help="Path to workspace_bounds.json. Default None = disabled for pi0 eval.")
    parser.add_argument("--outputdir", default=None)
    parser.add_argument(
        "--config_name",
        default="pi0_droid",
        help="OpenPI training config name. Use pi05_droid for pi0.5 checkpoints.",
    )
    return parser


def run_evaluation(args: argparse.Namespace) -> None:
    if args.eval_episodes <= 0:
        raise ValueError("--eval_episodes must be positive.")
    if args.max_rollout_steps <= 0:
        raise ValueError("--max_rollout_steps must be positive.")
    if args.execution_steps <= 0:
        raise ValueError("--execution_steps must be positive.")
    if args.action_scale <= 0:
        raise ValueError("--action_scale must be positive.")
    if not Path(args.checkpoint_path).exists():
        raise FileNotFoundError(f"pi0 checkpoint path does not exist: {args.checkpoint_path}")

    outputdir = resolve_outputdir(args.outputdir, prefix="pi0_delta_eval_sim")
    csv_path = outputdir / "eval_results.csv"
    logging.info("Writing pi0 sim evaluation outputs to %s", outputdir)

    env = ManiSkillRemoteEnv(robofac_python=args.robofac_python,
                             workspace_bounds_path=args.workspace_bounds_path or None,
                             control_mode="pd_joint_delta_pos")
    agent_dp = load_pi0_policy(args.checkpoint_path, args.config_name)
    completed = 0
    successes = 0
    try:
        for episode_id in range(args.eval_episodes):
            result, wrist_video_path = run_rollout(args, env, agent_dp, episode_id, outputdir)
            completed += 1
            successes += int(result.success)

            row = {
                "episode_id": result.episode_id,
                "success": int(result.success),
                "failure_reason": result.failure_reason,
                "env_steps": result.env_steps,
                "duration_s": f"{result.duration_s:.3f}",
                "video_path": result.video_path,
                "wrist_video_path": wrist_video_path,
                "timestamp": result.timestamp,
                "instruction": args.instruction,
                "checkpoint_path": args.checkpoint_path,
                "execution_steps": args.execution_steps,
                "max_rollout_steps": args.max_rollout_steps,
                "action_scale": args.action_scale,
                "robofac_python": args.robofac_python,
            }
            append_result(csv_path, row, RESULT_FIELDS)
            logging.info(
                "Episode %d done: success=%s reason=%s steps=%d duration=%.2fs rate=%.3f",
                result.episode_id,
                result.success,
                result.failure_reason or "success",
                result.env_steps,
                result.duration_s,
                successes / completed,
            )
    finally:
        try:
            agent_dp.close()
        finally:
            env.close()

    logging.info("Pi0 sim evaluation complete. Results: %s", csv_path)
    print(f"Pi0 sim evaluation complete. Results: {csv_path}")


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--_policy_worker":
        _cfg = sys.argv[3] if len(sys.argv) >= 4 else "pi0_droid"
        run_policy_worker(sys.argv[2], _cfg)
        return
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    run_evaluation(args)


if __name__ == "__main__":
    main()
