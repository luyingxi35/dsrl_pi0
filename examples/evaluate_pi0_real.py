#!/usr/bin/env python3
"""Standalone pi0-only real-world evaluation with wrist/exterior camera observations."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import logging
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.real_robot_common import (
    DEFAULT_WRIST_CAMERA_ID,
    DEFAULT_EXTERIOR_CAMERA_ID,
    PolicyServerConfig,
    PolicyService,
    RolloutResult,
    HumanEvalUI,
    binarize_and_clip_action,
    extract_observation_eval,
    get_pi0_input_eval,
    save_rollout_video,
    select_video_frame_eval,
    reset_robot,
    resolve_outputdir,
    append_result,
    format_stats,
)
DEFAULT_CONTROL_FREQUENCY = 10
VIDEO_FPS = 15

RESULT_FIELDS = [
    "episode_id", "success", "failure_reason", "env_steps", "duration_s",
    "video_path", "timestamp", "instruction",
    "use_wrist_camera", "use_exterior_camera",
    "wrist_camera_id", "exterior_camera_id",
    "policy_host", "policy_port",
]


@dataclasses.dataclass(frozen=True)
class EvalRobotConfig:
    """Lightweight robot config for pi0-eval (wrist + optional exterior camera)."""
    max_duration_s: float = 60.0    # episode timeout in wall-clock seconds
    wrist_camera_id: str | None = None
    exterior_camera_id: str | None = None
    control_frequency_hz: int = DEFAULT_CONTROL_FREQUENCY
    # ── Camera observation latencies (seconds) — calibrate empirically ─────────
    # Each value = time from actual frame capture to when read_end is recorded.
    # Used to anchor t_obs to the true camera capture moment (UMI-style).
    # t_obs = camera_read_end_ms / 1000 - obs_latency
    wrist_camera_obs_latency: float = 0.125     # ZedMini 60 fps — placeholder
    exterior_camera_obs_latency: float = 0.175  # RealSense      — placeholder
    # ── Proprioception latencies (seconds) — calibrate empirically ─────────────
    # Time from physical robot state to when the NUC's gRPC read completes.
    # Used to interpolate the 200 Hz state buffer to the camera obs timestamp.
    proprioceptive_latency: float = 0.001  # joint positions read delay — placeholder
    gripper_obs_latency: float = 0.020     # gripper state read delay   — placeholder

    def validate(self) -> None:
        if not self.wrist_camera_id and not self.exterior_camera_id:
            raise ValueError("At least one of --use_wrist_camera or --use_exterior_camera must be enabled.")


@dataclasses.dataclass(frozen=True)
class ExecutionConfig:
    """Timing configuration for non-blocking timestamped action scheduling."""
    # Number of actions to execute before re-inferring (analogous to UMI's steps_per_inference).
    execution_steps: int = 6
    # Time (s) from arm command issuance to robot physically responding.
    # Arm waypoint times are advanced by this amount so the robot arrives at
    # target pose at the intended moment (matches UMI's robot_action_latency).
    robot_action_latency: float = 0.20
    # Time (s) from gripper command to gripper physically responding.
    gripper_action_latency: float = 0.15
    # Minimum lead time (s) to schedule an action. Actions whose
    # target_time <= curr_time + action_exec_latency are skipped as stale.
    action_exec_latency: float = 0.01
    # Frequency (Hz) of the high-frequency joint position controller.
    controller_frequency: float = 200.0
    # Scale factor applied to DROID's training max_joint_delta (0.2 rad/step).
    # action_scale=1.0 → 0.20 rad/step (full training speed)
    # action_scale=0.5 → 0.10 rad/step (half speed, safer default)
    action_scale: float = 0.5


class RobotIO:
    """Thin wrapper around DROID RobotEnv for pi0-eval scripts."""

    def __init__(self, robot_config: EvalRobotConfig) -> None:
        from droid.robot_env import RobotEnv
        self._robot_config = robot_config
        # pi0 outputs joint_velocity (normalized [-1,1]); client integrates to
        # absolute positions before sending to the NUC HighFreqController.
        self._env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")

    @property
    def env(self):
        return self._env

    @property
    def robot_config(self) -> EvalRobotConfig:
        return self._robot_config

    def preflight(self) -> None:
        from utils.real_robot_common import _find_camera_image
        obs = self._env.get_observation()
        images = obs["image"]
        missing = []
        if self._robot_config.wrist_camera_id and _find_camera_image(images, self._robot_config.wrist_camera_id) is None:
            missing.append("wrist")
        if self._robot_config.exterior_camera_id and _find_camera_image(images, self._robot_config.exterior_camera_id) is None:
            missing.append("exterior")
        if missing:
            raise RuntimeError(
                "DROID camera preflight failed. Missing: " + ", ".join(missing) +
                f". Available: {sorted(images.keys())}."
            )


def run_rollout(
    args: argparse.Namespace,
    env: Any,
    policy_service: PolicyService,
    robot_io: RobotIO,
    ui: HumanEvalUI,
    episode_id: int,
    completed: int,
    successes: int,
    outputdir: Path,
    exec_config: ExecutionConfig | None = None,
) -> RolloutResult:
    from tqdm import tqdm

    if exec_config is None:
        exec_config = ExecutionConfig()

    robot_config = robot_io.robot_config
    dt_step = 1.0 / robot_config.control_frequency_hz

    # ── Warm-up: trigger impedance controller on NUC ──────────────────────────
    # update_joints(blocking=False) starts DROID's impedance controller via the
    # NUC server's helper_non_blocking thread, which calls start_cartesian_impedance().
    # The HighFreqController needs impedance active before sending targets.
    current_joints = np.array(env.get_observation()["robot_state"]["joint_positions"])
    env._robot.update_joints(current_joints, velocity=False, blocking=False)
    time.sleep(0.15)  # allow impedance to activate on NUC

    # ── Start high-frequency controller ON THE NUC (zerorpc call) ─────────────
    # HighFreqController runs on the NUC alongside Polymetis, not on GPU server.
    # env._robot is ServerInterface → this call goes over zerorpc to NUC:4242.
    env._robot.start_trajectory_controller(exec_config.controller_frequency)

    # ── Inference concurrency ──────────────────────────────────────────────────
    # Gripper and obs calls use zerorpc/gevent → must stay on main thread.
    # policy_service.infer() uses openpi websocket (not gevent) → safe in thread.
    inference_queue: queue.Queue = queue.Queue()
    inference_in_progress = threading.Event()

    def _run_inference(obs_snapshot: dict, t_obs: float) -> None:
        try:
            request_data = get_pi0_input_eval(obs_snapshot, args.instruction)
            response = policy_service.infer(request_data)
            inference_queue.put((np.asarray(response["actions"]), t_obs))
        except Exception:
            logging.exception("run_rollout: inference failed in background thread")
        finally:
            inference_in_progress.clear()

    # ── Gripper scheduling (15Hz discrete, main thread) ────────────────────────
    scheduled_gripper_actions: list[tuple[float, float]] = []

    # ── Episode bookkeeping ────────────────────────────────────────────────────
    ui.set_running(episode_id, completed, successes)
    start_time = time.time()
    t_loop_start = start_time
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    image_list: list[np.ndarray] = []
    decision: tuple[bool, str] | None = None
    env_steps = 0

    try:
        pbar = tqdm(desc=f"pi0 eval episode {episode_id}", unit="step")
        t = 0
        while True:
            # ── Timeout check (time-based, frequency-independent) ──────────────
            if time.time() - start_time >= robot_config.max_duration_s:
                decision = (False, "timeout")
                break

            decision = ui.poll()
            if decision is not None:
                break

            t_step_end = t_loop_start + (t + 1) * dt_step

            # ── 1. Observation ─────────────────────────────────────────────────
            # Pull the NUC's high-frequency state history before get_observation()
            # so the ring buffer already contains data up to this moment.
            # get_state_history() is a single zerorpc call (~1–2 ms).
            state_history = env._robot.get_state_history(n=100)   # last 0.5 s @ 200 Hz

            curr_obs, obs_timestamp = extract_observation_eval(
                robot_config.wrist_camera_id,
                robot_config.exterior_camera_id,
                env.get_observation(),
                wrist_obs_latency=robot_config.wrist_camera_obs_latency,
                exterior_obs_latency=robot_config.exterior_camera_obs_latency,
                proprioceptive_latency=robot_config.proprioceptive_latency,
                gripper_latency=robot_config.gripper_obs_latency,
                state_history=state_history,
            )
            # obs_timestamp is now anchored to the camera capture time (past),
            # mirroring UMI's hardware-timestamp-based t_obs.
            # action_timestamps = obs_timestamp + k*dt will correctly place
            # early actions in the past so is_new filters them out.
            if t == 0:
                logging.info(
                    "t_obs drift: %.1f ms (camera frame age = obs_latency + any extra delay; "
                    "tune *_camera_obs_latency if this deviates from expected latency)",
                    (time.time() - obs_timestamp) * 1000,
                )

            # ── 2. Drain inference queue (keep most recent) ────────────────────
            latest_result = None
            while not inference_queue.empty():
                try:
                    latest_result = inference_queue.get_nowait()
                except queue.Empty:
                    break

            if latest_result is not None:
                new_actions, t_obs = latest_result
                action_timestamps = t_obs + np.arange(len(new_actions)) * dt_step
                curr_time = time.time()
                is_new = action_timestamps > (curr_time + exec_config.action_exec_latency)
                if np.any(is_new):
                    new_a = new_actions[is_new][: exec_config.execution_steps]
                    new_t = action_timestamps[is_new][: exec_config.execution_steps]

                    # Arm: integrate joint_velocity → cumulative absolute joint angles.
                    # pi0 outputs normalized joint velocities in [-1, 1].
                    # delta_k = clip(v_k, -1, 1) * MAX_JOINT_DELTA  (rad/step)
                    # MAX_JOINT_DELTA = 0.2 (DROID training constant) * action_scale
                    # pos_k   = current_joints + sum(delta_0 .. delta_k)
                    # Source: droid/robot_ik/robot_ik_solver.py, relative_max_joint_delta
                    _MAX_JOINT_DELTA = 0.2 * exec_config.action_scale  # rad per step
                    _running_joints = curr_obs["joint_position"].copy()
                    _arm_abs: list[np.ndarray] = []
                    for _a in new_a:
                        _vel = np.clip(_a[:-1], -1.0, 1.0)
                        _running_joints = _running_joints + _vel * _MAX_JOINT_DELTA
                        _arm_abs.append(_running_joints.copy())
                    arm_positions = np.array(_arm_abs)  # (N, 7) radians

                    # Send to NUC's HighFreqController via zerorpc.
                    # Subtract robot_action_latency so the 200Hz loop calls
                    # update_desired_joint_positions early enough to arrive on time.
                    # Lists (not numpy) because msgpack serialises them natively.
                    arm_times = new_t - exec_config.robot_action_latency
                    env._robot.add_waypoints(arm_times.tolist(), arm_positions.tolist())

                    # Gripper: keep in 10Hz discrete schedule
                    scheduled_gripper_actions = [
                        (ts, float(binarize_and_clip_action(a)[-1]))
                        for ts, a in zip(new_t, new_a)
                    ]
                else:
                    logging.warning("run_rollout: all actions stale at t=%d", t)

            # ── 3. Trigger inference (unconditional cadence, aligned with UMI) ──
            if t % exec_config.execution_steps == 0 and not inference_in_progress.is_set():
                inference_in_progress.set()
                threading.Thread(
                    target=_run_inference,
                    args=(curr_obs, obs_timestamp),
                    daemon=True,
                    name=f"Infer-t{t}",
                ).start()

            # ── 4. Gripper execution (main thread, gevent-safe) ────────────────
            curr_time = time.time()
            gripper_to_exec = None
            while (
                scheduled_gripper_actions
                and scheduled_gripper_actions[0][0] - exec_config.gripper_action_latency <= curr_time
            ):
                _, gripper_to_exec = scheduled_gripper_actions.pop(0)

            if gripper_to_exec is not None:
                env._robot.update_gripper(gripper_to_exec, velocity=False, blocking=False)

            # ── 5. UI & bookkeeping ────────────────────────────────────────────
            ui.update_camera_previews(
                wrist=curr_obs.get("wrist_image"),
                exterior=curr_obs.get("exterior_image"),
            )
            image_list.append(select_video_frame_eval(curr_obs))
            env_steps = t + 1
            ui.update_step(episode_id, env_steps, completed, successes)

            decision = ui.poll()
            if decision is not None:
                break

            # ── 6. Wait for tick deadline ──────────────────────────────────────
            sleep_s = t_step_end - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)

            pbar.update(1)
            t += 1

    finally:
        pbar.close()
        env._robot.stop_trajectory_controller()   # stops HighFreqController on NUC
        inference_in_progress.wait(timeout=5.0)

    if decision is None:
        decision = (False, "timeout")

    success, failure_reason = decision
    duration_s = time.time() - start_time
    video_path = save_rollout_video(outputdir, episode_id, image_list)

    return RolloutResult(
        episode_id=episode_id,
        success=bool(success),
        failure_reason="" if success else failure_reason,
        env_steps=env_steps,
        duration_s=duration_s,
        video_path=video_path,
        timestamp=timestamp,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate pi0-only policy on a real DROID robot using selected cameras."
    )
    parser.add_argument("--instruction", default="put the spoon on the plate")
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--max_duration_s", default=60.0, type=float,
        help="Max episode duration in seconds (timeout). Default: 60.")
    parser.add_argument("--execution_steps", default=6, type=int,
        help="Actions to execute before re-inferring (analogous to UMI steps_per_inference). Default: 6.")
    parser.add_argument("--robot_action_latency", default=0.20, type=float,
        help="Arm command lead time in seconds (robot_action_latency). Default: 0.20.")
    parser.add_argument("--gripper_action_latency", default=None, type=float,
        help="Gripper command lead time in seconds. Defaults to robot_action_latency.")
    parser.add_argument("--action_exec_latency", default=0.01, type=float,
        help="Minimum scheduling lead time in seconds. Default: 0.01.")
    parser.add_argument("--control_frequency_hz", default=DEFAULT_CONTROL_FREQUENCY, type=int,
        help=f"Target DROID control frequency in Hz. Default: {DEFAULT_CONTROL_FREQUENCY}.")
    parser.add_argument("--controller_frequency", default=200.0, type=float,
        help="High-frequency joint controller loop rate in Hz. Default: 200.")
    parser.add_argument("--action_scale", default=0.5, type=float,
        help="Scale factor on DROID training max_joint_delta (0.2 rad/step). "
             "action_scale=1.0 = full speed, 0.5 = half speed. Default: 0.5.")
    # ── Camera / proprioception observation latencies ──────────────────────────
    parser.add_argument("--wrist_camera_obs_latency", default=None, type=float,
        help="ZedMini wrist camera obs latency (s). "
             "Default: EvalRobotConfig default (0.125 s). Calibrate empirically.")
    parser.add_argument("--exterior_camera_obs_latency", default=None, type=float,
        help="RealSense exterior camera obs latency (s). "
             "Default: EvalRobotConfig default (0.175 s). Calibrate empirically.")
    parser.add_argument("--proprioceptive_latency", default=None, type=float,
        help="Joint position read latency (s). "
             "Default: EvalRobotConfig default (0.001 s). Calibrate empirically.")
    parser.add_argument("--gripper_obs_latency", default=None, type=float,
        help="Gripper state read latency (s). "
             "Default: EvalRobotConfig default (0.020 s). Calibrate empirically.")
    parser.add_argument("--use_wrist_camera", default=1, type=int, choices=(0, 1))
    parser.add_argument("--use_exterior_camera", default=0, type=int, choices=(0, 1))
    parser.add_argument("--policy_host", default="127.0.0.1")
    parser.add_argument("--policy_port", default=8000, type=int)
    parser.add_argument("--outputdir", default=None)
    return parser


def run_evaluation(args: argparse.Namespace) -> None:
    if args.eval_episodes <= 0:
        raise ValueError("--eval_episodes must be positive.")
    if args.execution_steps <= 0:
        raise ValueError("--execution_steps must be positive.")
    if not args.use_wrist_camera and not args.use_exterior_camera:
        raise ValueError("At least one camera must be enabled.")

    exec_config = ExecutionConfig(
        execution_steps=args.execution_steps,
        robot_action_latency=args.robot_action_latency,
        gripper_action_latency=(
            args.robot_action_latency if args.gripper_action_latency is None
            else args.gripper_action_latency
        ),
        action_exec_latency=args.action_exec_latency,
        controller_frequency=args.controller_frequency,
        action_scale=args.action_scale,
    )
    logging.info(
        "ExecutionConfig: execution_steps=%d robot_action_latency=%.3fs "
        "gripper_action_latency=%.3fs action_exec_latency=%.3fs "
        "controller_frequency=%.0fHz action_scale=%.2f (max_joint_delta=%.3f rad/step)",
        exec_config.execution_steps, exec_config.robot_action_latency,
        exec_config.gripper_action_latency, exec_config.action_exec_latency,
        exec_config.controller_frequency,
        exec_config.action_scale, 0.2 * exec_config.action_scale,
    )

    _cfg_defaults = EvalRobotConfig.__dataclass_fields__
    robot_config = EvalRobotConfig(
        max_duration_s=args.max_duration_s,
        wrist_camera_id=DEFAULT_WRIST_CAMERA_ID if args.use_wrist_camera else None,
        exterior_camera_id=DEFAULT_EXTERIOR_CAMERA_ID if args.use_exterior_camera else None,
        control_frequency_hz=args.control_frequency_hz,
        wrist_camera_obs_latency=(
            _cfg_defaults["wrist_camera_obs_latency"].default
            if args.wrist_camera_obs_latency is None else args.wrist_camera_obs_latency
        ),
        exterior_camera_obs_latency=(
            _cfg_defaults["exterior_camera_obs_latency"].default
            if args.exterior_camera_obs_latency is None else args.exterior_camera_obs_latency
        ),
        proprioceptive_latency=(
            _cfg_defaults["proprioceptive_latency"].default
            if args.proprioceptive_latency is None else args.proprioceptive_latency
        ),
        gripper_obs_latency=(
            _cfg_defaults["gripper_obs_latency"].default
            if args.gripper_obs_latency is None else args.gripper_obs_latency
        ),
    )
    logging.info(
        "Camera obs latencies: wrist=%.3fs exterior=%.3fs | "
        "State latencies: proprioceptive=%.3fs gripper=%.3fs",
        robot_config.wrist_camera_obs_latency, robot_config.exterior_camera_obs_latency,
        robot_config.proprioceptive_latency, robot_config.gripper_obs_latency,
    )
    robot_config.validate()

    policy_service = PolicyService(PolicyServerConfig(host=args.policy_host, port=args.policy_port))
    policy_service.preflight()

    robot_io = RobotIO(robot_config)
    robot_io.preflight()
    env = robot_io.env

    outputdir = resolve_outputdir(args.outputdir, prefix="pi0_eval_real")
    csv_path = outputdir / "eval_results.csv"
    logging.info("Writing pi0 evaluation outputs to %s", outputdir)

    ui = HumanEvalUI(
        title="Pi0 Real Evaluation",
        total_episodes=args.eval_episodes,
        preview_names=(("wrist", "Wrist"), ("exterior", "Exterior")),
    )

    completed = 0
    successes = 0
    try:
        for episode_id in range(args.eval_episodes):
            if not ui.wait_for_start(episode_id, completed, successes):
                break

            reset_robot(env, reason=f"before episode {episode_id}")
            result = run_rollout(
                args, env, policy_service, robot_io, ui,
                episode_id, completed, successes, outputdir,
                exec_config=exec_config,
            )
            completed += 1
            successes += int(result.success)

            row = {
                "episode_id": result.episode_id,
                "success": int(result.success),
                "failure_reason": result.failure_reason,
                "env_steps": result.env_steps,
                "duration_s": f"{result.duration_s:.3f}",
                "video_path": result.video_path,
                "timestamp": result.timestamp,
                "instruction": args.instruction,
                "use_wrist_camera": int(args.use_wrist_camera),
                "use_exterior_camera": int(args.use_exterior_camera),
                "wrist_camera_id": DEFAULT_WRIST_CAMERA_ID if args.use_wrist_camera else "",
                "exterior_camera_id": DEFAULT_EXTERIOR_CAMERA_ID if args.use_exterior_camera else "",
                "policy_host": args.policy_host,
                "policy_port": args.policy_port,
            }
            append_result(csv_path, row, RESULT_FIELDS)
            logging.info(
                "Episode %d done: success=%s reason=%s steps=%d duration=%.2fs rate=%.3f",
                result.episode_id, result.success,
                result.failure_reason or "success",
                result.env_steps, result.duration_s,
                successes / completed,
            )

            ui.set_resetting(episode_id, completed, successes)
            reset_robot(env, reason=f"after episode {episode_id}")

            if ui.quit_requested:
                break
    finally:
        ui.close()

    logging.info("Pi0 evaluation complete. Results: %s", csv_path)
    print(f"Pi0 evaluation complete. Results: {csv_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    run_evaluation(args)


if __name__ == "__main__":
    main()
