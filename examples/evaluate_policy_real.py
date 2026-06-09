#!/usr/bin/env python3
"""Real-world evaluation for DSRL pi0: StateSACLearner (train_real_dino ckpt) + pi0 server.

The agent architecture mirrors train_real_dino.py exactly:
  1. Extract observation with latency-corrected UMI-style timestamps.
  2. Build 2440-dim state: joint(8) + pi0 VLM embed(2048) + wrist DINO-v2(384).
  3. StateSAC actor predicts denoising noise  (shape: rl_noise_horizon × PI0_NOISE_DIM).
  4. Pass noise to pi0 server → receive joint-velocity action chunk.
  5. Integrate velocities → absolute positions, send timestamped waypoints to NUC.
  6. Schedule binarised gripper commands separately at control frequency.

Observation and action-execution sides are aligned with evaluate_pi0_real.py (latency
calibration, is_new stale-action filtering, async inference thread, high-freq controller).
"""

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
    "video_path", "timestamp", "instruction", "restore_path",
    "use_wrist_camera", "use_exterior_camera",
    "wrist_camera_id", "exterior_camera_id",
    "policy_host", "policy_port",
    "rl_noise_horizon",
]


# ── Dataclasses (identical to evaluate_pi0_real.py) ───────────────────────────

@dataclasses.dataclass(frozen=True)
class EvalRobotConfig:
    """Lightweight robot config for dino-eval (wrist + optional exterior camera)."""
    max_duration_s: float = 60.0
    wrist_camera_id: str | None = None
    exterior_camera_id: str | None = None
    control_frequency_hz: int = DEFAULT_CONTROL_FREQUENCY
    wrist_camera_obs_latency: float = 0.084
    exterior_camera_obs_latency: float = 0.08
    proprioceptive_latency: float = 0.0003
    gripper_obs_latency: float = 0.00003

    def validate(self) -> None:
        if not self.wrist_camera_id and not self.exterior_camera_id:
            raise ValueError("At least one of --use_wrist_camera or --use_exterior_camera must be enabled.")


@dataclasses.dataclass(frozen=True)
class ExecutionConfig:
    """Timing configuration for non-blocking timestamped action scheduling."""
    execution_steps: int = 6
    robot_action_latency: float = 0.20
    gripper_action_latency: float = 0.15
    action_exec_latency: float = 0.01
    controller_frequency: float = 200.0
    action_scale: float = 0.5
    max_joint_speed_rad_s: float = 0.5


class RobotIO:
    """Thin wrapper around DROID RobotEnv for dino-eval scripts."""

    def __init__(self, robot_config: EvalRobotConfig) -> None:
        from droid.robot_env import RobotEnv
        self._robot_config = robot_config
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


# ── Agent helpers ──────────────────────────────────────────────────────────────

def create_agent(args: argparse.Namespace) -> Any:
    """Build and restore a StateSACLearner from a train_real_dino checkpoint."""
    from jaxrl2.agents.state_sac.state_sac_learner import StateSACLearner
    from jaxrl2.utils.general_utils import add_batch_dim
    from examples.train_real_dino import DummyEnv, STATE_DIM, PI0_NOISE_DIM

    class _VariantLike:
        rl_noise_horizon = args.rl_noise_horizon

    dummy_env = DummyEnv(_VariantLike())
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())

    train_kwargs = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=tuple(args.hidden_dims),
        network_type=args.network_type,
        transformer_dim=args.transformer_dim,
        transformer_depth=args.transformer_depth,
        transformer_heads=args.transformer_num_heads,   # StateSACLearner kwarg name
        discount=0.99,
        tau=0.005,
        num_qs=2,
        action_magnitude=2.0,
    )

    agent = StateSACLearner(args.seed, sample_obs, sample_action, **train_kwargs)
    agent.restore_checkpoint(args.restore_path)
    logging.info("StateSACLearner restored from %s  (action_chunk_shape=%s)",
                 args.restore_path, agent.action_chunk_shape)
    return agent


def create_obs_builder(args: argparse.Namespace) -> Any:
    """Build WristDinoObservationBuilder (loads DINO-v2 model once)."""
    from examples.train_real_dino import WristDinoFeatureExtractor, WristDinoObservationBuilder
    dino_extractor = WristDinoFeatureExtractor(
        model_name=args.dino_model_name,
        device=args.dino_device,
    )
    logging.info("WristDinoFeatureExtractor loaded (model=%s, device=%s, feat_dim=%d)",
                 args.dino_model_name, args.dino_device, dino_extractor.feature_dim)
    return WristDinoObservationBuilder(dino_extractor)


# ── Main rollout ───────────────────────────────────────────────────────────────

def run_rollout(
    args: argparse.Namespace,
    env: Any,
    policy_service: PolicyService,
    robot_io: RobotIO,
    agent: Any,
    obs_builder: Any,
    ui: HumanEvalUI,
    episode_id: int,
    completed: int,
    successes: int,
    outputdir: Path,
    exec_config: ExecutionConfig | None = None,
) -> RolloutResult:
    from jaxrl2.utils.noise_utils import make_full_horizon_noise
    from tqdm import tqdm

    if exec_config is None:
        exec_config = ExecutionConfig()

    robot_config = robot_io.robot_config
    dt_step = 1.0 / robot_config.control_frequency_hz

    # ── Warm-up: trigger impedance controller on NUC ──────────────────────────
    current_joints = np.array(env.get_observation()["robot_state"]["joint_positions"])
    env._robot.update_joints(current_joints, velocity=False, blocking=False)
    time.sleep(0.15)

    # ── Start high-frequency controller on NUC ────────────────────────────────
    env._robot.start_trajectory_controller(exec_config.controller_frequency)

    # ── Pre-populate interpolator with a hold-in-place trajectory ─────────────
    _hold_joints  = np.tile(current_joints, (4, 1))
    _hold_offsets = [0.05, 0.20, 0.50, 1.00]
    env._robot.add_waypoints(_hold_offsets, _hold_joints.tolist(),
                             max_joint_speed_rad_s=exec_config.max_joint_speed_rad_s)
    time.sleep(0.05)

    # ── Inference concurrency ──────────────────────────────────────────────────
    # policy_service.infer + DINO encode + JAX agent.eval_actions run in background.
    # Gripper and zerorpc calls stay on main thread (gevent-safe requirement).
    inference_queue: queue.Queue = queue.Queue()
    inference_in_progress = threading.Event()

    def _run_inference(obs_snapshot: dict, t_obs: float) -> None:
        try:
            # 1. Build pi0 request dict (images + joint/gripper + instruction)
            request_data = get_pi0_input_eval(obs_snapshot, args.instruction)
            # 2. Build 2440-dim state: proprio(8) + pi0_vlm(2048) + dino(384)
            #    obs_builder.build() calls policy_service.get_prefix_rep() internally.
            obs_dict = obs_builder.build(obs_snapshot, request_data, policy_service)
            # 3. StateSAC actor predicts denoising noise
            actions_noise = agent.eval_actions(obs_dict)
            # 4. Reshape to (1, rl_noise_horizon, PI0_NOISE_DIM) for pi0 server
            _, noise = make_full_horizon_noise(actions_noise, agent.action_chunk_shape)
            # 5. Pi0 server denoises with RL noise → joint-velocity action chunk
            response = policy_service.infer(request_data, noise=np.asarray(noise))
            inference_queue.put((np.asarray(response["actions"]), t_obs))
        except Exception:
            logging.exception("run_rollout: inference failed in background thread")
        finally:
            inference_in_progress.clear()

    # ── Gripper scheduling ─────────────────────────────────────────────────────
    scheduled_gripper_actions: list[tuple[float, float]] = []

    # ── Episode bookkeeping ────────────────────────────────────────────────────
    ui.set_running(episode_id, completed, successes)
    start_time = time.time()
    t_loop_start = start_time
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    image_list: list[np.ndarray] = []
    decision: tuple[bool, str] | None = None
    env_steps = 0

    _DROID_MAX_JOINT_DELTA = 0.2
    _MAX_JOINT_DELTA = _DROID_MAX_JOINT_DELTA * exec_config.action_scale

    try:
        pbar = tqdm(desc=f"dino eval episode {episode_id}", unit="step")
        t = 0
        while True:
            # ── Timeout check ──────────────────────────────────────────────────
            if time.time() - start_time >= robot_config.max_duration_s:
                decision = (False, "timeout")
                break

            decision = ui.poll()
            if decision is not None:
                break

            t_step_end = t_loop_start + (t + 1) * dt_step

            # ── 1. Observation (latency-corrected, UMI-style) ──────────────────
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
            if t == 0:
                logging.info(
                    "t_obs drift: %.1f ms",
                    (time.time() - obs_timestamp) * 1000,
                )

            # ── 2. Drain inference queue (keep most recent result) ─────────────
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
                    # Integrate the FULL chunk first (including stale frames) so
                    # absolute positions are consistent; then extract is_new slice.
                    _running_joints = curr_obs["joint_position"].copy()
                    _all_abs: list[np.ndarray] = []
                    for _a in new_actions:
                        _vel = np.clip(_a[:-1], -1.0, 1.0)
                        _running_joints = _running_joints + _vel * _MAX_JOINT_DELTA
                        _all_abs.append(_running_joints.copy())
                    _all_abs_arr = np.array(_all_abs)   # (N_total, 7)

                    arm_positions = _all_abs_arr[is_new][: exec_config.execution_steps]
                    new_t = action_timestamps[is_new][: exec_config.execution_steps]
                    new_a = new_actions[is_new][: exec_config.execution_steps]

                    arm_time_offsets = (new_t - exec_config.robot_action_latency) - time.time()
                    env._robot.add_waypoints(arm_time_offsets.tolist(), arm_positions.tolist(),
                                             max_joint_speed_rad_s=exec_config.max_joint_speed_rad_s)

                    scheduled_gripper_actions = [
                        (ts, float(binarize_and_clip_action(a)[-1]))
                        for ts, a in zip(new_t, new_a)
                    ]
                else:
                    logging.warning("run_rollout: all actions stale at t=%d", t)

            # ── 3. Trigger inference (aligned with execution_steps cadence) ─────
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
        env._robot.stop_trajectory_controller()
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


# ── Argument parser ────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a train_real_dino StateSAC policy (+ pi0 server) on a real DROID robot."
    )
    # ── Required ──────────────────────────────────────────────────────────────
    parser.add_argument("--restore_path", required=True,
        help="Path to the StateSACLearner checkpoint saved by train_real_dino.")
    # ── Episode config ────────────────────────────────────────────────────────
    parser.add_argument("--instruction", default="put the spoon on the plate")
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--max_duration_s", default=60.0, type=float,
        help="Max episode duration in seconds. Default: 60.")
    # ── Action scheduling ─────────────────────────────────────────────────────
    parser.add_argument("--execution_steps", default=6, type=int,
        help="Actions to execute before re-inferring. Default: 6.")
    parser.add_argument("--robot_action_latency", default=0.20, type=float)
    parser.add_argument("--gripper_action_latency", default=0.15, type=float)
    parser.add_argument("--action_exec_latency", default=0.01, type=float)
    parser.add_argument("--control_frequency_hz", default=DEFAULT_CONTROL_FREQUENCY, type=int)
    parser.add_argument("--controller_frequency", default=200.0, type=float)
    parser.add_argument("--action_scale", default=0.5, type=float,
        help="Scale factor on DROID training max_joint_delta (0.2 rad/step). Default: 0.5.")
    parser.add_argument("--max_joint_speed_rad_s", default=0.5, type=float,
        help="NUC-side per-joint speed cap (rad/s) passed to add_waypoints. Default: 0.5.")
    # ── Observation latencies ─────────────────────────────────────────────────
    parser.add_argument("--wrist_camera_obs_latency", default=None, type=float)
    parser.add_argument("--exterior_camera_obs_latency", default=None, type=float)
    parser.add_argument("--proprioceptive_latency", default=None, type=float)
    parser.add_argument("--gripper_obs_latency", default=None, type=float)
    # ── Camera selection ──────────────────────────────────────────────────────
    parser.add_argument("--use_wrist_camera", default=1, type=int, choices=(0, 1))
    parser.add_argument("--use_exterior_camera", default=0, type=int, choices=(0, 1))
    # ── Policy server ─────────────────────────────────────────────────────────
    parser.add_argument("--policy_host", default="127.0.0.1")
    parser.add_argument("--policy_port", default=8000, type=int)
    # ── StateSAC model ────────────────────────────────────────────────────────
    parser.add_argument("--rl_noise_horizon", default=8, type=int,
        help="RL noise horizon (must match the trained checkpoint). Default: 8.")
    parser.add_argument("--network_type", default="transformer",
        choices=("transformer", "mlp"),
        help="StateSAC network type (must match the trained checkpoint). Default: transformer.")
    parser.add_argument("--hidden_dims", nargs="+", default=[1024, 1024, 1024], type=int,
        help="MLP hidden dims (only used when --network_type=mlp). Default: 1024 1024 1024.")
    parser.add_argument("--transformer_dim", default=256, type=int)
    parser.add_argument("--transformer_depth", default=3, type=int)
    parser.add_argument("--transformer_num_heads", default=4, type=int)
    # ── DINO feature extractor ────────────────────────────────────────────────
    parser.add_argument("--dino_model_name", default="facebook/dinov2-small",
        help="HuggingFace model name for wrist DINO-v2 feature extractor.")
    parser.add_argument("--dino_device", default="auto",
        help="Torch device for DINO ('auto', 'cuda', 'cpu'). Default: auto.")
    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--outputdir", default=None)
    return parser


# ── Top-level evaluation loop ─────────────────────────────────────────────────

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
        gripper_action_latency=args.gripper_action_latency,
        action_exec_latency=args.action_exec_latency,
        controller_frequency=args.controller_frequency,
        action_scale=args.action_scale,
        max_joint_speed_rad_s=args.max_joint_speed_rad_s,
    )
    logging.info(
        "ExecutionConfig: execution_steps=%d robot_action_latency=%.3fs "
        "gripper_action_latency=%.3fs action_exec_latency=%.3fs "
        "controller_frequency=%.0fHz action_scale=%.2f (max_joint_delta=%.3f rad/step) "
        "max_joint_speed_rad_s=%.2f",
        exec_config.execution_steps, exec_config.robot_action_latency,
        exec_config.gripper_action_latency, exec_config.action_exec_latency,
        exec_config.controller_frequency,
        exec_config.action_scale, 0.2 * exec_config.action_scale,
        exec_config.max_joint_speed_rad_s,
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

    # ── Model & obs builder (load once before robot preflight) ─────────────────
    logging.info("Loading StateSACLearner from %s ...", args.restore_path)
    agent = create_agent(args)

    logging.info("Loading WristDinoFeatureExtractor (%s) ...", args.dino_model_name)
    obs_builder = create_obs_builder(args)

    # ── Policy server ─────────────────────────────────────────────────────────
    policy_service = PolicyService(PolicyServerConfig(host=args.policy_host, port=args.policy_port))
    metadata = policy_service.preflight()

    # Validate server action shape against RL noise shape
    if metadata and "action_horizon" in metadata and "action_dim" in metadata:
        server_horizon = int(metadata["action_horizon"])
        server_dim = int(metadata["action_dim"])
        from examples.train_real_dino import PI0_NOISE_DIM
        noise_h, noise_d = agent.action_chunk_shape
        if server_horizon != noise_h or server_dim != noise_d:
            raise RuntimeError(
                f"pi0 server action shape ({server_horizon}, {server_dim}) does not match "
                f"RL noise shape ({noise_h}, {noise_d}). "
                "Restart the policy server from this repo."
            )
        logging.info("pi0 server action shape validated: (%d, %d)", server_horizon, server_dim)

    # ── Robot env ─────────────────────────────────────────────────────────────
    robot_io = RobotIO(robot_config)
    robot_io.preflight()
    env = robot_io.env

    outputdir = resolve_outputdir(args.outputdir, prefix="dino_eval_real")
    csv_path = outputdir / "eval_results.csv"
    logging.info("Writing dino evaluation outputs to %s", outputdir)

    ui = HumanEvalUI(
        title="DINO Policy Evaluation",
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
                args, env, policy_service, robot_io, agent, obs_builder, ui,
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
                "restore_path": args.restore_path,
                "use_wrist_camera": int(args.use_wrist_camera),
                "use_exterior_camera": int(args.use_exterior_camera),
                "wrist_camera_id": DEFAULT_WRIST_CAMERA_ID if args.use_wrist_camera else "",
                "exterior_camera_id": DEFAULT_EXTERIOR_CAMERA_ID if args.use_exterior_camera else "",
                "policy_host": args.policy_host,
                "policy_port": args.policy_port,
                "rl_noise_horizon": args.rl_noise_horizon,
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

    logging.info("Dino evaluation complete. Results: %s", csv_path)
    print(f"Dino evaluation complete. Results: {csv_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    run_evaluation(args)


if __name__ == "__main__":
    main()
