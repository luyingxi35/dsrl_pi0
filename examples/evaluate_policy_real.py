#!/usr/bin/env python3
"""Train-aligned real-world evaluation for DSRL pi0 checkpoints."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.real_robot_common import (
    DEFAULT_EXTERIOR_CAMERA_ID,
    DEFAULT_WRIST_CAMERA_ID,
    HumanEvalUI,
    PolicyServerConfig,
    PolicyService,
    RobotIO,
    RobotRuntimeConfig,
    RolloutResult,
    action_timestamps_from_obs,
    append_result,
    binarize_and_clip_action,
    get_pi0_input_train,
    integrate_joint_velocity_actions,
    reset_robot,
    resolve_outputdir,
    save_rollout_video,
    select_video_frame_train,
    extract_observation_train,
)


DEFAULT_CONTROL_FREQUENCY = 10
DEFAULT_QUERY_FREQ = 8

RESULT_FIELDS = [
    "episode_id", "success", "failure_reason", "env_steps", "duration_s",
    "video_path", "timestamp", "instruction", "restore_path",
    "use_wrist_camera", "use_exterior_camera", "external_camera",
    "wrist_camera_id", "left_camera_id", "right_camera_id",
    "control_frequency_hz", "query_freq", "max_rollout_steps",
    "dsrl_eval_timing_mode", "min_future_actions", "min_future_horizon_s",
    "policy_host", "policy_port", "rl_noise_horizon",
]


def create_agent(args: argparse.Namespace) -> Any:
    """Build and restore a StateSACLearner from a train_real_dino checkpoint."""
    from jaxrl2.agents.state_sac.state_sac_learner import StateSACLearner
    from jaxrl2.utils.general_utils import add_batch_dim
    from examples.train_real_dino import DummyEnv

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
        transformer_heads=args.transformer_num_heads,
        discount=0.99,
        tau=0.005,
        num_qs=2,
        action_magnitude=2.0,
    )

    agent = StateSACLearner(args.seed, sample_obs, sample_action, **train_kwargs)
    agent.restore_checkpoint(args.restore_path)
    logging.info(
        "StateSACLearner restored from %s (action_chunk_shape=%s)",
        args.restore_path,
        agent.action_chunk_shape,
    )
    return agent


def create_obs_builder(args: argparse.Namespace) -> Any:
    """Build WristDinoObservationBuilder once for the whole eval run."""
    from examples.train_real_dino import WristDinoFeatureExtractor, WristDinoObservationBuilder

    dino_extractor = WristDinoFeatureExtractor(
        model_name=args.dino_model_name,
        device=args.dino_device,
    )
    logging.info(
        "WristDinoFeatureExtractor loaded (model=%s, device=%s, feat_dim=%d)",
        args.dino_model_name,
        args.dino_device,
        dino_extractor.feature_dim,
    )
    return WristDinoObservationBuilder(dino_extractor)


def _append_diag(diag: dict[str, list[Any]] | None, key: str, value: Any) -> None:
    if diag is not None:
        diag[key].append(value)


def _future_waypoint_horizon_s(exec_timestamps: list[float], now: float) -> float:
    future = [ts for ts in exec_timestamps if ts > now]
    if not future:
        return 0.0
    return max(0.0, max(future) - now)


def _count_future_waypoints(exec_timestamps: list[float], now: float) -> int:
    return sum(1 for ts in exec_timestamps if ts > now)


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
    diagnostic_dir: Path | None = None,
) -> RolloutResult:
    """Run one train-aligned DSRL evaluation rollout."""
    from jaxrl2.utils.noise_utils import make_full_horizon_noise
    from tqdm import tqdm

    runtime_config = robot_io.runtime_config
    query_freq = int(args.query_freq)
    dt_step = 1.0 / float(runtime_config.control_frequency_hz)
    max_joint_delta = 0.2 * float(runtime_config.action_scale)

    # Warm up impedance control and seed the high-frequency interpolator.
    current_joints = np.asarray(env.get_observation()["robot_state"]["joint_positions"])
    env._robot.update_joints(current_joints, velocity=False, blocking=False)
    time.sleep(0.15)

    env._robot.start_trajectory_controller(float(runtime_config.controller_frequency))

    hold_joints = np.tile(current_joints, (4, 1))
    hold_offsets = [0.05, 0.20, 0.50, 1.00]
    env._robot.add_waypoints(
        hold_offsets,
        hold_joints.tolist(),
        max_joint_speed_rad_s=float(runtime_config.max_joint_speed_rad_s),
    )
    time.sleep(0.05)

    diag: dict[str, list[Any]] | None = None
    if diagnostic_dir is not None:
        diag = dict(
            joint_positions=[],
            gripper_positions=[],
            obs_timestamps=[],
            infer_steps=[],
            rl_noises=[],
            pi0_chunks=[],
            source_joint_positions=[],
            exec_positions=[],
            exec_timestamps=[],
            n_fresh=[],
            n_stale=[],
            infer_latency_s=[],
            obs_age_at_infer_s=[],
            future_waypoint_horizon_s=[],
            future_waypoint_count=[],
            infer_trigger_reasons=[],
            all_stale_steps=[],
        )

    scheduled_gripper_actions: list[tuple[float, float]] = []
    scheduled_arm_timestamps: list[float] = []
    ui.set_running(episode_id, completed, successes)

    start_time = time.time()
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    image_list: list[np.ndarray] = []
    decision: tuple[bool, str] | None = None
    env_steps = 0

    try:
        pbar = tqdm(
            total=runtime_config.max_timesteps,
            desc=f"dino eval episode {episode_id}",
            unit="step",
        )
        for t in range(runtime_config.max_timesteps):
            step_started = time.time()
            if time.time() - start_time >= float(args.max_duration_s):
                decision = (False, "timeout")
                break

            decision = ui.poll()
            if decision is not None:
                break

            try:
                raw_obs = env.get_observation()
                state_history = env._robot.get_state_history(n=100)
            except Exception:
                logging.exception("Observation failed at t=%d", t)
                raise

            curr_obs, obs_timestamp = extract_observation_train(
                runtime_config,
                raw_obs,
                state_history=state_history,
                wrist_obs_latency=float(runtime_config.wrist_camera_obs_latency),
                proprioceptive_latency=float(runtime_config.proprioceptive_latency),
                gripper_obs_latency=float(runtime_config.gripper_obs_latency),
            )
            if t == 0:
                logging.info("t_obs drift: %.1f ms", (time.time() - obs_timestamp) * 1000.0)

            _append_diag(diag, "joint_positions", curr_obs["joint_position"].copy())
            _append_diag(diag, "gripper_positions", curr_obs["gripper_position"].copy())
            _append_diag(diag, "obs_timestamps", float(obs_timestamp))

            now_before_infer = time.time()
            future_horizon_s = _future_waypoint_horizon_s(
                scheduled_arm_timestamps, now_before_infer
            )
            future_count = _count_future_waypoints(
                scheduled_arm_timestamps, now_before_infer
            )

            infer_trigger_reason = ""
            if args.dsrl_eval_timing_mode == "train":
                if t % query_freq == 0:
                    infer_trigger_reason = "query_freq"
            elif args.dsrl_eval_timing_mode == "low_watermark":
                if t == 0:
                    infer_trigger_reason = "initial"
                elif (
                    future_count <= args.min_future_actions
                    or future_horizon_s <= args.min_future_horizon_s
                ):
                    infer_trigger_reason = "low_watermark"
            else:
                raise ValueError(f"Unknown DSRL eval timing mode: {args.dsrl_eval_timing_mode}")

            if infer_trigger_reason:
                request_data = get_pi0_input_train(curr_obs, runtime_config, args.instruction)
                infer_start = time.time()
                obs_age_at_infer_s = infer_start - obs_timestamp

                obs_dict = obs_builder.build(curr_obs, request_data, policy_service)
                actions_noise = agent.eval_actions(obs_dict)
                _, noise = make_full_horizon_noise(actions_noise, agent.action_chunk_shape)
                response = policy_service.infer(request_data, noise=np.asarray(noise))
                actions = np.asarray(response["actions"])
                infer_done = time.time()

                abs_positions = integrate_joint_velocity_actions(
                    curr_obs["joint_position"],
                    actions,
                    max_joint_delta,
                )
                n_targets = min(query_freq, len(actions))
                chunk_actions = actions[:n_targets]
                chunk_positions = abs_positions[:n_targets]
                action_timestamps = action_timestamps_from_obs(
                    obs_timestamp,
                    n_targets,
                    dt_step,
                )

                is_new = action_timestamps > (time.time() + float(runtime_config.action_exec_latency))

                _append_diag(diag, "infer_steps", int(t))
                _append_diag(diag, "rl_noises", np.asarray(actions_noise))
                _append_diag(diag, "pi0_chunks", actions)
                _append_diag(diag, "source_joint_positions", curr_obs["joint_position"].copy())
                _append_diag(diag, "n_fresh", int(np.sum(is_new)))
                _append_diag(diag, "n_stale", int(np.sum(~is_new)))
                _append_diag(diag, "infer_latency_s", float(infer_done - infer_start))
                _append_diag(diag, "obs_age_at_infer_s", float(obs_age_at_infer_s))
                _append_diag(diag, "future_waypoint_horizon_s", float(future_horizon_s))
                _append_diag(diag, "future_waypoint_count", int(future_count))
                _append_diag(diag, "infer_trigger_reasons", str(infer_trigger_reason))

                if np.any(is_new):
                    arm_positions = chunk_positions[is_new]
                    arm_timestamps = action_timestamps[is_new]
                    arm_offsets = (
                        arm_timestamps - float(runtime_config.robot_action_latency)
                    ) - time.time()
                    try:
                        env._robot.add_waypoints(
                            arm_offsets.tolist(),
                            arm_positions.tolist(),
                            max_joint_speed_rad_s=float(runtime_config.max_joint_speed_rad_s),
                        )
                    except Exception:
                        logging.exception("add_waypoints failed at t=%d", t)
                        raise

                    if diag is not None:
                        for pos, ts in zip(arm_positions, arm_timestamps):
                            diag["exec_positions"].append(pos.copy())
                            diag["exec_timestamps"].append(float(ts))
                    scheduled_arm_timestamps.extend(float(ts) for ts in arm_timestamps)
                else:
                    logging.warning(
                        "run_rollout: all actions stale at t=%d "
                        "(infer_latency=%.3fs, horizon=%.3fs, obs_age=%.3fs, n_actions=%d)",
                        t,
                        infer_done - infer_start,
                        n_targets * dt_step,
                        time.time() - obs_timestamp,
                        n_targets,
                    )
                    _append_diag(diag, "all_stale_steps", int(t))

                scheduled_gripper_actions = [
                    (float(ts), float(binarize_and_clip_action(action)[-1]))
                    for ts, action in zip(action_timestamps.tolist(), chunk_actions)
                    if ts > time.time()
                ]

            curr_time = time.time()
            gripper_to_exec = None
            while (
                scheduled_gripper_actions
                and scheduled_gripper_actions[0][0] - float(runtime_config.gripper_action_latency) <= curr_time
            ):
                _, gripper_to_exec = scheduled_gripper_actions.pop(0)

            if gripper_to_exec is not None:
                try:
                    env._robot.update_gripper(gripper_to_exec, velocity=False, blocking=False)
                except Exception:
                    logging.exception("update_gripper failed at t=%d", t)
                    raise

            ui.update_camera_previews(
                wrist=curr_obs.get("wrist_image"),
                external=curr_obs.get(runtime_config.camera_to_use + "_image"),
            )
            image_list.append(select_video_frame_train(curr_obs, runtime_config))
            env_steps = t + 1
            ui.update_step(episode_id, env_steps, completed, successes)

            decision = ui.poll()
            if decision is not None:
                break

            elapsed = time.time() - step_started
            sleep_s = dt_step - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

            pbar.update(1)
    finally:
        try:
            pbar.close()
        except UnboundLocalError:
            pass
        try:
            env._robot.stop_trajectory_controller()
        except Exception:
            logging.exception("Failed to stop trajectory controller.")

    if decision is None:
        decision = (False, "timeout")

    success, failure_reason = decision
    duration_s = time.time() - start_time
    video_path = save_rollout_video(outputdir, episode_id, image_list)

    if diag is not None and diagnostic_dir is not None:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        npz_path = diagnostic_dir / f"episode_{episode_id:03d}.npz"

        def _arr(values: list[Any]) -> np.ndarray:
            return np.asarray(values) if values else np.asarray([])

        np.savez(
            npz_path,
            joint_positions=_arr(diag["joint_positions"]),
            gripper_positions=_arr(diag["gripper_positions"]),
            obs_timestamps=_arr(diag["obs_timestamps"]),
            infer_steps=_arr(diag["infer_steps"]),
            rl_noises=_arr(diag["rl_noises"]),
            pi0_chunks=_arr(diag["pi0_chunks"]),
            source_joint_positions=_arr(diag["source_joint_positions"]),
            exec_positions=_arr(diag["exec_positions"]),
            exec_timestamps=_arr(diag["exec_timestamps"]),
            n_fresh=_arr(diag["n_fresh"]),
            n_stale=_arr(diag["n_stale"]),
            infer_latency_s=_arr(diag["infer_latency_s"]),
            obs_age_at_infer_s=_arr(diag["obs_age_at_infer_s"]),
            future_waypoint_horizon_s=_arr(diag["future_waypoint_horizon_s"]),
            future_waypoint_count=_arr(diag["future_waypoint_count"]),
            infer_trigger_reasons=_arr(diag["infer_trigger_reasons"]),
            all_stale_steps=_arr(diag["all_stale_steps"]),
            success=np.asarray(success),
            duration_s=np.asarray(duration_s),
            episode_id=np.asarray(episode_id),
            action_scale=np.asarray(runtime_config.action_scale),
            max_joint_delta=np.asarray(max_joint_delta),
            dt_step=np.asarray(dt_step),
            query_freq=np.asarray(query_freq),
            dsrl_eval_timing_mode=np.asarray(args.dsrl_eval_timing_mode),
            min_future_actions=np.asarray(args.min_future_actions),
            min_future_horizon_s=np.asarray(args.min_future_horizon_s),
        )
        logging.info("Diagnostic data saved to %s", npz_path)

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
        description="Evaluate a train_real_dino StateSAC policy (+ pi0 server) on a real DROID robot."
    )
    parser.add_argument(
        "--restore_path",
        required=True,
        help="Path to the StateSACLearner checkpoint saved by train_real_dino.",
    )

    parser.add_argument("--instruction", default="put the spoon on the plate")
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--max_rollout_steps", default=600, type=int)
    parser.add_argument(
        "--max_duration_s",
        default=60.0,
        type=float,
        help="Safety wall-clock timeout per episode in seconds. Default: 60.",
    )

    parser.add_argument("--query_freq", default=DEFAULT_QUERY_FREQ, type=int)
    parser.add_argument(
        "--dsrl_eval_timing_mode",
        default="train",
        choices=("train", "low_watermark"),
        help=(
            "train: infer every query_freq steps with train-style per-step timing. "
            "low_watermark: infer whenever scheduled future arm targets run low."
        ),
    )
    parser.add_argument(
        "--min_future_actions",
        default=2,
        type=int,
        help="Low-watermark mode: infer when this many or fewer future arm targets remain.",
    )
    parser.add_argument(
        "--min_future_horizon_s",
        default=0.25,
        type=float,
        help="Low-watermark mode: infer when future arm target horizon is at or below this many seconds.",
    )
    parser.add_argument("--robot_action_latency", default=0.20, type=float)
    parser.add_argument("--gripper_action_latency", default=0.15, type=float)
    parser.add_argument("--action_exec_latency", default=0.0, type=float)
    parser.add_argument("--control_frequency_hz", default=DEFAULT_CONTROL_FREQUENCY, type=int)
    parser.add_argument("--controller_frequency", default=200.0, type=float)
    parser.add_argument(
        "--action_scale",
        default=0.5,
        type=float,
        help="Scale factor on DROID training max_joint_delta (0.2 rad/step). Default: 0.5.",
    )
    parser.add_argument(
        "--max_joint_speed_rad_s",
        default=0.3,
        type=float,
        help="NUC-side per-joint speed cap (rad/s) passed to add_waypoints. Default: 0.3.",
    )

    parser.add_argument("--wrist_camera_obs_latency", default=None, type=float)
    parser.add_argument("--proprioceptive_latency", default=None, type=float)
    parser.add_argument("--gripper_obs_latency", default=None, type=float)

    parser.add_argument("--external_camera", default="right", choices=("left", "right"))
    parser.add_argument("--use_wrist_camera", default=1, type=int, choices=(0, 1))
    parser.add_argument("--use_exterior_camera", default=0, type=int, choices=(0, 1))
    parser.add_argument("--left_camera_id", default="")
    parser.add_argument("--right_camera_id", default=DEFAULT_EXTERIOR_CAMERA_ID)
    parser.add_argument("--wrist_camera_id", default=DEFAULT_WRIST_CAMERA_ID)

    parser.add_argument("--policy_host", default="127.0.0.1")
    parser.add_argument("--policy_port", default=8000, type=int)

    parser.add_argument(
        "--rl_noise_horizon",
        default=8,
        type=int,
        help="RL noise horizon (must match the trained checkpoint). Default: 8.",
    )
    parser.add_argument(
        "--network_type",
        default="transformer",
        choices=("transformer", "mlp"),
        help="StateSAC network type (must match the trained checkpoint). Default: transformer.",
    )
    parser.add_argument("--hidden_dims", nargs="+", default=[1024, 1024, 1024], type=int)
    parser.add_argument("--transformer_dim", default=256, type=int)
    parser.add_argument("--transformer_depth", default=3, type=int)
    parser.add_argument("--transformer_num_heads", "--transformer_heads", default=4, type=int)

    parser.add_argument(
        "--dino_model",
        "--dino_model_name",
        dest="dino_model_name",
        default="facebook/dinov2-small",
        help="HuggingFace model name for wrist DINO-v2 feature extractor.",
    )
    parser.add_argument("--dino_device", default="auto")

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--outputdir", default=None)
    parser.add_argument(
        "--diagnostic_dir",
        default=None,
        help="If set, save per-episode .npz diagnostic files to this directory.",
    )
    return parser


def _runtime_default(field_name: str) -> Any:
    return RobotRuntimeConfig.__dataclass_fields__[field_name].default


def _build_runtime_config(args: argparse.Namespace) -> RobotRuntimeConfig:
    return RobotRuntimeConfig(
        external_camera=args.external_camera,
        left_camera_id=args.left_camera_id,
        right_camera_id=args.right_camera_id,
        wrist_camera_id=args.wrist_camera_id,
        max_timesteps=args.max_rollout_steps,
        control_frequency_hz=args.control_frequency_hz,
        use_wrist_camera=bool(args.use_wrist_camera),
        use_exterior_camera=bool(args.use_exterior_camera),
        allow_missing_cameras=True,
        wrist_camera_obs_latency=(
            _runtime_default("wrist_camera_obs_latency")
            if args.wrist_camera_obs_latency is None
            else args.wrist_camera_obs_latency
        ),
        proprioceptive_latency=(
            _runtime_default("proprioceptive_latency")
            if args.proprioceptive_latency is None
            else args.proprioceptive_latency
        ),
        gripper_obs_latency=(
            _runtime_default("gripper_obs_latency")
            if args.gripper_obs_latency is None
            else args.gripper_obs_latency
        ),
        robot_action_latency=args.robot_action_latency,
        gripper_action_latency=args.gripper_action_latency,
        action_exec_latency=args.action_exec_latency,
        action_scale=args.action_scale,
        controller_frequency=args.controller_frequency,
        max_joint_speed_rad_s=args.max_joint_speed_rad_s,
    )


def run_evaluation(args: argparse.Namespace) -> None:
    if args.eval_episodes <= 0:
        raise ValueError("--eval_episodes must be positive.")
    if args.max_rollout_steps <= 0:
        raise ValueError("--max_rollout_steps must be positive.")
    if args.max_duration_s <= 0:
        raise ValueError("--max_duration_s must be positive.")
    if args.query_freq <= 0:
        raise ValueError("--query_freq must be positive.")
    if args.query_freq > args.rl_noise_horizon:
        raise ValueError(
            f"--query_freq ({args.query_freq}) must be <= --rl_noise_horizon ({args.rl_noise_horizon})."
        )
    if args.min_future_actions < 0:
        raise ValueError("--min_future_actions must be non-negative.")
    if args.min_future_horizon_s < 0:
        raise ValueError("--min_future_horizon_s must be non-negative.")
    if args.control_frequency_hz <= 0:
        raise ValueError("--control_frequency_hz must be positive.")
    if not args.use_wrist_camera and not args.use_exterior_camera:
        raise ValueError("At least one camera must be enabled.")
    if not args.use_wrist_camera:
        raise ValueError("DSRL Wrist-DINO eval requires --use_wrist_camera 1.")

    runtime_config = _build_runtime_config(args)
    runtime_config.validate()

    logging.info(
        "RuntimeConfig: control=%dHz query_freq=%d max_rollout_steps=%d "
        "mode=%s min_future_actions=%d min_future_horizon_s=%.3fs "
        "robot_action_latency=%.3fs gripper_action_latency=%.3fs action_exec_latency=%.3fs "
        "controller_frequency=%.0fHz action_scale=%.2f max_joint_delta=%.3f "
        "max_joint_speed_rad_s=%.2f",
        runtime_config.control_frequency_hz,
        args.query_freq,
        runtime_config.max_timesteps,
        args.dsrl_eval_timing_mode,
        args.min_future_actions,
        args.min_future_horizon_s,
        runtime_config.robot_action_latency,
        runtime_config.gripper_action_latency,
        runtime_config.action_exec_latency,
        runtime_config.controller_frequency,
        runtime_config.action_scale,
        0.2 * runtime_config.action_scale,
        runtime_config.max_joint_speed_rad_s,
    )
    logging.info(
        "Camera config: external=%s use_wrist=%s use_exterior=%s "
        "wrist_id=%s left_id=%s right_id=%s",
        runtime_config.external_camera,
        runtime_config.use_wrist_camera,
        runtime_config.use_exterior_camera,
        runtime_config.wrist_camera_id,
        runtime_config.left_camera_id,
        runtime_config.right_camera_id,
    )
    logging.info(
        "Latencies: wrist=%.3fs proprioceptive=%.4fs gripper=%.5fs",
        runtime_config.wrist_camera_obs_latency,
        runtime_config.proprioceptive_latency,
        runtime_config.gripper_obs_latency,
    )

    logging.info("Loading StateSACLearner from %s ...", args.restore_path)
    agent = create_agent(args)
    if args.query_freq > agent.action_chunk_shape[0]:
        raise ValueError(
            f"--query_freq ({args.query_freq}) must be <= restored action horizon "
            f"({agent.action_chunk_shape[0]})."
        )

    logging.info("Loading WristDinoFeatureExtractor (%s) ...", args.dino_model_name)
    obs_builder = create_obs_builder(args)

    policy_service = PolicyService(PolicyServerConfig(host=args.policy_host, port=args.policy_port))
    metadata = policy_service.preflight()
    if metadata and "action_horizon" in metadata and "action_dim" in metadata:
        server_horizon = int(metadata["action_horizon"])
        server_dim = int(metadata["action_dim"])
        from examples.train_real_dino import PI0_NOISE_DIM

        noise_h, noise_d = agent.action_chunk_shape
        if server_horizon != noise_h or server_dim != noise_d:
            raise RuntimeError(
                f"pi0 server action shape ({server_horizon}, {server_dim}) does not match "
                f"RL noise shape ({noise_h}, {noise_d}). Restart the policy server from this repo."
            )
        logging.info("pi0 server action shape validated: (%d, %d)", server_horizon, server_dim)

    robot_io = RobotIO(runtime_config)
    robot_io.preflight()
    env = robot_io.env

    outputdir = resolve_outputdir(args.outputdir, prefix="dino_eval_real")
    csv_path = outputdir / "eval_results.csv"
    logging.info("Writing dino evaluation outputs to %s", outputdir)

    ui = HumanEvalUI(
        title="DINO Policy Evaluation",
        total_episodes=args.eval_episodes,
        preview_names=(("wrist", "Wrist"), ("external", "External")),
    )

    completed = 0
    successes = 0
    try:
        for episode_id in range(args.eval_episodes):
            if not ui.wait_for_start(episode_id, completed, successes):
                break

            reset_robot(env, reason=f"before episode {episode_id}")

            result = run_rollout(
                args,
                env,
                policy_service,
                robot_io,
                agent,
                obs_builder,
                ui,
                episode_id,
                completed,
                successes,
                outputdir,
                diagnostic_dir=Path(args.diagnostic_dir) if args.diagnostic_dir else None,
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
                "external_camera": args.external_camera,
                "wrist_camera_id": args.wrist_camera_id,
                "left_camera_id": args.left_camera_id,
                "right_camera_id": args.right_camera_id,
                "control_frequency_hz": args.control_frequency_hz,
                "query_freq": args.query_freq,
                "max_rollout_steps": args.max_rollout_steps,
                "dsrl_eval_timing_mode": args.dsrl_eval_timing_mode,
                "min_future_actions": args.min_future_actions,
                "min_future_horizon_s": args.min_future_horizon_s,
                "policy_host": args.policy_host,
                "policy_port": args.policy_port,
                "rl_noise_horizon": args.rl_noise_horizon,
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
