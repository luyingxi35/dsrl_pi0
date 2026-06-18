#!/usr/bin/env python3
"""Train-aligned DSRL + pi0_droid evaluation in ManiSkill simulation."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_triton_gemm_any=True" not in _xla_flags:
    _xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = _xla_flags

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.envs.mani_skill_client import ManiSkillRemoteEnv
from examples.sim_action_utils import RealTimeActionChunker, pi0_velocity_chunk_to_sim_actions
from examples.train_utils_sim_dino import (
    PI0_NOISE_DIM,
    STATE_DIM,
    SimDinoObservationBuilder,
    WristDinoFeatureExtractor,
    _extract_sim_obs,
    _obs_to_pi0_input,
)
from examples.utils.real_robot_common import (
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
    "restore_path",
    "checkpoint_path",
    "query_freq",
    "max_rollout_steps",
    "rl_noise_horizon",
    "action_scale",
    "network_type",
    "robofac_python",
]


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob("checkpoint*"))
    if not checkpoints:
        return None
    return checkpoints[-1]


def validate_restore_path(restore_path: str) -> None:
    run_dir = Path(restore_path)
    if not run_dir.exists():
        raise FileNotFoundError(f"--restore_path does not exist: {restore_path}")
    if not run_dir.is_dir():
        raise ValueError(f"--restore_path must be a training run directory: {restore_path}")
    if _latest_checkpoint(run_dir) is None:
        raise FileNotFoundError(
            f"No Flax checkpoint* files found under --restore_path: {restore_path}"
        )


def load_pi0_policy(checkpoint_path: str):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"pi0 checkpoint path does not exist: {checkpoint_path}")

    from openpi.policies import policy_config as openpi_policy_config
    from openpi.training import config as openpi_config

    pi0_cfg = openpi_config.get_config("pi0_droid")
    policy = openpi_policy_config.create_trained_policy(pi0_cfg, checkpoint_path)
    logging.info("Loaded pi0_droid from %s", checkpoint_path)
    return policy


def create_agent(args: argparse.Namespace):
    from jaxrl2.agents.state_sac.state_sac_learner import StateSACLearner

    sample_obs = {
        "state": np.zeros((1, STATE_DIM, 1), dtype=np.float32),
    }
    sample_action = np.zeros((1, args.rl_noise_horizon, PI0_NOISE_DIM), dtype=np.float32)

    train_kwargs = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        discount=0.99,
        tau=0.005,
        hidden_dims=tuple(args.hidden_dims),
        network_type=args.network_type,
        transformer_dim=args.transformer_dim,
        transformer_depth=args.transformer_depth,
        transformer_heads=args.transformer_heads,
        transformer_mlp_dim=args.transformer_mlp_dim,
        transformer_dropout=args.transformer_dropout,
        critic_reduction=args.critic_reduction,
        dropout_rate=args.dropout_rate,
        target_entropy=args.target_entropy,
        num_qs=args.num_qs,
        action_magnitude=args.action_magnitude,
    )
    agent = StateSACLearner(args.seed, sample_obs, sample_action, **train_kwargs)
    agent.restore_checkpoint(args.restore_path)
    logging.info(
        "StateSACLearner restored from %s (latest=%s, action_chunk_shape=%s)",
        args.restore_path,
        _latest_checkpoint(Path(args.restore_path)),
        agent.action_chunk_shape,
    )
    return agent


def run_rollout(
    args: argparse.Namespace,
    env: ManiSkillRemoteEnv,
    agent: Any,
    agent_dp: Any,
    obs_builder: SimDinoObservationBuilder,
    episode_id: int,
    outputdir: Path,
) -> tuple[RolloutResult, str]:
    from jaxrl2.utils.noise_utils import make_full_horizon_noise
    from tqdm import tqdm

    env_obs, _ = env.reset()
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    start_time = time.time()
    side_image_list: list[np.ndarray] = []
    wrist_image_list: list[np.ndarray] = []
    action_chunker = RealTimeActionChunker(
        action_horizon=args.action_horizon,
        action_dim=8,
        m=args.action_chunk_decay,
    )
    success = False
    failure_reason = "timeout"
    env_steps = 0

    pbar = tqdm(total=args.max_rollout_steps, desc=f"dino sim episode {episode_id}", unit="step")
    try:
        for t in range(args.max_rollout_steps):
            qpos, ext_rgb, wrist_rgb = _extract_sim_obs(env_obs)
            side_image_list.append(ext_rgb)
            wrist_image_list.append(wrist_rgb)

            pi0_obs = _obs_to_pi0_input(qpos, ext_rgb, wrist_rgb, args.instruction)
            obs_dict = obs_builder.build(qpos, ext_rgb, wrist_rgb, pi0_obs, agent_dp)
            state = np.asarray(obs_dict["state"])
            if state.shape != (1, STATE_DIM, 1):
                raise RuntimeError(f"Expected DSRL state shape (1, {STATE_DIM}, 1), got {state.shape}")

            actions_noise = agent.eval_actions(obs_dict)
            _, noise = make_full_horizon_noise(actions_noise, agent.action_chunk_shape)
            response = agent_dp.infer(pi0_obs, noise=np.asarray(noise))
            actions = np.asarray(response["actions"])
            if actions.ndim != 2 or actions.shape[-1] < 8:
                raise RuntimeError(f"Expected pi0 actions shape (H, >=8), got {actions.shape}")
            if len(actions) < args.action_horizon:
                raise RuntimeError(
                    f"--action_horizon ({args.action_horizon}) exceeds pi0 action horizon "
                    f"({len(actions)})."
                )
            sim_actions = pi0_velocity_chunk_to_sim_actions(
                qpos, actions[: args.action_horizon], action_scale=args.action_scale
            )
            action_8d = action_chunker.step(sim_actions)
            env_obs, _reward, terminated, truncated, info = env.step(action_8d)
            env_steps = t + 1
            pbar.update(1)

            # Workspace constraint check (pre/post-grasp bounding box)
            if bool(info.get("workspace_violated", False)):
                failure_reason = "workspace_" + info.get("phase", "pre_grasp")
                break

            if bool(info.get("success", False)):
                success = True
                failure_reason = ""
                break
            if bool(terminated) or bool(truncated):
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
    parser = argparse.ArgumentParser(
        description="Evaluate a train_sim_dino_v2 StateSAC policy with pi0_droid in ManiSkill simulation."
    )
    parser.add_argument("--restore_path", required=True)
    parser.add_argument("--instruction", default="pick up the peg and insert it vertically")
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--max_rollout_steps", default=600, type=int)
    parser.add_argument(
        "--query_freq",
        default=8,
        type=int,
        help=(
            "Legacy recorded query frequency. Sim eval now runs pi0 inference every "
            "control step and temporally ensembles overlapping chunks."
        ),
    )
    parser.add_argument("--action_horizon", default=8, type=int)
    parser.add_argument(
        "--action_chunk_decay",
        default=0.01,
        type=float,
        help="Exponential decay m for real-time action chunk ensembling.",
    )
    parser.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--robofac_python", default=DEFAULT_ROBOFAC_PYTHON)
    parser.add_argument("--workspace_bounds_path", default=None,
                        help="Path to workspace_bounds.json "
                             "(None = no workspace constraint)")
    parser.add_argument("--outputdir", default=None)

    parser.add_argument("--rl_noise_horizon", default=8, type=int)
    parser.add_argument("--network_type", default="transformer", choices=("transformer", "mlp"))
    parser.add_argument("--hidden_dims", nargs="+", default=[1024, 1024, 1024], type=int)
    parser.add_argument("--transformer_dim", default=256, type=int)
    parser.add_argument("--transformer_depth", default=3, type=int)
    parser.add_argument("--transformer_heads", "--transformer_num_heads", default=4, type=int)
    parser.add_argument("--transformer_mlp_dim", default=1024, type=int)
    parser.add_argument("--transformer_dropout", default=0.0, type=float)
    parser.add_argument("--critic_reduction", default="min")
    parser.add_argument("--dropout_rate", default=0.0, type=float)
    parser.add_argument("--target_entropy", default=0.0, type=float)
    parser.add_argument("--num_qs", default=2, type=int)
    parser.add_argument("--action_magnitude", default=2.0, type=float)
    parser.add_argument(
        "--action_scale",
        default=0.5,
        type=float,
        help="Scale on DROID max_joint_delta=0.2 rad/step. Default 0.5 => 0.1 rad/step.",
    )
    parser.add_argument("--dino_model", default="facebook/dinov2-small")
    parser.add_argument("--dino_device", default="auto")
    parser.add_argument("--seed", default=0, type=int)
    return parser


def run_evaluation(args: argparse.Namespace) -> None:
    if args.eval_episodes <= 0:
        raise ValueError("--eval_episodes must be positive.")
    if args.max_rollout_steps <= 0:
        raise ValueError("--max_rollout_steps must be positive.")
    if args.query_freq <= 0:
        raise ValueError("--query_freq must be positive.")
    if args.action_horizon <= 0:
        raise ValueError("--action_horizon must be positive.")
    if args.action_chunk_decay < 0:
        raise ValueError("--action_chunk_decay must be non-negative.")
    if args.rl_noise_horizon <= 0:
        raise ValueError("--rl_noise_horizon must be positive.")
    if args.action_scale <= 0:
        raise ValueError("--action_scale must be positive.")
    if args.action_horizon > args.rl_noise_horizon:
        raise ValueError(
            f"--action_horizon ({args.action_horizon}) must be <= --rl_noise_horizon "
            f"({args.rl_noise_horizon})."
        )

    validate_restore_path(args.restore_path)
    outputdir = resolve_outputdir(args.outputdir, prefix="dino_eval_sim")
    csv_path = outputdir / "eval_results.csv"
    logging.info("Writing dino sim evaluation outputs to %s", outputdir)

    env = ManiSkillRemoteEnv(robofac_python=args.robofac_python,
                             workspace_bounds_path=args.workspace_bounds_path)

    agent = create_agent(args)
    if args.action_horizon > agent.action_chunk_shape[0]:
        raise ValueError(
            f"--action_horizon ({args.action_horizon}) must be <= restored action horizon "
            f"({agent.action_chunk_shape[0]})."
        )

    agent_dp = load_pi0_policy(args.checkpoint_path)
    metadata = getattr(agent_dp, "metadata", {})
    if metadata:
        server_horizon = int(metadata.get("action_horizon", agent.action_chunk_shape[0]))
        server_dim = int(metadata.get("action_dim", agent.action_chunk_shape[1]))
        noise_h, noise_d = agent.action_chunk_shape
        if server_horizon != noise_h or server_dim != noise_d:
            raise RuntimeError(
                f"pi0 policy action shape ({server_horizon}, {server_dim}) does not match "
                f"RL noise shape ({noise_h}, {noise_d})."
            )

    dino_extractor = WristDinoFeatureExtractor(args.dino_model, args.dino_device)
    obs_builder = SimDinoObservationBuilder(dino_extractor)
    logging.info("Loaded DINO extractor %s on %s", args.dino_model, args.dino_device)

    completed = 0
    successes = 0
    try:
        for episode_id in range(args.eval_episodes):
            result, wrist_video_path = run_rollout(
                args, env, agent, agent_dp, obs_builder, episode_id, outputdir
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
                "wrist_video_path": wrist_video_path,
                "timestamp": result.timestamp,
                "instruction": args.instruction,
                "restore_path": args.restore_path,
                "checkpoint_path": args.checkpoint_path,
                "query_freq": args.query_freq,
                "max_rollout_steps": args.max_rollout_steps,
                "rl_noise_horizon": args.rl_noise_horizon,
                "action_scale": args.action_scale,
                "network_type": args.network_type,
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
        env.close()

    logging.info("DINO sim evaluation complete. Results: %s", csv_path)
    print(f"DINO sim evaluation complete. Results: {csv_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    run_evaluation(args)


if __name__ == "__main__":
    main()
