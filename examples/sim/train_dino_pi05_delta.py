#!/usr/bin/env python
"""Simulation DSRL training: PegInsertionVertical + pi05_droid + StateSAC + DINOv2.

Architecture is **identical to train_dino.py** — the only differences are:
  - uses pi05_droid openpi config (pi05=True, action_horizon=15)
  - uses pd_joint_delta_pos control (velocity → normalised per-step delta, no accumulation)
  - checkpoint loaded from /opt/yingxi/pi05_droid

Observation space mirrors train_dino.py (unchanged):
  STATE_DIM = 8 + 2048 + 384 = 2440
  PI0_NOISE_DIM = 32  (same as pi0; pi0.5 action_dim=32)

rl_noise_horizon must be set to 15 to match pi0.5 action_horizon.
query_freq can remain at 8 (execute 8 of 15 steps per pi0.5 call).
"""
import os
import sys
import tempfile
from functools import partial
from pathlib import Path

# ── XLA flag (must precede JAX import) ────────────────────────────────────────
_xla_flags = os.environ.get("XLA_FLAGS", "")
_xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = _xla_flags

import jax
import numpy as np
import tensorflow as tf
from gym.spaces import Box, Dict
import gymnasium as gym
from jax.experimental.compilation_cache import compilation_cache

from jaxrl2.agents.state_sac.state_sac_learner import StateSACLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.general_utils import add_batch_dim
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name

from examples.sim.train_utils_dino_delta import (
    STATE_DIM,
    PI0_NOISE_DIM,
    WristDinoFeatureExtractor,
    SimDinoObservationBuilder,
    trajwise_alternating_training_loop,
)
from examples.sim.envs.mani_skill_client import ManiSkillRemoteEnv

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openpi.training import config as openpi_config
from openpi.policies import policy_config as openpi_policy_config

_home_dir = os.environ["HOME"]
compilation_cache.initialize_cache(os.path.join(_home_dir, "jax_compilation_cache"))


def shard_batch(batch, sharding):
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class DummyEnv(gym.ObservationWrapper):
    """Stub for initialising SAC observation/action spaces."""

    def __init__(self, variant):
        self.observation_space = Dict({
            "state": Box(-np.inf, np.inf, shape=(STATE_DIM, 1), dtype=np.float32),
        })
        self.action_space = Box(
            -1, 1, shape=(variant.rl_noise_horizon, PI0_NOISE_DIM), dtype=np.float32
        )


def main(variant):
    if variant.query_freq <= 0:
        raise ValueError(f"--query_freq must be positive, got {variant.query_freq}.")
    if variant.query_freq > variant.rl_noise_horizon:
        raise ValueError(
            f"--query_freq ({variant.query_freq}) must be <= "
            f"--rl_noise_horizon ({variant.rl_noise_horizon})."
        )

    devices     = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    print(f"num_devices={num_devices}, batch_size={variant.batch_size}")
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    tf.config.set_visible_devices([], "GPU")

    kwargs = dict(variant["train_kwargs"])
    if kwargs.pop("cosine_decay", False):
        kwargs["decay_steps"] = variant.max_steps

    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    expname = create_exp_name(variant.prefix, seed=variant.seed)
    if getattr(variant, "suffix", ""):
        expname += f"_{variant.suffix}"

    outputdir = os.path.abspath(os.path.join(os.environ["EXP"], expname))
    variant.outputdir = outputdir
    os.makedirs(outputdir, exist_ok=True)
    print(f"Writing outputs to: {outputdir}")

    group_name       = variant.prefix + "_" + getattr(variant, "launch_group_id", "")
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(
        variant.prefix != "",
        variant,
        variant.wandb_project,
        experiment_id=expname,
        output_dir=wandb_output_dir,
        group_name=group_name,
    )

    # ── Environments (subprocess) ───────────────────────────────────────────────
    robofac_python = getattr(variant, "robofac_python",
                             "/opt/yingxi/envs/robofac/bin/python3")
    workspace_bounds = getattr(variant, "workspace_bounds_path", None)
    env      = ManiSkillRemoteEnv(robofac_python=robofac_python,
                                  workspace_bounds_path=workspace_bounds,
                                        control_mode="pd_joint_delta_pos")
    eval_env = ManiSkillRemoteEnv(robofac_python=robofac_python,
                                  workspace_bounds_path=workspace_bounds,
                                        control_mode="pd_joint_delta_pos")

    if not hasattr(variant, "max_timesteps"):
        variant.max_timesteps = 600
    variant.env_max_reward = 1

    # ── SAC agent (init BEFORE pi0.5 to claim cuSolver handle first) ────────────
    dummy_env     = DummyEnv(variant)
    sample_obs    = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print("sample obs shapes:", [(k, v.shape) for k, v in sample_obs.items()])
    print("sample action shape:", sample_action.shape)

    agent = StateSACLearner(variant.seed, sample_obs, sample_action, **kwargs)

    # ── pi05_droid policy (local, frozen) ────────────────────────────────────────
    pi05_cfg = openpi_config.get_config("pi05_droid")
    agent_dp = openpi_policy_config.create_trained_policy(
        pi05_cfg, variant.checkpoint_path
    )
    print(f"Loaded pi05_droid from: {variant.checkpoint_path}")

    # ── DINOv2 feature extractor ────────────────────────────────────────────────
    dino_extractor = WristDinoFeatureExtractor(variant.dino_model, variant.dino_device)
    obs_builder    = SimDinoObservationBuilder(dino_extractor)

    # ── Replay buffer ───────────────────────────────────────────────────────────
    online_buffer_size   = max(2 * variant.max_steps // max(variant.multi_grad_step, 1), 10000)
    online_replay_buffer = ReplayBuffer(
        dummy_env.observation_space,
        dummy_env.action_space,
        int(online_buffer_size),
    )
    online_replay_buffer.seed(variant.seed)

    # ── Training loop ───────────────────────────────────────────────────────────
    eval_csv = (
        os.path.join(outputdir, "eval_curve.csv")
        if getattr(variant, "eval_env_step_interval", -1) > 0
        else None
    )

    try:
        trajwise_alternating_training_loop(
            variant,
            agent,
            env,
            eval_env,
            online_replay_buffer,
            online_replay_buffer,
            wandb_logger,
            shard_fn=shard_fn,
            agent_dp=agent_dp,
            obs_builder=obs_builder,
            eval_env_step_interval=getattr(variant, "eval_env_step_interval", -1),
            stop_success_rate=getattr(variant, "stop_success_rate", 0.95),
            stop_window=getattr(variant, "stop_window", 2),
            eval_csv_path=eval_csv,
        )
    finally:
        env.close()
        eval_env.close()
