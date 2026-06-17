#!/usr/bin/env python
"""Simulation DSRL training: PegInsertionVertical-v1 + pi0_droid (local) + StateSAC + DINOv2.

Architecture mirrors train_real_dino.py:
  - StateSACLearner with Transformer architecture
  - Observation: proprio(8) + pi0 VLM embed(2048) + DINOv2-small CLS(384) = STATE_DIM=2440
  - Action:      RL noise chunk (rl_noise_horizon=8, noise_dim=32) fed into local pi0_droid
  - Env:         RoboFPE PegInsertionVertical-v1 (ManiSkill2/SAPIEN, Panda robot)
  - Success:     rule-based has_peg_inserted() from task definition

Differences from train_real_dino.py (no hardware):
  - No HighFreqController / latency compensation → direct env.step()
  - No HumanEvalUI → success judged automatically by env
  - No PolicyService → pi0 inference is local (agent_dp.infer / get_prefix_rep)
  - Wrist camera image = zeros (sim has only base_camera)

XLA Triton GEMM optimisation (same as train_sim.py, improves steps/sec ~30%).
"""
import os
import sys
import tempfile
from functools import partial
from pathlib import Path

# ── XLA flag: must be set before JAX is imported ───────────────────────────────
_xla_flags = os.environ.get("XLA_FLAGS", "")
_xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = _xla_flags

import gymnasium as gym
import jax
import numpy as np
import tensorflow as tf
from gym.spaces import Box, Dict
from jax.experimental.compilation_cache import compilation_cache

from jaxrl2.agents.state_sac.state_sac_learner import StateSACLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.general_utils import add_batch_dim
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name

from examples.train_utils_sim_dino import (
    STATE_DIM,
    PI0_NOISE_DIM,
    WristDinoFeatureExtractor,
    SimDinoObservationBuilder,
    trajwise_alternating_training_loop,
)

# ── Ensure RoboFPE and repo root are on sys.path ───────────────────────────────
_REPO_ROOT    = Path(__file__).resolve().parents[1]
_ROBOFPE_ROOT = _REPO_ROOT.parent / "RoboFPE"

for _p in [str(_REPO_ROOT), str(_ROBOFPE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# mani_envs registered inside main() to avoid sapien import at module level

from openpi.training import config as openpi_config
from openpi.policies import policy_config as openpi_policy_config

# JAX compilation cache
_home_dir = os.environ["HOME"]
compilation_cache.initialize_cache(os.path.join(_home_dir, "jax_compilation_cache"))


def shard_batch(batch, sharding):
    """Shard batch leading dimension across devices (same as other train scripts)."""
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class DummyEnv(gym.ObservationWrapper):
    """Minimal env stub for initialising the SAC agent's observation/action spaces."""

    def __init__(self, variant):
        self.observation_space = Dict({
            "state": Box(
                low=-np.inf, high=np.inf,
                shape=(STATE_DIM, 1),
                dtype=np.float32,
            ),
        })
        self.action_space = Box(
            low=-1, high=1,
            shape=(variant.rl_noise_horizon, PI0_NOISE_DIM),
            dtype=np.float32,
        )


def main(variant):
    # ── Sanity checks ───────────────────────────────────────────────────────────
    if variant.query_freq <= 0:
        raise ValueError(f"--query_freq must be positive, got {variant.query_freq}.")
    if variant.query_freq > variant.rl_noise_horizon:
        raise ValueError(
            f"--query_freq ({variant.query_freq}) must be <= "
            f"--rl_noise_horizon ({variant.rl_noise_horizon})."
        )

    # ── Multi-device setup ──────────────────────────────────────────────────────
    devices     = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0, (
        f"batch_size ({variant.batch_size}) must be divisible by num_devices ({num_devices})"
    )
    print(f"num_devices={num_devices}, batch_size={variant.batch_size}")
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # Prevent TensorFlow from allocating GPU memory
    tf.config.set_visible_devices([], "GPU")

    # ── Experiment naming & output directory ────────────────────────────────────
    kwargs = dict(variant["train_kwargs"])
    if kwargs.pop("cosine_decay", False):
        kwargs["decay_steps"] = variant.max_steps

    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    expname = create_exp_name(variant.prefix, seed=variant.seed)
    if variant.suffix:
        expname += f"_{variant.suffix}"

    outputdir = os.path.join(os.environ["EXP"], expname)
    variant.outputdir = outputdir
    os.makedirs(outputdir, exist_ok=True)
    print(f"Writing outputs to: {outputdir}")

    # ── WandB logger ────────────────────────────────────────────────────────────
    group_name       = variant.prefix + "_" + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(
        variant.prefix != "",
        variant,
        variant.wandb_project,
        experiment_id=expname,
        output_dir=wandb_output_dir,
        group_name=group_name,
    )

    # ── Simulation environments ─────────────────────────────────────────────────
    # Compatibility patch: mani_skill 3.0.1 removed noTableSceneBuilder.
    # Patch before mani_envs.__init__ imports all tasks (which need this symbol).
    import mani_skill.utils.scene_builder.table as _mst
    from mani_skill.utils.scene_builder.table import TableSceneBuilder as _TSB
    if not hasattr(_mst, "noTableSceneBuilder"):
        _mst.noTableSceneBuilder          = _TSB
        _mst.noTableSceneBuilder_microwave = _TSB

    # Register PegInsertionVertical-v1 (deferred to avoid sapien at import time)
    import mani_envs.tasks.task_PegInsertionVertical  # noqa: F401
    _env_kwargs = dict(
        obs_mode="rgb+state",
        render_mode="rgb_array",
        num_envs=1,
    )
    env      = gym.make("PegInsertionVertical-v1", **_env_kwargs)
    eval_env = gym.make("PegInsertionVertical-v1", **_env_kwargs)

    # Expose max_timesteps on variant (used by collect_traj and _perform_eval)
    if not hasattr(variant, "max_timesteps"):
        variant.max_timesteps = 300   # matches PegInsertionVertical-v1 max_episode_steps
    variant.env_max_reward = 1

    # ── pi0_droid policy (local, frozen) ────────────────────────────────────────
    pi0_cfg  = openpi_config.get_config("pi0_droid")
    agent_dp = openpi_policy_config.create_trained_policy(
        pi0_cfg, variant.checkpoint_path
    )
    print(f"Loaded pi0_droid policy from: {variant.checkpoint_path}")

    # ── DINOv2 feature extractor ────────────────────────────────────────────────
    dino_extractor = WristDinoFeatureExtractor(variant.dino_model, variant.dino_device)
    obs_builder    = SimDinoObservationBuilder(dino_extractor)

    # ── SAC agent (StateSAC + Transformer) ──────────────────────────────────────
    dummy_env     = DummyEnv(variant)
    sample_obs    = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print("sample obs shapes:",    [(k, v.shape) for k, v in sample_obs.items()])
    print("sample action shape:",  sample_action.shape)

    agent = StateSACLearner(variant.seed, sample_obs, sample_action, **kwargs)

    # ── Replay buffer ───────────────────────────────────────────────────────────
    # Size: 2x max gradient steps worth of transitions (same headroom as train_real_dino.py)
    online_buffer_size   = max(2 * variant.max_steps // max(variant.multi_grad_step, 1), 10000)
    online_replay_buffer = ReplayBuffer(
        dummy_env.observation_space,
        dummy_env.action_space,
        int(online_buffer_size),
    )
    online_replay_buffer.seed(variant.seed)

    # ── Training loop ───────────────────────────────────────────────────────────
    # CSV path for sweep: one file per run, written to outputdir
    eval_csv = os.path.join(outputdir, "eval_curve.csv") if getattr(variant, "eval_env_step_interval", -1) > 0 else None

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
