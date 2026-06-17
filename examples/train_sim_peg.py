#! /usr/bin/env python
"""Training script for PegInsertionVertical in LIBERO (variant.env == 'libero_peg').

Mirrors train_sim.py exactly except:
  1. Registers the custom peg-insertion LIBERO objects and problem class.
  2. Adds  elif variant.env == 'libero_peg':  environment loading branch.
  3. DummyEnv handles state_dim = 8 for libero_peg (Panda 7-DOF + gripper).

All training logic (SAC, buffer, training loop) is identical to train_sim.py.
Neither train_sim.py nor any other existing file is modified.
"""
import os

# XLA Triton GEMM optimisation (same as train_sim.py)
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import pathlib
import copy
import tempfile
from functools import partial

import jax
import numpy as np
import tensorflow as tf
from gym.spaces import Dict, Box
import gymnasium as gym
import gym_aloha

from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim
from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
from jax.experimental.compilation_cache import compilation_cache

from examples.train_utils_sim import trajwise_alternating_training_loop
from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi.shared import download

# Patch openpi.transforms.Normalize._normalize to handle shape mismatch:
# pi05_base stats are 8-D; LiberoInputs pads state to 32-D.
# Use pure JAX ops so the patch is JIT-compatible.
import openpi.transforms as _openpit
import jax.numpy as _jnp
def _safe_normalize(self, x, stats):
    x = _jnp.asarray(x)
    m = _jnp.asarray(stats.mean)
    s = _jnp.asarray(stats.std)
    if x.shape != m.shape:
        if x.size > m.size:
            m = _jnp.concatenate([m.ravel(), _jnp.zeros(x.size - m.size)])
            s = _jnp.concatenate([s.ravel(), _jnp.ones(x.size - s.size)])
        elif x.size < m.size:
            m = m.ravel()[:x.size]
            s = s.ravel()[:x.size]
        m = m.reshape(x.shape)
        s = s.reshape(x.shape)
    return (x - m) / (s + 1e-6)
_openpit.Normalize._normalize = _safe_normalize  # JIT-compatible patch


# ── Register peg-insertion LIBERO objects + problem class ────────────────────
# These imports fire @register_object / @register_problem decorators so the
# classes appear in OBJECTS_DICT / TASK_MAPPING before OffScreenRenderEnv loads.
# No existing file is modified.
from libero.libero.envs.objects.peg_insertion_objects import Peg, BoxWithHole          # noqa: F401
from libero.libero.envs.problems.peg_insertion_tabletop import (                       # noqa: F401
    PegInsertion_Tabletop_Manipulation,
)

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))

# Absolute path to the custom BDDL file (inside the LIBERO submodule)
_PEG_BDDL = str(
    pathlib.Path(__file__).resolve().parents[1]
    / "LIBERO" / "libero" / "libero" / "bddl_files"
    / "peg_insertion" / "TABLETOP_peg_insertion_vertical.bddl"
)


def _get_libero_env(task, resolution, seed):
    """Standard LIBERO env loader (kept for 'libero' variant)."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task_description = task.language
    task_bddl_file = (pathlib.Path(get_libero_path("bddl_files"))
                      / task.problem_folder / task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file,
                "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _get_peg_libero_env(resolution, seed):
    """Load the custom PegInsertion LIBERO environment."""
    from libero.libero.envs import OffScreenRenderEnv
    env_args = {"bddl_file_name": _PEG_BDDL,
                "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    task_description = "pick up the peg and insert it vertically into the hole"
    return env, task_description


def shard_batch(batch, sharding):
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))),
        batch,
    )


class DummyEnv(gym.ObservationWrapper):
    """Dummy env for initialising SAC observation / action spaces."""

    def __init__(self, variant):
        self.variant = variant
        self.image_shape = (variant.resize_image, variant.resize_image,
                            3 * variant.num_cameras, 1)
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.add_states:
            if variant.env in ('libero', 'libero_peg'):
                state_dim = 8    # Panda: 7 joints + 1 gripper avg
            elif variant.env == 'aloha_cube':
                state_dim = 14
            else:
                state_dim = 8
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        self.action_space = Box(low=-1, high=1, shape=(1, 32), dtype=np.float32)


def main(variant):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    print('num devices', num_devices)
    print('batch size', variant.batch_size)

    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    tf.config.set_visible_devices([], "GPU")

    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps

    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)

    outputdir = os.path.join(os.environ['EXP'], expname)
    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir)

    # ── Environment loading ───────────────────────────────────────────────────
    if variant.env == 'libero':
        from libero.libero import benchmark
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["libero_90"]()
        task_id = 57
        task = task_suite.get_task(task_id)
        env, task_description = _get_libero_env(task, 256, variant.seed)
        eval_env = env
        variant.task_description = task_description
        variant.env_max_reward = 1
        variant.max_timesteps = 400

    elif variant.env == 'libero_peg':
        env, task_description = _get_peg_libero_env(256, variant.seed)
        eval_env = env
        variant.task_description = task_description
        variant.env_max_reward = 1
        variant.max_timesteps = 300    # matches PegInsertionVertical max_episode_steps

    elif variant.env == 'aloha_cube':
        from gymnasium.envs.registration import register
        register(
            id="gym_aloha/AlohaTransferCube-v0",
            entry_point="gym_aloha.env:AlohaEnv",
            max_episode_steps=400,
            nondeterministic=True,
            kwargs={"obs_type": "pixels", "task": "transfer_cube"},
        )
        env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos",
                       render_mode="rgb_array")
        eval_env = copy.deepcopy(env)
        variant.env_max_reward = 4
        variant.max_timesteps = 400

    else:
        raise NotImplementedError(f"Unknown env: {variant.env}")

    # ── WandB ─────────────────────────────────────────────────────────────────
    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(
        variant.prefix != '', variant, variant.wandb_project,
        experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name)

    # ── SAC agent ─────────────────────────────────────────────────────────────
    dummy_env     = DummyEnv(variant)
    sample_obs    = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)

    # ── SAC agent (init BEFORE pi0 to claim cuSolver handle first) ──────────
    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)

    # ── pi0 policy ────────────────────────────────────────────────────────────
    if variant.env in ('libero', 'libero_peg'):
        pi0_config     = openpi_config.get_config("pi0_libero")
        checkpoint_dir = download.maybe_download(
            "/opt/yingxi/pi0_droid")
    elif variant.env == 'aloha_cube':
        pi0_config     = openpi_config.get_config("pi0_aloha_sim")
        checkpoint_dir = download.maybe_download(
            "/opt/yingxi/pi0_droid")
    else:
        raise NotImplementedError()

    agent_dp = policy_config.create_trained_policy(pi0_config, checkpoint_dir)
    print("Loaded pi0 policy from %s", checkpoint_dir)

    # ── Replay buffer ─────────────────────────────────────────────────────────
    online_buffer_size   = variant.max_steps // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space,
                                        dummy_env.action_space,
                                        int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)

    # libero_peg obs format is identical to libero_90;
    # rename so train_utils_sim obs helpers (obs_to_img etc.) work correctly.
    if getattr(variant, 'env', '') == 'libero_peg':
        variant.env = 'libero'

    # ── CSV logging wrapper (env_steps vs success_rate) ──────────────────────
    import csv as _csv
    _csv_path = os.path.join(outputdir, 'eval_curve.csv')
    with open(_csv_path, 'w', newline='') as _f:
        _csv.writer(_f).writerow(['env_steps', 'success_rate'])
    _env_steps_buf = [0]
    _orig_log = wandb_logger.log
    def _log_with_csv(data, step=None):
        if 'env_steps' in data:
            _env_steps_buf[0] = int(data['env_steps'])
        if 'evaluation/success_rate' in data:
            with open(_csv_path, 'a', newline='') as _f:
                _csv.writer(_f).writerow([_env_steps_buf[0], float(data['evaluation/success_rate'])])
        return _orig_log(data, step=step)
    wandb_logger.log = _log_with_csv
    print(f'[sweep] eval_curve.csv -> {_csv_path}')

    # ── Training loop ─────────────────────────────────────────────────────────
    trajwise_alternating_training_loop(
        variant, agent, env, eval_env,
        online_replay_buffer, replay_buffer, wandb_logger,
        shard_fn=shard_fn, agent_dp=agent_dp)
