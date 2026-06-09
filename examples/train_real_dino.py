#! /usr/bin/env python
import json
import logging
import os
import sys
import tempfile
from functools import partial
from pathlib import Path

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

from examples.train_utils_real import trajwise_alternating_training_loop

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.real_robot_common import (
    PolicyServerConfig,
    RobotRuntimeConfig,
    PolicyService,
    RobotIO,
    _has_camera_image,
)

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))

PROPRIO_DIM = 8
PI0_VLM_EMBED_DIM = 2048
DINO_V2_SMALL_CLS_DIM = 384
PI0_NOISE_DIM = 32
STATE_DIM = PROPRIO_DIM + PI0_VLM_EMBED_DIM + DINO_V2_SMALL_CLS_DIM


def _validate_policy_metadata(metadata, expected_horizon: int, expected_action_dim: int) -> None:
    missing = [key for key in ("action_horizon", "action_dim") if key not in metadata]
    if missing:
        raise RuntimeError(
            "OpenPI policy server metadata is missing "
            f"{', '.join(missing)}. Restart the policy server from this repo so "
            "the RL noise shape can be validated before real rollouts."
        )

    action_horizon = int(metadata["action_horizon"])
    action_dim = int(metadata["action_dim"])
    if action_horizon != expected_horizon or action_dim != expected_action_dim:
        raise RuntimeError(
            "OpenPI policy server action shape does not match RL noise shape: "
            f"server=({action_horizon}, {action_dim}), "
            f"RL=({expected_horizon}, {expected_action_dim})."
        )


def shard_batch(batch, sharding):
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class WristDinoFeatureExtractor:
    def __init__(self, model_name: str, device: str):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "Wrist-DINO real training requires torch and transformers. "
                "Install openpi or add those packages to the environment."
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._device = torch.device(device)
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device)
        self._model.eval()

    @property
    def feature_dim(self) -> int:
        return DINO_V2_SMALL_CLS_DIM

    def encode(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected wrist RGB image with shape (H, W, 3), got {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        feature = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()[0].astype(np.float32)
        if feature.shape != (self.feature_dim,):
            raise RuntimeError(f"Expected DINO CLS feature shape ({self.feature_dim},), got {feature.shape}")
        return feature


class WristDinoObservationBuilder:
    def __init__(self, dino_extractor: WristDinoFeatureExtractor):
        self._dino_extractor = dino_extractor

    def build(self, curr_obs, request_data, policy_service):
        img_rep_pi0, _ = policy_service.get_prefix_rep(request_data)
        img_rep_pi0 = np.asarray(img_rep_pi0[:, -1, :]).reshape(-1).astype(np.float32)
        if img_rep_pi0.shape != (PI0_VLM_EMBED_DIM,):
            raise RuntimeError(f"Expected pi0 VLM embedding shape ({PI0_VLM_EMBED_DIM},), got {img_rep_pi0.shape}")

        wrist_dino_cls = self._dino_extractor.encode(curr_obs["wrist_image"])
        state = np.concatenate([
            np.asarray(curr_obs["joint_position"], dtype=np.float32),
            np.asarray(curr_obs["gripper_position"], dtype=np.float32),
            img_rep_pi0,
            wrist_dino_cls,
        ]).astype(np.float32)
        if state.shape != (STATE_DIM,):
            raise RuntimeError(f"Expected state shape ({STATE_DIM},), got {state.shape}")
        return {"state": state[np.newaxis, ..., np.newaxis]}


class DummyEnv(gym.ObservationWrapper):
    def __init__(self, variant):
        self.variant = variant
        obs_dict = {
            'state': Box(low=-np.inf, high=np.inf, shape=(STATE_DIM, 1), dtype=np.float32),
        }
        self.observation_space = Dict(obs_dict)
        self.action_space = Box(low=-1, high=1, shape=(variant.rl_noise_horizon, PI0_NOISE_DIM), dtype=np.float32)


def main(variant):
    if variant.query_freq <= 0:
        raise ValueError(f"--query_freq must be positive, got {variant.query_freq}.")
    if variant.query_freq > variant.rl_noise_horizon:
        raise ValueError(
            f"--query_freq ({variant.query_freq}) must be <= --rl_noise_horizon ({variant.rl_noise_horizon})."
        )

    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    logging.info('num devices', num_devices)
    logging.info('batch size', variant.batch_size)
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    tf.config.set_visible_devices([], "GPU")

    kwargs = dict(variant['train_kwargs'])
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

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(
        variant.prefix != '',
        variant,
        variant.wandb_project,
        experiment_id=expname,
        output_dir=wandb_output_dir,
        group_name=group_name,
    )

    dino_extractor = WristDinoFeatureExtractor(variant.dino_model, variant.dino_device)
    obs_builder = WristDinoObservationBuilder(dino_extractor)

    policy_config = PolicyServerConfig(
        host=variant.policy_host,
        port=variant.policy_port,
    )
    policy_service = PolicyService(policy_config)
    policy_metadata = policy_service.preflight()
    _validate_policy_metadata(policy_metadata, variant.rl_noise_horizon, PI0_NOISE_DIM)

    runtime_config = RobotRuntimeConfig(
        external_camera=variant.external_camera,
        left_camera_id=variant.left_camera_id,
        right_camera_id=variant.right_camera_id,
        wrist_camera_id=variant.wrist_camera_id,
        max_timesteps=variant.max_rollout_steps,
        control_frequency_hz=variant.control_frequency_hz,
        allow_missing_cameras=True,
        action_scale=variant.action_scale,
        max_joint_speed_rad_s=variant.max_joint_speed_rad_s,
        robot_action_latency=variant.robot_action_latency,
        controller_frequency=variant.controller_frequency,
    )
    runtime_config.validate()
    logging.info("Initializing DROID client runtime...")
    robot_io = RobotIO(runtime_config)
    robot_io.preflight()
    env = robot_io.env
    eval_env = env
    logging.info("Created DROID-aligned robot client runtime.")

    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    logging.info('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    logging.info('sample action shape', sample_action.shape)

    agent = StateSACLearner(variant.seed, sample_obs, sample_action, **kwargs)

    # ── Checkpoint restore (simple path) ─────────────────────────────────────
    if variant.restore_path and variant.resume_from:
        raise ValueError("--restore_path and --resume_from are mutually exclusive.")
    if variant.restore_path:
        logging.info('restoring from %s', variant.restore_path)
        agent.restore_checkpoint(variant.restore_path)

    online_buffer_size = 2 * variant.max_steps // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)

    # ── Resume from a previous outputdir ─────────────────────────────────────
    initial_step = 0
    initial_total_env_steps = 0
    initial_total_num_traj = 0
    initial_completed = 0
    initial_successes = 0

    if variant.resume_from:
        from flax.training import checkpoints as flax_ckpts
        resume_dir = variant.resume_from
        if not os.path.isdir(resume_dir):
            raise ValueError(f"--resume_from directory does not exist: {resume_dir}")

        # Re-point outputdir to the existing experiment directory.
        outputdir = resume_dir
        variant.outputdir = outputdir
        logging.info('Resuming into existing outputdir: %s', outputdir)

        # 1. Restore agent from the latest Flax checkpoint.
        latest = flax_ckpts.latest_checkpoint(outputdir, prefix='checkpoint')
        if latest is None:
            raise RuntimeError(f"No checkpoint found in {outputdir}. "
                               "Use --restore_path for a one-off weight restore.")
        agent.restore_checkpoint(outputdir)
        initial_step = int(os.path.basename(latest).split('checkpoint_')[-1])
        logging.info('Restored agent from step %d (%s)', initial_step, latest)

        # 2. Restore training counters from training_state.json.
        state_path = os.path.join(outputdir, 'training_state.json')
        if os.path.exists(state_path):
            with open(state_path) as f:
                saved_state = json.load(f)
            initial_step            = saved_state.get('i',                initial_step)
            initial_total_env_steps = saved_state.get('total_env_steps',  0)
            initial_total_num_traj  = saved_state.get('total_num_traj',   0)
            initial_completed       = saved_state.get('completed',        0)
            initial_successes       = saved_state.get('successes',        0)
            logging.info('Restored training state: %s', saved_state)
        else:
            logging.warning('training_state.json not found in %s; counters start at 0.', outputdir)

        # 3. Restore replay buffer (optional — proceed if missing).
        buf_path = os.path.join(outputdir, 'replay_buffer.pkl')
        if os.path.exists(buf_path):
            online_replay_buffer.restore(buf_path)
            logging.info('Restored replay buffer (%d transitions) from %s',
                         len(online_replay_buffer), buf_path)
        else:
            logging.warning('replay_buffer.pkl not found in %s; starting with empty buffer.', outputdir)

    trajwise_alternating_training_loop(
        variant,
        agent,
        env,
        eval_env,
        online_replay_buffer,
        replay_buffer,
        wandb_logger,
        shard_fn=shard_fn,
        policy_service=policy_service,
        robot_io=robot_io,
        obs_builder=obs_builder,
        initial_step=initial_step,
        initial_total_env_steps=initial_total_env_steps,
        initial_total_num_traj=initial_total_num_traj,
        initial_completed=initial_completed,
        initial_successes=initial_successes,
    )
