#! /usr/bin/env python
import dataclasses
import logging
import os
import tempfile
from functools import partial

import gymnasium as gym
import jax
import numpy as np
import tensorflow as tf
from droid.robot_env import RobotEnv
from gym.spaces import Box, Dict
from jax.experimental.compilation_cache import compilation_cache
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.general_utils import add_batch_dim
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
from openpi_client import websocket_client_policy as _websocket_client_policy

from examples.train_utils_real import trajwise_alternating_training_loop

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))

DEFAULT_DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass(frozen=True)
class PolicyServerConfig:
    host: str
    port: int


@dataclasses.dataclass(frozen=True)
class RobotRuntimeConfig:
    external_camera: str
    left_camera_id: str
    right_camera_id: str
    wrist_camera_id: str
    max_timesteps: int
    control_frequency_hz: int = DEFAULT_DROID_CONTROL_FREQUENCY

    @property
    def camera_to_use(self) -> str:
        return self.external_camera

    def validate(self) -> None:
        camera_ids = {
            "left_camera_id": self.left_camera_id,
            "right_camera_id": self.right_camera_id,
            "wrist_camera_id": self.wrist_camera_id,
        }
        missing = [name for name, value in camera_ids.items() if not value]
        if missing:
            raise ValueError(
                "DROID camera IDs must be set before running real rollouts. "
                f"Missing: {', '.join(missing)}. Fill these in examples/scripts/run_real.sh."
            )
        if self.wrist_camera_id in {self.left_camera_id, self.right_camera_id}:
            raise ValueError(f"The wrist camera must be different from the external camera IDs: {camera_ids}")


class PolicyService:
    def __init__(self, config: PolicyServerConfig):
        self._config = config
        self._client = _websocket_client_policy.WebsocketClientPolicy(
            host=config.host,
            port=config.port,
        )

    def preflight(self):
        metadata = self._client.get_server_metadata()
        logging.info("OpenPI policy server metadata: %s", metadata)
        return metadata

    def infer(self, obs, noise=None):
        return self._client.infer(obs, noise=noise)

    def get_prefix_rep(self, obs):
        return self._client.get_prefix_rep(obs)


class RobotIO:
    def __init__(self, runtime_config: RobotRuntimeConfig):
        self._runtime_config = runtime_config
        self._env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")

    @property
    def env(self):
        return self._env

    @property
    def runtime_config(self):
        return self._runtime_config

    def preflight(self):
        obs = self._env.get_observation()
        image_observations = obs["image"]
        required_cameras = [
            (self._runtime_config.left_camera_id, "left external"),
            (self._runtime_config.right_camera_id, "right external"),
            (self._runtime_config.wrist_camera_id, "wrist"),
        ]
        missing = [label for cam_id, label in required_cameras if not _has_camera_image(image_observations, cam_id)]
        if missing:
            raise RuntimeError(
                "DROID camera preflight failed. Missing image feeds for: "
                + ", ".join(missing)
            )
        return obs


def _has_camera_image(image_observations, camera_id: str) -> bool:
    candidates = {camera_id}
    if camera_id.startswith("realsense_"):
        candidates.add(camera_id.removeprefix("realsense_"))
    else:
        candidates.add(f"realsense_{camera_id}")
    return any(
        key == candidate or key.startswith(f"{candidate}_")
        for key in image_observations.keys()
        for candidate in candidates
    )

def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension.

    Args:
        batch: A pytree of arrays.
        sharding: A jax Sharding object with shape (num_devices,).
    """
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )

class DummyEnv(gym.ObservationWrapper):

    def __init__(self, variant):
        self.variant = variant
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.add_states:
            state_dim = 8 + 2024 # 8 is the proprioceptive state's dim, 2024 is the image representation's dim
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        self.action_space = Box(low=-1, high=1, shape=(1, 32,), dtype=np.float32) # 32 is the noise action space of pi 0

def main(variant):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    logging.info('num devices', num_devices)
    logging.info('batch size', variant.batch_size)
    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # prevent tensorflow from using GPUs
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

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name)

    policy_config = PolicyServerConfig(
        host=variant.policy_host,
        port=variant.policy_port,
    )
    policy_service = PolicyService(policy_config)
    policy_service.preflight()

    runtime_config = RobotRuntimeConfig(
        external_camera=variant.external_camera,
        left_camera_id=variant.left_camera_id,
        right_camera_id=variant.right_camera_id,
        wrist_camera_id=variant.wrist_camera_id,
        max_timesteps=variant.max_rollout_steps,
        control_frequency_hz=variant.control_frequency_hz,
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
    
    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)
    
    if variant.restore_path:
        logging.info('restoring from %s', variant.restore_path)
        agent.restore_checkpoint(variant.restore_path)

    online_buffer_size = 2 * variant.max_steps  // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
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
    )
 
