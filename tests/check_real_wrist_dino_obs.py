#!/usr/bin/env python
import argparse
import pathlib
import sys

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.train_real_dino import DINO_V2_SMALL_CLS_DIM
from examples.train_real_dino import PI0_VLM_EMBED_DIM
from examples.train_real_dino import PolicyServerConfig
from examples.train_real_dino import PolicyService
from examples.train_real_dino import RobotIO
from examples.train_real_dino import RobotRuntimeConfig
from examples.train_real_dino import STATE_DIM
from examples.train_real_dino import WristDinoFeatureExtractor
from examples.train_real_dino import WristDinoObservationBuilder
from examples.train_utils_real import _extract_observation
from examples.train_utils_real import get_pi0_input


class CachedPolicyService:
    def __init__(self, policy_service):
        self._policy_service = policy_service
        self.prefix_rep_response = None

    def get_prefix_rep(self, obs):
        if self.prefix_rep_response is None:
            self.prefix_rep_response = self._policy_service.get_prefix_rep(obs)
        return self.prefix_rep_response


class CachedDinoExtractor:
    def __init__(self, dino_extractor):
        self._dino_extractor = dino_extractor
        self.feature = None

    @property
    def feature_dim(self):
        return self._dino_extractor.feature_dim

    def encode(self, image):
        if self.feature is None:
            self.feature = self._dino_extractor.encode(image)
        return self.feature


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preflight real Wrist-DINO RL observation construction without stepping the robot."
    )
    parser.add_argument("--policy_host", default="127.0.0.1", help="OpenPI policy server host")
    parser.add_argument("--policy_port", default=8000, type=int, help="OpenPI policy server port")
    parser.add_argument("--external_camera", default="right", choices=["left", "right"], help="External camera for pi0 inputs")
    parser.add_argument("--left_camera_id", required=True, help="DROID left external camera ID")
    parser.add_argument("--right_camera_id", required=True, help="DROID right external camera ID")
    parser.add_argument("--wrist_camera_id", required=True, help="DROID wrist camera ID")
    parser.add_argument("--dino_model", default="facebook/dinov2-small", help="HuggingFace DINO-v2 model name")
    parser.add_argument("--dino_device", default="auto", help="DINO device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--instruction", default="put the spoon on the plate", help="Language instruction for pi0 prefix rep")
    parser.add_argument("--max_rollout_steps", default=200, type=int, help="Unused rollout horizon for runtime config")
    parser.add_argument("--control_frequency_hz", default=15, type=int, help="Unused control frequency for runtime config")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading DINO model: {args.dino_model} on {args.dino_device}")
    dino_extractor = CachedDinoExtractor(
        WristDinoFeatureExtractor(args.dino_model, args.dino_device)
    )

    print(f"Connecting to OpenPI policy server at {args.policy_host}:{args.policy_port}")
    policy_service = PolicyService(
        PolicyServerConfig(host=args.policy_host, port=args.policy_port)
    )
    metadata = policy_service.preflight()
    print(f"OpenPI metadata: {metadata}")
    cached_policy_service = CachedPolicyService(policy_service)

    runtime_config = RobotRuntimeConfig(
        external_camera=args.external_camera,
        left_camera_id=args.left_camera_id,
        right_camera_id=args.right_camera_id,
        wrist_camera_id=args.wrist_camera_id,
        max_timesteps=args.max_rollout_steps,
        control_frequency_hz=args.control_frequency_hz,
    )
    runtime_config.validate()

    print("Connecting to DROID robot runtime and reading one observation")
    robot_io = RobotIO(runtime_config)
    raw_obs = robot_io.preflight()
    curr_obs = _extract_observation(runtime_config, raw_obs)

    request_data = get_pi0_input(curr_obs, runtime_config, args.instruction)
    obs_builder = WristDinoObservationBuilder(dino_extractor)
    obs_dict = obs_builder.build(curr_obs, request_data, cached_policy_service)

    assert set(obs_dict.keys()) == {"state"}, f"Expected only state obs, got keys {sorted(obs_dict.keys())}"
    state = obs_dict["state"]
    assert state.shape == (1, STATE_DIM, 1), f"Expected state shape (1, {STATE_DIM}, 1), got {state.shape}"
    assert state.dtype == np.float32, f"Expected state dtype float32, got {state.dtype}"

    pi0_prefix, _ = cached_policy_service.prefix_rep_response
    pi0_vlm_embedding = np.asarray(pi0_prefix[:, -1, :]).reshape(-1)
    dino_feature = dino_extractor.feature

    joint_position = np.asarray(curr_obs["joint_position"])
    gripper_position = np.asarray(curr_obs["gripper_position"])

    assert joint_position.shape == (7,), f"Expected joint shape (7,), got {joint_position.shape}"
    assert gripper_position.shape == (1,), f"Expected gripper shape (1,), got {gripper_position.shape}"
    assert pi0_vlm_embedding.shape == (PI0_VLM_EMBED_DIM,), (
        f"Expected pi0 VLM embedding shape ({PI0_VLM_EMBED_DIM},), got {pi0_vlm_embedding.shape}"
    )
    assert dino_feature.shape == (DINO_V2_SMALL_CLS_DIM,), (
        f"Expected DINO feature shape ({DINO_V2_SMALL_CLS_DIM},), got {dino_feature.shape}"
    )

    print(f"joint shape: {joint_position.shape}")
    print(f"gripper shape: {gripper_position.shape}")
    print(f"pi0 VLM embedding shape: {pi0_vlm_embedding.shape}")
    print(f"DINO CLS shape: {dino_feature.shape}")
    print(f"state shape: {state.shape}")
    print(f"dtype: {state.dtype}")
    print("ok")


if __name__ == "__main__":
    main()
