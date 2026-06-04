import dataclasses

import numpy as np
import pytest

from examples.train_real_dino import RobotRuntimeConfig
from examples.train_utils_real import _extract_observation
from examples.train_utils_real import get_pi0_input


@dataclasses.dataclass(frozen=True)
class DummyRuntimeConfig:
    external_camera: str = "right"
    use_wrist_camera: bool = True
    use_exterior_camera: bool = False

    @property
    def camera_to_use(self):
        return self.external_camera


def _make_obs():
    return {
        "joint_position": np.zeros((7,), dtype=np.float32),
        "gripper_position": np.zeros((1,), dtype=np.float32),
        "wrist_image": np.full((8, 8, 3), 127, dtype=np.uint8),
        "right_image": np.full((8, 8, 3), 255, dtype=np.uint8),
        "wrist_image_present": True,
        "right_image_present": True,
    }


def _make_raw_obs(image_observations):
    return {
        "image": image_observations,
        "robot_state": {
            "cartesian_position": np.zeros((6,), dtype=np.float32),
            "joint_positions": np.zeros((7,), dtype=np.float32),
            "gripper_position": 0.0,
        },
    }


def test_get_pi0_input_wrist_only_omits_exterior_image():
    request_data = get_pi0_input(
        _make_obs(),
        DummyRuntimeConfig(use_wrist_camera=True, use_exterior_camera=False),
        "pick the blue peg",
    )

    assert "observation/wrist_image_left" in request_data
    assert "observation/exterior_image_1_left" not in request_data
    assert request_data["prompt"] == "pick the blue peg"


def test_get_pi0_input_wrist_and_exterior_includes_both_images():
    request_data = get_pi0_input(
        _make_obs(),
        DummyRuntimeConfig(use_wrist_camera=True, use_exterior_camera=True),
        "pick the blue peg",
    )

    assert "observation/wrist_image_left" in request_data
    assert "observation/exterior_image_1_left" in request_data


def test_wrist_only_runtime_accepts_missing_exterior_ids():
    config = RobotRuntimeConfig(
        external_camera="right",
        left_camera_id="",
        right_camera_id="",
        wrist_camera_id="17396664",
        max_timesteps=10,
        use_wrist_camera=True,
        use_exterior_camera=False,
    )

    config.validate()


def test_wrist_only_runtime_ignores_unselected_exterior_id_duplicates():
    config = RobotRuntimeConfig(
        external_camera="right",
        left_camera_id="",
        right_camera_id="17396664",
        wrist_camera_id="17396664",
        max_timesteps=10,
        use_wrist_camera=True,
        use_exterior_camera=False,
    )

    config.validate()


def test_exterior_runtime_requires_selected_exterior_id():
    config = RobotRuntimeConfig(
        external_camera="right",
        left_camera_id="241122302552",
        right_camera_id="",
        wrist_camera_id="17396664",
        max_timesteps=10,
        use_wrist_camera=True,
        use_exterior_camera=True,
    )

    with pytest.raises(ValueError, match="right_camera_id"):
        config.validate()


def test_runtime_rejects_all_pi0_cameras_disabled():
    config = RobotRuntimeConfig(
        external_camera="right",
        left_camera_id="",
        right_camera_id="",
        wrist_camera_id="17396664",
        max_timesteps=10,
        use_wrist_camera=False,
        use_exterior_camera=False,
    )

    with pytest.raises(ValueError, match="At least one"):
        config.validate()


def test_extract_observation_requires_wrist_for_dino_training():
    config = RobotRuntimeConfig(
        external_camera="right",
        left_camera_id="",
        right_camera_id="",
        wrist_camera_id="17396664",
        max_timesteps=10,
        use_wrist_camera=True,
        use_exterior_camera=False,
    )

    with pytest.raises(RuntimeError, match="wrist camera image"):
        _extract_observation(config, _make_raw_obs({}))
