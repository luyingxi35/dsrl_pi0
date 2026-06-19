#!/usr/bin/env python3
"""Checks for pi0 velocity to ManiSkill action conversion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.sim.action_utils import RealTimeActionChunker, pi0_velocity_chunk_to_sim_actions


def test_unit_velocity_maps_to_point_one_delta() -> None:
    qpos = np.zeros(7, dtype=np.float32)
    actions = np.ones((1, 8), dtype=np.float32)
    sim_actions = pi0_velocity_chunk_to_sim_actions(qpos, actions)
    np.testing.assert_allclose(sim_actions[0, :7], np.ones(7), atol=1e-6)


def test_double_velocity_maps_to_point_two_delta() -> None:
    qpos = np.zeros(7, dtype=np.float32)
    actions = np.full((1, 8), 2.0, dtype=np.float32)
    sim_actions = pi0_velocity_chunk_to_sim_actions(qpos, actions, action_scale=1.0)
    np.testing.assert_allclose(sim_actions[0, :7], np.full(7, 2.0), atol=1e-6)


def test_default_real_scale_clips_velocity_to_point_one_delta() -> None:
    qpos = np.zeros(7, dtype=np.float32)
    actions = np.full((1, 8), 2.0, dtype=np.float32)
    sim_actions = pi0_velocity_chunk_to_sim_actions(qpos, actions)
    np.testing.assert_allclose(sim_actions[0, :7], np.ones(7), atol=1e-6)


def test_chunk_includes_prefix_integration() -> None:
    qpos = np.zeros(7, dtype=np.float32)
    actions = np.zeros((3, 8), dtype=np.float32)
    actions[:, 0] = 1.0
    sim_actions = pi0_velocity_chunk_to_sim_actions(qpos, actions)
    np.testing.assert_allclose(sim_actions[:, 0], np.array([1.0, 2.0, 3.0]), atol=1e-6)


def test_gripper_open_maps_to_maniskill_open() -> None:
    qpos = np.zeros(7, dtype=np.float32)
    actions = np.zeros((2, 8), dtype=np.float32)
    actions[:, 7] = [0.0, 0.5]
    sim_actions = pi0_velocity_chunk_to_sim_actions(qpos, actions)
    np.testing.assert_allclose(sim_actions[:, 7], np.ones(2), atol=1e-6)


def test_gripper_closed_maps_to_maniskill_closed() -> None:
    qpos = np.zeros(7, dtype=np.float32)
    actions = np.zeros((2, 8), dtype=np.float32)
    actions[:, 7] = [0.51, 1.0]
    sim_actions = pi0_velocity_chunk_to_sim_actions(qpos, actions)
    np.testing.assert_allclose(sim_actions[:, 7], -np.ones(2), atol=1e-6)


def test_action_chunker_first_step_returns_newest_first_action() -> None:
    chunker = RealTimeActionChunker(action_horizon=3, action_dim=2, m=0.5)
    chunk = np.array([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]], dtype=np.float32)
    np.testing.assert_allclose(chunker.step(chunk), chunk[0], atol=1e-6)


def test_action_chunker_second_step_blends_aligned_predictions() -> None:
    chunker = RealTimeActionChunker(action_horizon=3, action_dim=1, m=0.5)
    first = np.array([[1.0], [10.0], [100.0]], dtype=np.float32)
    second = np.array([[2.0], [20.0], [200.0]], dtype=np.float32)
    chunker.step(first)

    weight_old = np.exp(-0.5)
    expected = (second[0] + first[1] * weight_old) / (1.0 + weight_old)
    np.testing.assert_allclose(chunker.step(second), expected, atol=1e-6)


def test_action_chunker_drops_predictions_after_horizon() -> None:
    chunker = RealTimeActionChunker(action_horizon=2, action_dim=1, m=0.0)
    first = np.array([[0.0], [1000.0]], dtype=np.float32)
    second = np.array([[2.0], [20.0]], dtype=np.float32)
    third = np.array([[4.0], [40.0]], dtype=np.float32)
    chunker.step(first)
    chunker.step(second)

    np.testing.assert_allclose(chunker.step(third), np.array([12.0], dtype=np.float32), atol=1e-6)


def test_action_chunker_accepts_torch_like_tensor() -> None:
    class TorchLikeTensor:
        def __init__(self, array: np.ndarray):
            self._array = array

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._array

    chunker = RealTimeActionChunker(action_horizon=2, action_dim=1)
    chunk = TorchLikeTensor(np.array([[3.0], [4.0]], dtype=np.float32))
    np.testing.assert_allclose(chunker.step(chunk), np.array([3.0], dtype=np.float32), atol=1e-6)


def test_action_chunker_rejects_wrong_shape() -> None:
    chunker = RealTimeActionChunker(action_horizon=2, action_dim=3)
    try:
        chunker.step(np.zeros((2, 4), dtype=np.float32))
    except ValueError as exc:
        assert "Expected action chunk shape (2, 3)" in str(exc)
    else:
        raise AssertionError("Expected ValueError for wrong action chunk shape.")


if __name__ == "__main__":
    test_unit_velocity_maps_to_point_one_delta()
    test_double_velocity_maps_to_point_two_delta()
    test_default_real_scale_clips_velocity_to_point_one_delta()
    test_chunk_includes_prefix_integration()
    test_gripper_open_maps_to_maniskill_open()
    test_gripper_closed_maps_to_maniskill_closed()
    test_action_chunker_first_step_returns_newest_first_action()
    test_action_chunker_second_step_blends_aligned_predictions()
    test_action_chunker_drops_predictions_after_horizon()
    test_action_chunker_accepts_torch_like_tensor()
    test_action_chunker_rejects_wrong_shape()
    print("sim action conversion checks passed")
