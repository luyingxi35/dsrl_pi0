#!/usr/bin/env python3
"""Checks for pi0 velocity to ManiSkill action conversion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.sim_action_utils import pi0_velocity_chunk_to_sim_actions


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


if __name__ == "__main__":
    test_unit_velocity_maps_to_point_one_delta()
    test_double_velocity_maps_to_point_two_delta()
    test_default_real_scale_clips_velocity_to_point_one_delta()
    test_chunk_includes_prefix_integration()
    test_gripper_open_maps_to_maniskill_open()
    test_gripper_closed_maps_to_maniskill_closed()
    print("sim action conversion checks passed")
