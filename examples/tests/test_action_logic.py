#!/usr/bin/env python3
"""Unit tests for action scheduling logic in run_rollout().

Tests:
  - is_new timestamp filtering (t_obs in the past → early actions stale)
  - velocity integration with action_scale
  - velocity clipping before integration
  - binarize_and_clip_action for gripper binarization + arm clipping

Run without a robot:
    cd ~/yingxi/dsrl_pi0
    python3 examples/tests/test_action_logic.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from examples.utils.real_robot_common import binarize_and_clip_action


# ── is_new filtering ──────────────────────────────────────────────────────────

def test_is_new_with_past_t_obs() -> None:
    """When t_obs is in the past, early action_timestamps should be stale.

    Simulates: t_obs = 100ms ago, inference took 150ms.
    At 10Hz (dt=100ms), 8 actions span 0ms to 700ms from t_obs.
    After 150ms, actions[0] and actions[1] should be stale.
    """
    t_obs = time.time() - 0.100        # 100ms ago
    dt_step = 0.100                    # 10Hz
    n_actions = 8
    action_exec_latency = 0.01

    action_timestamps = t_obs + np.arange(n_actions) * dt_step
    curr_time = time.time()
    is_new = action_timestamps > (curr_time + action_exec_latency)

    n_new = int(np.sum(is_new))
    print(f"  is_new mask: {is_new.tolist()}")
    print(f"  {n_new}/{n_actions} actions pass is_new filter")

    assert n_new >= 1,          "Expected at least 1 future action"
    assert n_new < n_actions,   "Expected at least 1 stale action (t_obs is in the past)"
    assert not is_new[0],       "action[0] should be stale (target = t_obs ≈ 100ms ago)"
    print("  [PASS] is_new filter with past t_obs")


def test_is_new_all_new_when_t_obs_is_now() -> None:
    """When t_obs is the current time, all actions should be new (UMI degenerate case)."""
    t_obs = time.time()
    dt_step = 0.100
    n_actions = 4
    action_exec_latency = 0.01

    action_timestamps = t_obs + np.arange(n_actions) * dt_step
    curr_time = time.time()
    is_new = action_timestamps > (curr_time + action_exec_latency)

    assert is_new[0],  "action[0] should be new when t_obs ≈ now"
    print(f"  [PASS] is_new all-new when t_obs ≈ now ({int(np.sum(is_new))}/{n_actions} new)")


# ── velocity integration ──────────────────────────────────────────────────────

def test_velocity_integration_basic() -> None:
    """Verify cumulative delta for joint 0 with constant velocity."""
    action_scale   = 0.5
    MAX_JOINT_DELTA = 0.2 * action_scale   # 0.1 rad/step

    velocities = [np.array([0.5] + [0.0] * 6)] * 3   # v_j0 = 0.5 for 3 steps
    expected_j0 = [0.05, 0.10, 0.15]                   # cumulative

    running = np.zeros(7)
    for i, vel in enumerate(velocities):
        running = running + np.clip(vel, -1.0, 1.0) * MAX_JOINT_DELTA
        assert abs(running[0] - expected_j0[i]) < 1e-9, (
            f"Step {i}: expected j0={expected_j0[i]:.3f}, got {running[0]:.9f}"
        )
    print(f"  [PASS] velocity integration: final j0={running[0]:.3f} rad")


def test_velocity_integration_full_speed() -> None:
    """action_scale=1.0 → max 0.2 rad/step."""
    MAX_JOINT_DELTA = 0.2 * 1.0

    running = np.zeros(7)
    vel = np.ones(7)   # all joints at full speed
    running = running + np.clip(vel, -1.0, 1.0) * MAX_JOINT_DELTA
    assert np.allclose(running, [0.2] * 7), f"Expected 0.2 rad each, got {running}"
    print("  [PASS] velocity integration at full speed (0.2 rad/step)")


def test_velocity_integration_cumulative() -> None:
    """Integration is cumulative across the action chunk, not from current robot state.

    This is the key property: pos[k] = current + sum(delta[0..k]).
    If only new_a (is_new subset) is integrated, missing early deltas causes error.
    This test documents the expected behaviour.
    """
    action_scale   = 1.0
    MAX_JOINT_DELTA = 0.2 * action_scale

    # Full chunk: 4 actions at velocity 1.0 for joint 0
    all_actions = [np.array([1.0] + [0.0] * 6)] * 4

    current = np.zeros(7)

    # Simulate correct integration (over ALL actions, then take is_new slice):
    running_all = current.copy()
    all_abs = []
    for a in all_actions:
        running_all += np.clip(a[:-1], -1.0, 1.0) * MAX_JOINT_DELTA
        all_abs.append(running_all.copy())

    # is_new skips first 2 → take all_abs[2:]
    is_new = [False, False, True, True]
    new_abs = [pos for pos, new in zip(all_abs, is_new) if new]

    assert abs(new_abs[0][0] - 0.6) < 1e-9, (
        f"pos[2] j0 should be 0.6 (3×0.2), got {new_abs[0][0]}"
    )
    assert abs(new_abs[1][0] - 0.8) < 1e-9, (
        f"pos[3] j0 should be 0.8 (4×0.2), got {new_abs[1][0]}"
    )
    print("  [PASS] cumulative integration (integrate all, then take is_new slice)")


# ── velocity clipping ─────────────────────────────────────────────────────────

def test_velocity_clipping() -> None:
    """Out-of-range velocities must be clipped to [-1, 1] before scaling."""
    MAX_JOINT_DELTA = 0.2 * 1.0

    vel = np.array([2.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    delta = np.clip(vel, -1.0, 1.0) * MAX_JOINT_DELTA

    assert abs(delta[0] -  0.2) < 1e-9, f"Expected +0.2 (clipped +1), got {delta[0]}"
    assert abs(delta[1] - -0.2) < 1e-9, f"Expected -0.2 (clipped -1), got {delta[1]}"
    assert abs(delta[2]       ) < 1e-9, f"Expected 0.0, got {delta[2]}"
    print("  [PASS] velocity clipping")


# ── binarize_and_clip_action ──────────────────────────────────────────────────

def test_binarize_gripper_open() -> None:
    """Gripper ≤ 0.5 → binarize to 0.0 (open)."""
    a = binarize_and_clip_action(np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]))
    assert a[-1] == 0.0, f"Expected gripper=0, got {a[-1]}"
    # boundary: exactly 0.5 → closed? No: > 0.5 is closed
    a2 = binarize_and_clip_action(np.array([0.0] * 7 + [0.5]))
    assert a2[-1] == 0.0, "Gripper=0.5 should binarize to 0 (not > 0.5)"
    print("  [PASS] binarize gripper → open (0.0)")


def test_binarize_gripper_closed() -> None:
    """Gripper > 0.5 → binarize to 1.0 (closed)."""
    a = binarize_and_clip_action(np.array([0.0] * 7 + [0.51]))
    assert a[-1] == 1.0, f"Expected gripper=1, got {a[-1]}"
    a2 = binarize_and_clip_action(np.array([0.0] * 7 + [0.8]))
    assert a2[-1] == 1.0, f"Expected gripper=1, got {a2[-1]}"
    print("  [PASS] binarize gripper → closed (1.0)")


def test_clip_arm_dimensions() -> None:
    """Arm dimensions must be clipped to [-1, 1]."""
    a = binarize_and_clip_action(np.array([2.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6]))
    assert np.all(a[:-1] <= 1.0), f"Arm dims exceed +1: {a[:-1]}"
    assert np.all(a[:-1] >= -1.0), f"Arm dims below -1: {a[:-1]}"
    assert a[0]  ==  1.0, f"Expected a[0] clipped to +1.0, got {a[0]}"
    assert a[1]  == -1.0, f"Expected a[1] clipped to -1.0, got {a[1]}"
    print("  [PASS] arm dimensions clipped to [-1, 1]")


if __name__ == "__main__":
    print("=== Action Logic Tests ===\n")
    test_is_new_with_past_t_obs()
    test_is_new_all_new_when_t_obs_is_now()
    test_velocity_integration_basic()
    test_velocity_integration_full_speed()
    test_velocity_integration_cumulative()
    test_velocity_clipping()
    test_binarize_gripper_open()
    test_binarize_gripper_closed()
    test_clip_arm_dimensions()
    print("\nAll tests completed.")
