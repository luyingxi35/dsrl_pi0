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
from examples.utils.real_robot_common import (
    LatestObservationBuffer,
    action_timestamps_from_obs,
    binarize_and_clip_action,
    integrate_joint_velocity_actions,
)


# ── is_new filtering ──────────────────────────────────────────────────────────

def test_is_new_with_past_t_obs() -> None:
    """When t_obs is in the past, early action_timestamps should be stale.

    Simulates: t_obs = 100ms ago, inference took 150ms.
    At 10Hz (dt=100ms), 8 actions span 100ms to 800ms from t_obs.
    After 150ms, action[0] should be stale.
    """
    t_obs = time.time() - 0.100        # 100ms ago
    dt_step = 0.100                    # 10Hz
    n_actions = 8
    action_exec_latency = 0.01

    action_timestamps = action_timestamps_from_obs(t_obs, n_actions, dt_step)
    curr_time = time.time()
    is_new = action_timestamps > (curr_time + action_exec_latency)

    n_new = int(np.sum(is_new))
    print(f"  is_new mask: {is_new.tolist()}")
    print(f"  {n_new}/{n_actions} actions pass is_new filter")

    assert n_new >= 1,          "Expected at least 1 future action"
    assert n_new < n_actions,   "Expected at least 1 stale action (t_obs is in the past)"
    assert not is_new[0],       "action[0] should be stale (target = t_obs + dt)"
    print("  [PASS] is_new filter with past t_obs")


def test_action_timestamps_start_at_next_tick() -> None:
    """Eval action timestamps must match training: first target is t_obs + dt."""
    t_obs = 123.0
    dt_step = 0.1
    stamps = action_timestamps_from_obs(t_obs, 4, dt_step)
    expected = np.array([123.1, 123.2, 123.3, 123.4])

    assert np.allclose(stamps, expected), f"Expected {expected}, got {stamps}"
    print("  [PASS] action timestamps start at t_obs + dt")


def test_eval_timestamp_off_by_one_reduces_stale_actions() -> None:
    """Starting at k=1 recovers one fresh action versus the old k=0 schedule."""
    t_obs = 1000.0
    dt_step = 0.1
    n_actions = 8
    action_exec_latency = 0.01
    curr_time = t_obs + 0.45

    old_timestamps = t_obs + np.arange(n_actions) * dt_step
    new_timestamps = action_timestamps_from_obs(t_obs, n_actions, dt_step)
    old_is_new = old_timestamps > (curr_time + action_exec_latency)
    new_is_new = new_timestamps > (curr_time + action_exec_latency)

    assert int(np.sum(new_is_new)) == int(np.sum(old_is_new)) + 1
    print(
        "  [PASS] k=1 timestamp schedule recovers one fresh action "
        f"({int(np.sum(old_is_new))} -> {int(np.sum(new_is_new))})"
    )


def test_latest_observation_buffer_skips_old_observations() -> None:
    """Continuous worker should consume the newest observation, not backlog."""
    buf = LatestObservationBuffer()
    buf.publish({"value": "old"}, t_obs=1.0, step_id=1, t_publish=10.0)
    buf.publish({"value": "new"}, t_obs=2.0, step_id=2, t_publish=20.0)

    snap = buf.wait_for_new(last_step_id=-1, timeout=0.01)

    assert snap is not None
    assert snap.step_id == 2
    assert snap.obs["value"] == "new"
    print("  [PASS] latest observation buffer skips old observations")


def test_latest_observation_buffer_waits_for_new_step() -> None:
    """A worker should not repeat the same observation step."""
    buf = LatestObservationBuffer()
    buf.publish({"value": "only"}, t_obs=1.0, step_id=7, t_publish=10.0)

    first = buf.wait_for_new(last_step_id=-1, timeout=0.01)
    repeated = buf.wait_for_new(last_step_id=7, timeout=0.01)
    buf.close()
    closed = buf.wait_for_new(last_step_id=7, timeout=0.01)

    assert first is not None and first.step_id == 7
    assert repeated is None
    assert closed is None
    print("  [PASS] latest observation buffer waits for a new step")


def test_is_new_all_new_when_t_obs_is_now() -> None:
    """When t_obs is the current time, all actions should be new (UMI degenerate case)."""
    t_obs = time.time()
    dt_step = 0.100
    n_actions = 4
    action_exec_latency = 0.01

    action_timestamps = action_timestamps_from_obs(t_obs, n_actions, dt_step)
    curr_time = time.time()
    t_obs = curr_time
    print(f"  action_timestamps: {action_timestamps}")
    print(f"desired action_exec_latency: {curr_time + action_exec_latency}")
    is_new = action_timestamps > (curr_time + action_exec_latency)
    print(f"  is_new: {is_new}")

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

    # Full chunk: 4 actions at velocity 1.0 for joint 0 plus gripper dim
    all_actions = [np.array([1.0] + [0.0] * 7)] * 4

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


def test_worker_side_integration_uses_source_joint_position() -> None:
    """Worker-side integration must use source obs joints, not drain-time joints."""
    source_joint = np.array([1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0])
    current_joint_at_drain = source_joint + 10.0
    max_joint_delta = 0.2
    actions = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    abs_positions = integrate_joint_velocity_actions(
        source_joint, actions, max_joint_delta
    )
    wrong_positions = integrate_joint_velocity_actions(
        current_joint_at_drain, actions, max_joint_delta
    )

    expected = np.array([
        source_joint + [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        source_joint + [0.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        source_joint + [0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    assert np.allclose(abs_positions, expected)
    assert not np.allclose(abs_positions, wrong_positions)
    print("  [PASS] worker-side integration uses source joint position")


def test_filter_uses_worker_integrated_absolute_suffix() -> None:
    """Fresh filtering should select positions from the full integrated chunk."""
    source_joint = np.zeros(7)
    max_joint_delta = 0.2
    actions = np.array([[1.0] + [0.0] * 7] * 4)
    is_new = np.array([False, False, True, True])

    abs_positions = integrate_joint_velocity_actions(
        source_joint, actions, max_joint_delta
    )
    scheduled = abs_positions[is_new]

    assert np.allclose(scheduled[:, 0], [0.6, 0.8])
    print("  [PASS] fresh filter selects worker-integrated absolute suffix")


# ── velocity clipping ─────────────────────────────────────────────────────────

def test_velocity_integration_integrate_all_then_filter() -> None:
    """Regression test for evaluate_pi0_real.py Bug 1.

    Stale velocities must be accumulated before applying the is_new filter.
    Integrating only the is_new subset starts from the wrong position.

    Setup:
        chunk = 4 actions, joint-0 velocity = 1.0, action_scale = 1.0
        MAX_JOINT_DELTA = 0.2 * 1.0 = 0.2 rad/step
        is_new = [False, False, True, True]  (first 2 stale)

    Correct (integrate ALL, then slice):
        all_abs = [0.2, 0.4, 0.6, 0.8]  → scheduled = [0.6, 0.8]
    Buggy   (integrate only is_new subset):
        abs     = [0.2, 0.4]             → scheduled = [0.2, 0.4]  ← WRONG
    """
    action_scale = 1.0
    MAX_JOINT_DELTA = 0.2 * action_scale

    # 4-action chunk, all joint-0 velocity = 1.0 plus gripper dim
    all_actions = [np.array([1.0] + [0.0] * 7)] * 4   # shape (4, 8)
    is_new = [False, False, True, True]
    current = np.zeros(7)

    # ── Correct: integrate all, then filter ──────────────────────────────────
    running = current.copy()
    all_abs = []
    for a in all_actions:
        running = running + np.clip(a[:-1], -1.0, 1.0) * MAX_JOINT_DELTA
        all_abs.append(running.copy())
    scheduled_correct = [pos for pos, new in zip(all_abs, is_new) if new]

    assert abs(scheduled_correct[0][0] - 0.6) < 1e-9, (
        f"pos[2] j0 should be 0.6 (3×0.2), got {scheduled_correct[0][0]}"
    )
    assert abs(scheduled_correct[1][0] - 0.8) < 1e-9, (
        f"pos[3] j0 should be 0.8 (4×0.2), got {scheduled_correct[1][0]}"
    )

    # ── Wrong (old bug): integrate only is_new subset ────────────────────────
    new_actions_only = [a for a, new in zip(all_actions, is_new) if new]
    running_buggy = current.copy()
    buggy_abs = []
    for a in new_actions_only:
        running_buggy = running_buggy + np.clip(a[:-1], -1.0, 1.0) * MAX_JOINT_DELTA
        buggy_abs.append(running_buggy.copy())

    # Buggy positions are 0.2 and 0.4, not 0.6 and 0.8
    assert abs(buggy_abs[0][0] - 0.2) < 1e-9, "Sanity: buggy gives 0.2 for first new action"
    assert abs(buggy_abs[1][0] - 0.4) < 1e-9, "Sanity: buggy gives 0.4 for second new action"

    # Confirm the two approaches differ — fixing Bug 1 changes behaviour
    assert scheduled_correct[0][0] != buggy_abs[0][0], (
        "Correct and buggy results should differ when stale actions exist"
    )
    print(f"  Correct: {[round(p[0],3) for p in scheduled_correct]}  "
          f"Buggy: {[round(p[0],3) for p in buggy_abs]}")
    print("  [PASS] integrate all, then filter (Bug 1 regression guard)")


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
    test_action_timestamps_start_at_next_tick()
    test_eval_timestamp_off_by_one_reduces_stale_actions()
    test_latest_observation_buffer_skips_old_observations()
    test_latest_observation_buffer_waits_for_new_step()
    # test_is_new_all_new_when_t_obs_is_now()
    test_velocity_integration_basic()
    test_velocity_integration_full_speed()
    test_velocity_integration_cumulative()
    test_worker_side_integration_uses_source_joint_position()
    test_filter_uses_worker_integrated_absolute_suffix()
    test_velocity_integration_integrate_all_then_filter()
    test_velocity_clipping()
    test_binarize_gripper_open()
    test_binarize_gripper_closed()
    test_clip_arm_dimensions()
    print("\nAll tests completed.")
