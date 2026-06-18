"""Action conversion helpers for ManiSkill pi0 simulation rollouts."""

from __future__ import annotations

import numpy as np

DROID_MAX_JOINT_DELTA = 0.2
MANISKILL_JOINT_DELTA = 0.1
DEFAULT_SIM_ACTION_SCALE = 0.5
SIM_ACTION_CLIP = 1.0
SIM_GRIPPER_OPEN = 1.0
SIM_GRIPPER_CLOSED = -1.0


def binarize_sim_gripper(action_value: float) -> float:
    return SIM_GRIPPER_CLOSED if float(action_value) > 0.5 else SIM_GRIPPER_OPEN


def pi0_velocity_chunk_to_sim_actions(
    source_qpos: np.ndarray,
    actions: np.ndarray,
    action_scale: float = DEFAULT_SIM_ACTION_SCALE,
    action_clip: float = SIM_ACTION_CLIP,
) -> np.ndarray:
    """Convert a pi0 velocity chunk to ManiSkill joint-target actions.

    pi0 DROID emits normalized joint velocities. ManiSkill's default Panda
    controller for this task expects normalized joint delta targets, so we
    integrate the pi0 velocity chunk with DROID's max joint delta and then
    express each target as a normalized delta for the current simulated qpos.
    """
    max_joint_delta = DROID_MAX_JOINT_DELTA * float(action_scale)
    if max_joint_delta <= 0:
        raise ValueError(f"action_scale must be positive, got {action_scale}")

    qpos = np.asarray(source_qpos, dtype=np.float32).reshape(-1)
    if qpos.shape[0] < 7:
        raise ValueError(f"Expected source_qpos with at least 7 values, got {qpos.shape}")

    actions_arr = np.asarray(actions, dtype=np.float32)
    if actions_arr.ndim != 2 or actions_arr.shape[1] < 8:
        raise ValueError(f"Expected pi0 actions shape (H, >=8), got {actions_arr.shape}")

    running_joints = qpos[:7].astype(np.float32).copy()
    sim_actions: list[np.ndarray] = []
    for action in actions_arr:
        velocity = np.clip(action[:7], -float(action_clip), float(action_clip))
        running_joints = running_joints + velocity * float(max_joint_delta)
        delta = running_joints - qpos[:7]
        sim_action = np.empty((8,), dtype=np.float32)
        sim_action[:7] = delta / MANISKILL_JOINT_DELTA
        sim_action[7] = binarize_sim_gripper(float(action[7]))
        sim_actions.append(sim_action)
    return np.stack(sim_actions, axis=0)
