"""Action conversion helpers for ManiSkill pi0 simulation rollouts."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

DROID_MAX_JOINT_DELTA = 0.2
MANISKILL_JOINT_DELTA = 0.1
DEFAULT_SIM_ACTION_SCALE = 0.5
SIM_ACTION_CLIP = 1.0
SIM_GRIPPER_OPEN = 1.0
SIM_GRIPPER_CLOSED = -1.0


class RealTimeActionChunker:
    """Temporal ensemble for overlapping real-time action chunks."""

    def __init__(self, action_dim: int, action_horizon: int = 8, m: float = 0.01):
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.m = float(m)
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if self.m < 0:
            raise ValueError(f"m must be non-negative, got {m}")

        self._chunks: deque[np.ndarray] = deque(maxlen=self.action_horizon)
        self._weights = np.exp(
            -self.m * np.arange(self.action_horizon, dtype=np.float32)
        ).astype(np.float32)

    def reset(self) -> None:
        self._chunks.clear()

    def step(self, new_chunk: Any) -> np.ndarray:
        """Add a chunk and return the smoothed action for this control step."""
        chunk = self._to_numpy(new_chunk)
        expected_shape = (self.action_horizon, self.action_dim)
        if chunk.shape != expected_shape:
            raise ValueError(f"Expected action chunk shape {expected_shape}, got {chunk.shape}")

        self._chunks.appendleft(chunk.copy())

        predictions: list[np.ndarray] = []
        weights: list[np.float32] = []
        for age, past_chunk in enumerate(self._chunks):
            # A chunk predicted `age` control steps ago targets the current
            # physical timestep at index `age`; newer chunks get larger weight.
            predictions.append(past_chunk[age])
            weights.append(self._weights[age])

        stacked = np.stack(predictions, axis=0)
        weights_arr = np.asarray(weights, dtype=np.float32)
        smoothed = np.average(stacked, axis=0, weights=weights_arr)
        return np.asarray(smoothed, dtype=np.float32)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=np.float32)


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
