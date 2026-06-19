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
    """Binarise a DROID gripper action value to a ManiSkill gripper command.

    DROID convention: action_value in [0, 1]
      > 0.5  →  open  (desired gripper position = open)
      < 0.5  →  close (desired gripper position = closed)

    ManiSkill PDJointPosMimicController (normalize_action=True):
      +1.0  →  target = 0.04 m  (open)
      -1.0  →  target = -0.01 m (closed)

    Real-robot parity: binarize_and_clip_action() in real_robot_common.py uses
      gripper = 1.0 if action[-1] > 0.5 else 0.0
    so action > 0.5 → open in both real and sim.
    """
    return SIM_GRIPPER_OPEN if float(action_value) > 0.5 else SIM_GRIPPER_CLOSED


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
    # Chunk-level gripper decision: binarize the mean over the whole chunk
    # to avoid within-chunk oscillation (e.g., pi0.5's long action_horizon=15
    # may predict a gripper transition mid-chunk).
    chunk_gripper = binarize_sim_gripper(float(np.mean(actions_arr[:, 7])))
    sim_actions: list[np.ndarray] = []
    for action in actions_arr:
        velocity = np.clip(action[:7], -float(action_clip), float(action_clip))
        running_joints = running_joints + velocity * float(max_joint_delta)
        delta = running_joints - qpos[:7]
        sim_action = np.empty((8,), dtype=np.float32)
        sim_action[:7] = delta / MANISKILL_JOINT_DELTA
        sim_action[7] = chunk_gripper
        sim_actions.append(sim_action)
    return np.stack(sim_actions, axis=0)



def pi0_vel_chunk_to_joint_pos_actions(
    source_qpos: np.ndarray,
    actions: np.ndarray,
    action_scale: float = DEFAULT_SIM_ACTION_SCALE,
    execution_steps: int = 6,
    action_clip: float = SIM_ACTION_CLIP,
) -> np.ndarray:
    """Convert a pi0 velocity chunk to absolute joint-position targets.

    Used with ManiSkill ``pd_joint_pos`` controller (normalize_action=False).
    Integrates the first ``execution_steps`` velocity steps from ``source_qpos``
    and returns absolute joint angles in radians.

    Args:
        source_qpos:     (>=7,) current joint positions [rad], read from env obs.
        actions:         (H, >=8) pi0 output — normalised joint velocities [:7]
                         and gripper command [7].
        action_scale:    Scale on DROID's max joint delta (0.2 rad/step).
                         Default 0.5 → 0.10 rad/step, matching evaluate_pi0_real.py.
        execution_steps: Number of waypoints to return (cap at len(actions)).
        action_clip:     Velocity clip bound before scaling.

    Returns:
        (N, 8) float32 where N = min(execution_steps, len(actions)):
            [:, :7]  absolute joint angles [rad]  (feed directly to pd_joint_pos)
            [:, 7]   binarised gripper ∈ {+1 open, -1 closed}  (ManiSkill convention)
    """
    max_joint_delta = DROID_MAX_JOINT_DELTA * float(action_scale)
    qpos = np.asarray(source_qpos, dtype=np.float32).reshape(-1)
    if qpos.shape[0] < 7:
        raise ValueError(f"Expected source_qpos with at least 7 values, got {qpos.shape}")

    actions_arr = np.asarray(actions, dtype=np.float32)
    if actions_arr.ndim != 2 or actions_arr.shape[1] < 8:
        raise ValueError(f"Expected pi0 actions shape (H, >=8), got {actions_arr.shape}")

    n_steps = min(int(execution_steps), len(actions_arr))
    running_joints = qpos[:7].copy()
    # Chunk-level gripper decision: binarize the mean over the executed steps.
    # Pi0.5 (action_horizon=15) may predict a gripper-state transition within
    # the 15-step chunk; applying binarize per-step causes rapid open/close
    # oscillation. A single decision per chunk matches real-robot behaviour
    # where the gripper command is held constant for the execution window.
    chunk_gripper = binarize_sim_gripper(float(np.mean(actions_arr[:n_steps, 7])))
    result: list[np.ndarray] = []
    for action in actions_arr[:n_steps]:
        velocity = np.clip(action[:7], -float(action_clip), float(action_clip))
        running_joints = running_joints + velocity * float(max_joint_delta)
        step_action = np.empty((8,), dtype=np.float32)
        step_action[:7] = running_joints   # absolute [rad]
        step_action[7]  = chunk_gripper    # ±1, same for all steps in this chunk
        result.append(step_action)
    return np.stack(result, axis=0)


# Physical Panda joint limits (rad) — used for direct-position mapping.
PANDA_JOINT_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32
)
PANDA_JOINT_UPPER = np.array(
    [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], dtype=np.float32
)


def pi0_action_as_joint_pos(
    actions: np.ndarray,
    execution_steps: int = 6,
) -> np.ndarray:
    """Treat pi0 raw output directly as normalised joint *positions*.

    Maps each pi0 action[:7] ∈ [-1, 1] to an absolute Panda joint angle
    using the physical joint limits::

        target_k = (upper + lower) / 2 + action_k * (upper - lower) / 2

    This is the diagnostic counterpart of ``pi0_vel_chunk_to_joint_pos_actions``:
    use it to test the hypothesis that the checkpoint encodes absolute positions
    rather than velocities, without any integration.

    Args:
        actions:         (H, >=8) pi0 output — raw values in approximately [-1, 1].
        execution_steps: Number of waypoints to return (cap at len(actions)).

    Returns:
        (N, 8) float32 where N = min(execution_steps, len(actions)):
            [:, :7]  absolute joint angles [rad]  (feed directly to pd_joint_pos)
            [:, 7]   binarised gripper ∈ {+1 open, -1 closed}  (ManiSkill convention)
    """
    actions_arr = np.asarray(actions, dtype=np.float32)
    if actions_arr.ndim != 2 or actions_arr.shape[1] < 8:
        raise ValueError(f"Expected pi0 actions shape (H, >=8), got {actions_arr.shape}")

    centers = (PANDA_JOINT_UPPER + PANDA_JOINT_LOWER) / 2.0   # (7,)
    ranges  = (PANDA_JOINT_UPPER - PANDA_JOINT_LOWER) / 2.0   # (7,)

    n_steps = min(int(execution_steps), len(actions_arr))
    result: list[np.ndarray] = []
    for action in actions_arr[:n_steps]:
        norm_pos = np.clip(action[:7], -1.0, 1.0)
        abs_pos  = centers + norm_pos * ranges                 # [rad]
        step_action = np.empty((8,), dtype=np.float32)
        step_action[:7] = abs_pos
        step_action[7]  = binarize_sim_gripper(float(action[7]))
        result.append(step_action)
    return np.stack(result, axis=0)


def pi0_vel_to_delta_actions(
    actions: np.ndarray,
    action_scale: float = DEFAULT_SIM_ACTION_SCALE,
    execution_steps: int = 6,
    action_clip: float = SIM_ACTION_CLIP,
) -> np.ndarray:
    """Convert a pi0 velocity chunk directly to pd_joint_delta_pos actions.

    Unlike ``pi0_vel_chunk_to_joint_pos_actions`` which integrates velocity
    into absolute positions (requires ``pd_joint_pos`` controller), this
    function emits **per-step deltas** suitable for the ``pd_joint_delta_pos``
    ManiSkill controller.

    The ManiSkill ``pd_joint_delta_pos`` controller (normalize_action=True,
    bounds ±0.1 rad) maps input ±1 → actual delta of ±0.1 rad per step.
    pi0 DROID velocity ±1 scaled by (action_scale × DROID_MAX_JOINT_DELTA)
    gives the desired delta in rad.  Dividing by the controller bound (0.1)
    yields the normalised input:

        normalised_delta = clip(vel, ±1) × action_scale × DROID_MAX_JOINT_DELTA / 0.1

    With the defaults (action_scale=0.5, DROID_MAX_JOINT_DELTA=0.2):
        normalised_delta = clip(vel, ±1) × 1.0   ← identity mapping!

    So the pi0 velocity output IS the normalised delta for the controller.

    Args:
        actions:         (H, >=8) pi0 output — normalised joint velocities [:7]
                         and gripper command [7].
        action_scale:    Velocity scale (0.5 = DROID default, safe speed).
        execution_steps: Number of waypoints to return (cap at len(actions)).
        action_clip:     Velocity clip bound.

    Returns:
        (N, 8) float32:
            [:, :7]  normalised arm delta ∈ [-1, 1]
                     → actual delta = value × 0.1 rad/step (pd_joint_delta_pos)
            [:, 7]   binarised gripper ∈ {+1 open, -1 closed}

    Note:
        The real-robot parity: real uses binarize_and_clip_action() where
        action[-1] > 0.5 → gripper = 1.0 (open).  We follow the same rule.
    """
    actions_arr = np.asarray(actions, dtype=np.float32)
    if actions_arr.ndim != 2 or actions_arr.shape[1] < 8:
        raise ValueError(f"Expected pi0 actions shape (H, >=8), got {actions_arr.shape}")

    n_steps = min(int(execution_steps), len(actions_arr))
    # Normalised delta = vel × action_scale × DROID_MAX_JOINT_DELTA / 0.1
    # = vel × action_scale × 2.0
    scale = float(action_scale) * DROID_MAX_JOINT_DELTA / 0.1  # 0.5 × 0.2 / 0.1 = 1.0 (default)

    result: list[np.ndarray] = []
    for action in actions_arr[:n_steps]:
        vel = np.clip(action[:7], -float(action_clip), float(action_clip))
        normalised = np.clip(vel * scale, -1.0, 1.0)   # clip in case scale > 1
        step_action = np.empty((8,), dtype=np.float32)
        step_action[:7] = normalised
        step_action[7]  = binarize_sim_gripper(float(action[7]))
        result.append(step_action)
    return np.stack(result, axis=0)
