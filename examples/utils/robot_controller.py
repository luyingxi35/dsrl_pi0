"""High-frequency joint position controller for DROID robots.

Analogous to UMI's RTDEInterpolationController: decouples policy inference
from robot execution by maintaining a continuous trajectory that runs at
high frequency independently of the 10Hz policy loop.

Architecture:
    Main thread (10Hz):  add_waypoints(times, joint_positions) → non-blocking
    Controller thread (200Hz): interpolates trajectory → Polymetis gRPC

The controller accesses env._robot._robot (polymetis.RobotInterface) directly,
bypassing DROID's zerorpc/gevent layer so it is safe to call from a background
threading.Thread (Polymetis uses gRPC which is thread-safe).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np


class JointTrajectoryInterpolator:
    """Linear interpolator over a sequence of (monotonic time, joint_positions_7d) waypoints.

    Internally all times are in time.monotonic() so the 200 Hz loop is immune
    to NTP wall-clock adjustments.  Callers that hold wall-clock times must
    convert them before calling update_waypoints() (see HighFreqController).

    Clamps to the first/last waypoint outside the time range.
    Thread-safe when protected by an external lock.

    NOTE: this is the GPU-server reference copy of
    droid/droid/franka/trajectory_controller.py — keep them in sync.
    """

    def __init__(self) -> None:
        self._times: np.ndarray = np.array([], dtype=np.float64)
        self._positions: np.ndarray = np.empty((0, 7), dtype=np.float64)

    def set_waypoints(self, times: np.ndarray, positions: np.ndarray) -> None:
        """Hard-replace the current trajectory. times must be sorted ascending.

        Prefer update_waypoints() for normal use — it guarantees C0 continuity
        and applies a joint-speed cap.  set_waypoints() is kept for callers
        that need unconditional replacement (e.g. tests, episode reset).
        """
        times = np.asarray(times, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)
        assert len(times) == len(positions), "times and positions must have equal length"
        self._times = times
        self._positions = positions

    def update_waypoints(
        self,
        times: np.ndarray,
        positions: np.ndarray,
        curr_time: float,
        max_joint_speed_rad_s: float = 3.0,
    ) -> None:
        """Replace trajectory, preserving C0 continuity from the current execution point.

        Two guarantees:

        1. **Continuity** — if ``curr_time`` precedes the first new waypoint,
           the current interpolated pose is prepended as a leading waypoint so
           the 200 Hz loop transitions smoothly from wherever it currently is.
           This correctly handles the *overlap* case where a new action chunk
           arrives while the previous chunk is still being executed: without
           this, ``__call__`` would clamp to the first new waypoint immediately
           and the robot would jump.

        2. **Speed cap** — if any consecutive waypoint pair implies a joint
           velocity exceeding ``max_joint_speed_rad_s``, the later waypoint's
           time is extended (and all subsequent times shifted) to satisfy the
           limit.  This mirrors UMI's
           ``PoseTrajectoryInterpolator.schedule_waypoint`` max-speed constraint
           and prevents runaway velocities when action chunks have large gaps.

        Args:
            times: (N,) monotonic target times for each waypoint.
            positions: (N, 7) absolute joint angles in radians.
            curr_time: current time.monotonic() value at the call site.
            max_joint_speed_rad_s: per-joint speed limit (rad/s). Default 3.0
                rad/s is roughly 1.5× the DROID training speed at action_scale=1.
        """
        times     = np.asarray(times,     dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)

        # ── 1. Continuity: bridge from current commanded position ───────────
        # arm_times = new_t - robot_action_latency puts times[0] ~170ms in the
        # past, so the old `curr_time < times[0]` condition was never True and
        # the bridge never fired.  Fix: trim past waypoints, always bridge.
        curr_pos = self.__call__(curr_time)
        if curr_pos is not None and len(times) > 0:
            future_mask = times > curr_time
            if np.any(future_mask):
                times     = np.concatenate([[curr_time], times[future_mask]])
                positions = np.vstack([curr_pos[None], positions[future_mask]])
            else:
                traj_duration = float(times[-1] - times[0]) if len(times) > 1 else 0.1
                times     = np.array([curr_time, curr_time + max(traj_duration, 0.05)])
                positions = np.vstack([curr_pos[None], positions[[-1]]])

        # ── 2. Speed cap: extend waypoint times where needed ────────────────
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            if dt <= 0:
                continue
            max_delta   = float(np.max(np.abs(positions[i] - positions[i - 1])))
            required_dt = max_delta / max_joint_speed_rad_s
            if required_dt > dt:
                times[i:] = times[i:] + (required_dt - dt)  # shift tail forward

        self._times     = times
        self._positions = positions

    def __call__(self, t: float) -> np.ndarray | None:
        """Return interpolated joint positions at monotonic time t."""
        if len(self._times) == 0:
            return None
        if t <= self._times[0]:
            return self._positions[0].copy()
        if t >= self._times[-1]:
            return self._positions[-1].copy()
        idx = int(np.searchsorted(self._times, t, side="right")) - 1
        t0, t1 = self._times[idx], self._times[idx + 1]
        alpha = (t - t0) / (t1 - t0)
        return (1.0 - alpha) * self._positions[idx] + alpha * self._positions[idx + 1]

    @property
    def is_empty(self) -> bool:
        return len(self._times) == 0


class HighFreqController(threading.Thread):
    """200Hz joint position controller using Polymetis gRPC directly.

    Bypasses DROID's zerorpc/gevent layer by calling
    env._robot._robot.update_desired_joint_positions() — a Polymetis gRPC
    call that is safe from a non-main thread (gRPC is thread-safe).

    Prerequisites (call before start()):
        env._robot.update_joints(current_pos, velocity=False, blocking=False)
    This triggers DROID's impedance controller startup so Polymetis is ready
    to accept continuous position targets.

    Usage::

        controller = HighFreqController(env._robot._robot, frequency=200.0)
        controller.start()
        controller.add_waypoints(times, arm_positions_7d)   # non-blocking
        ...
        controller.stop()
        controller.join()
    """

    def __init__(self, polymetis_robot: Any, frequency: float = 200.0) -> None:
        super().__init__(daemon=True, name="HighFreqController")
        self._robot = polymetis_robot  # polymetis.RobotInterface
        self._dt = 1.0 / frequency
        self._interp = JointTrajectoryInterpolator()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def add_waypoints(self, times: np.ndarray, positions: np.ndarray) -> None:
        """Replace the current trajectory batch. Non-blocking, thread-safe.

        The incoming ``times`` are **seconds relative to when the GPU server
        called add_waypoints** (``arm_times - time.time()`` on the GPU).
        Using offsets instead of absolute wall-clock values eliminates
        GPU-NUC clock-skew error.  Residual network latency (~5 ms) is
        acceptable and handled by update_waypoints' trim-and-bridge logic.

        Args:
            times: (N,) float64 time offsets (seconds from GPU call time).
            positions: (N, 7) absolute joint angles in radians.
        """
        mono_times = time.monotonic() + np.asarray(times, dtype=np.float64)
        with self._lock:
            self._interp.update_waypoints(
                mono_times,
                np.asarray(positions, dtype=np.float64),
                curr_time=time.monotonic(),
            )

    def stop(self) -> None:
        """Signal the controller loop to exit."""
        self._stop_event.set()

    def run(self) -> None:
        import torch

        # Use time.monotonic() for the control loop so NTP wall-clock
        # adjustments cannot corrupt inter-tick sleep timing.
        t_start = time.monotonic()
        iter_idx = 0

        while not self._stop_event.is_set():
            t_now_mono = time.monotonic()

            with self._lock:
                joint_target = self._interp(t_now_mono)

            if joint_target is not None:
                try:
                    self._robot.update_desired_joint_positions(
                        torch.tensor(joint_target, dtype=torch.float32)
                    )
                except Exception:
                    logging.exception("HighFreqController: update_desired_joint_positions failed")

            iter_idx += 1
            sleep_s = t_start + iter_idx * self._dt - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
