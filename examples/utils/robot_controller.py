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
    """Linear interpolator over a sequence of (time, joint_positions_7d) waypoints.

    Clamps to the first/last waypoint outside the time range.
    Thread-safe when protected by an external lock.
    """

    def __init__(self) -> None:
        self._times: np.ndarray = np.array([], dtype=np.float64)
        self._positions: np.ndarray = np.empty((0, 7), dtype=np.float64)

    def set_waypoints(self, times: np.ndarray, positions: np.ndarray) -> None:
        """Replace the current trajectory. times must be sorted ascending."""
        times = np.asarray(times, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)
        assert len(times) == len(positions), "times and positions must have equal length"
        self._times = times
        self._positions = positions

    def __call__(self, t: float) -> np.ndarray | None:
        """Return interpolated joint positions at wall-clock time t."""
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

        Args:
            times: (N,) wall-clock target times (seconds, from time.time()).
                   Each entry is compensated for robot_action_latency before
                   being passed here (see run_rollout).
            positions: (N, 7) absolute joint angles in radians.
        """
        with self._lock:
            self._interp.set_waypoints(times, positions)

    def stop(self) -> None:
        """Signal the controller loop to exit."""
        self._stop_event.set()

    def run(self) -> None:
        import torch

        t_start = time.time()
        iter_idx = 0

        while not self._stop_event.is_set():
            t_now = time.time()

            with self._lock:
                joint_target = self._interp(t_now)

            if joint_target is not None:
                try:
                    self._robot.update_desired_joint_positions(
                        torch.tensor(joint_target, dtype=torch.float32)
                    )
                except Exception:
                    logging.exception("HighFreqController: update_desired_joint_positions failed")

            iter_idx += 1
            t_next = t_start + iter_idx * self._dt
            sleep_s = t_next - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
