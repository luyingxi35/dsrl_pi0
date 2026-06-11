"""Per-tick diagnostic data collector for pi0 real-robot evaluation.

Attach a DiagnosticLogger to run_rollout() to record timing, inference
statistics, planned arm trajectories and actual robot state.  The collected
data can then be passed to visualize_rollout.py for plotting.

Usage inside run_rollout():
    from examples.tests.diagnostic_logger import DiagnosticLogger
    logger = DiagnosticLogger()
    ...
    logger.record_tick(t_tick, ...)
    ...
    logger.save("/path/to/episode_000.npz")

Zero overhead when no logger is passed (all call sites check for None).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


class DiagnosticLogger:
    """Accumulates per-tick rollout data and saves to a .npz archive."""

    # Sentinel for "no command sent this tick"
    GRIPPER_NO_CMD = float("nan")

    def __init__(self) -> None:
        # ── Timing ────────────────────────────────────────────────────────────
        self._t_tick:             list[float] = []   # wall-clock at tick start
        self._t_step_end:         list[float] = []   # intended deadline
        self._t_obs:              list[float] = []   # camera-anchored obs timestamp
        self._tick_overrun_ms:    list[float] = []   # time.time() - t_step_end, clamped ≥ 0
        # ── Observation ───────────────────────────────────────────────────────
        self._joint_snapshot:     list[Any]   = []   # (7,) from robot_state (snapshot)
        self._joint_interp:       list[Any]   = []   # (7,) from StateInterpolator, or NaN×7
        self._state_history_n:    list[int]   = []   # entries in state history
        self._state_history_span: list[float] = []   # seconds of state history coverage
        # ── Inference ─────────────────────────────────────────────────────────
        self._infer_triggered:    list[bool]  = []   # inference thread started this tick
        self._infer_recv:         list[bool]  = []   # inference result arrived this tick
        self._infer_source_step:  list[int]   = []   # obs step used by returned result, -1 if none
        self._n_returned:         list[int]   = []   # actions returned by pi0 (0 if none)
        self._n_is_new:           list[int]   = []   # passed is_new filter
        self._n_stale:            list[int]   = []   # stale actions in returned chunk
        self._n_scheduled:        list[int]   = []   # min(n_is_new, execution_steps)
        self._all_stale:          list[bool]  = []   # all actions were stale
        self._infer_latency_s:    list[float] = []   # submit→result wall-clock latency
        self._obs_age_at_submit_s: list[float] = []  # submit time minus t_obs
        self._obs_age_at_drain_s:  list[float] = []  # drain time minus t_obs
        self._result_queue_delay_s: list[float] = [] # drain time minus inference done
        # ── Arm execution ─────────────────────────────────────────────────────
        # Ragged lists: each element is an (N, 7) or (0, 7) array.
        self._arm_times_sent:     list[Any]   = []   # wall-clock target times per waypoint
        self._arm_positions_sent: list[Any]   = []   # (N_scheduled, 7) or empty
        # ── Gripper ───────────────────────────────────────────────────────────
        self._gripper_cmd:        list[float] = []   # NaN = no command this tick

    # ── Public recording API ──────────────────────────────────────────────────

    def record_tick(
        self,
        *,
        t_tick: float,
        t_step_end: float,
        t_obs: float,
        joint_pos_snapshot: np.ndarray,         # (7,)
        joint_pos_interp:   np.ndarray | None,  # (7,) or None
        state_history_n:    int,
        state_history_span: float,
        infer_triggered:    bool,
        infer_result_recv:  bool,
        infer_source_step:  int | None,
        n_returned:         int,
        n_is_new:           int,
        n_stale:            int,
        n_scheduled:        int,
        all_stale:          bool,
        infer_latency_s:    float | None,
        obs_age_at_submit_s: float | None,
        obs_age_at_drain_s:  float | None,
        result_queue_delay_s: float | None,
        arm_times_sent:     np.ndarray | None,      # (N,) or None
        arm_positions_sent: np.ndarray | None,      # (N, 7) or None
        gripper_cmd_sent:   float | None,
        t_after_sleep:      float | None = None,    # wall-clock after tick sleep
    ) -> None:
        """Record one tick's worth of diagnostic data. Call once per tick."""
        self._t_tick.append(t_tick)
        self._t_step_end.append(t_step_end)
        self._t_obs.append(t_obs)

        t_end = t_after_sleep if t_after_sleep is not None else t_step_end
        self._tick_overrun_ms.append(max(0.0, (t_end - t_step_end) * 1000.0))

        self._joint_snapshot.append(np.asarray(joint_pos_snapshot, dtype=np.float64))
        if joint_pos_interp is not None:
            self._joint_interp.append(np.asarray(joint_pos_interp, dtype=np.float64))
        else:
            self._joint_interp.append(np.full(7, np.nan, dtype=np.float64))

        self._state_history_n.append(state_history_n)
        self._state_history_span.append(state_history_span)

        self._infer_triggered.append(infer_triggered)
        self._infer_recv.append(infer_result_recv)
        self._infer_source_step.append(
            int(infer_source_step) if infer_source_step is not None else -1
        )
        self._n_returned.append(n_returned)
        self._n_is_new.append(n_is_new)
        self._n_stale.append(n_stale)
        self._n_scheduled.append(n_scheduled)
        self._all_stale.append(all_stale)
        self._infer_latency_s.append(
            float(infer_latency_s) if infer_latency_s is not None else math.nan
        )
        self._obs_age_at_submit_s.append(
            float(obs_age_at_submit_s) if obs_age_at_submit_s is not None else math.nan
        )
        self._obs_age_at_drain_s.append(
            float(obs_age_at_drain_s) if obs_age_at_drain_s is not None else math.nan
        )
        self._result_queue_delay_s.append(
            float(result_queue_delay_s) if result_queue_delay_s is not None else math.nan
        )

        if arm_times_sent is not None and len(arm_times_sent) > 0:
            self._arm_times_sent.append(np.asarray(arm_times_sent, dtype=np.float64))
            self._arm_positions_sent.append(np.asarray(arm_positions_sent, dtype=np.float64))
        else:
            self._arm_times_sent.append(np.empty(0, dtype=np.float64))
            self._arm_positions_sent.append(np.empty((0, 7), dtype=np.float64))

        self._gripper_cmd.append(
            gripper_cmd_sent if gripper_cmd_sent is not None else self.GRIPPER_NO_CMD
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save all recorded data to a .npz file.

        Ragged arrays (arm_times_sent, arm_positions_sent, joint_snapshot,
        joint_interp) are stored as object arrays so each tick can have a
        different length.  Visualize with visualize_rollout.py.
        """
        n = len(self._t_tick)
        if n == 0:
            raise ValueError("No ticks recorded — nothing to save.")

        arm_times_arr     = np.empty(n, dtype=object)
        arm_positions_arr = np.empty(n, dtype=object)
        joint_snapshot_arr = np.empty(n, dtype=object)
        joint_interp_arr   = np.empty(n, dtype=object)
        for i in range(n):
            arm_times_arr[i]      = self._arm_times_sent[i]
            arm_positions_arr[i]  = self._arm_positions_sent[i]
            joint_snapshot_arr[i] = self._joint_snapshot[i]
            joint_interp_arr[i]   = self._joint_interp[i]

        np.savez_compressed(
            str(path),
            # Timing
            t_tick             = np.array(self._t_tick),
            t_step_end         = np.array(self._t_step_end),
            t_obs              = np.array(self._t_obs),
            tick_overrun_ms    = np.array(self._tick_overrun_ms),
            # Observation
            joint_pos_snapshot = joint_snapshot_arr,
            joint_pos_interp   = joint_interp_arr,
            state_history_n    = np.array(self._state_history_n, dtype=np.int32),
            state_history_span = np.array(self._state_history_span),
            # Inference
            infer_triggered    = np.array(self._infer_triggered, dtype=bool),
            infer_recv         = np.array(self._infer_recv,      dtype=bool),
            infer_source_step  = np.array(self._infer_source_step, dtype=np.int32),
            n_returned         = np.array(self._n_returned,  dtype=np.int32),
            n_is_new           = np.array(self._n_is_new,    dtype=np.int32),
            n_stale            = np.array(self._n_stale,     dtype=np.int32),
            n_scheduled        = np.array(self._n_scheduled, dtype=np.int32),
            all_stale          = np.array(self._all_stale,   dtype=bool),
            infer_latency_s    = np.array(self._infer_latency_s),
            obs_age_at_submit_s = np.array(self._obs_age_at_submit_s),
            obs_age_at_drain_s  = np.array(self._obs_age_at_drain_s),
            result_queue_delay_s = np.array(self._result_queue_delay_s),
            # Arm execution (object arrays for ragged shapes)
            arm_times_sent     = arm_times_arr,
            arm_positions_sent = arm_positions_arr,
            # Gripper
            gripper_cmd        = np.array(self._gripper_cmd),
        )

    @classmethod
    def load(cls, path: str | Path) -> "dict[str, Any]":
        """Load a .npz diagnostic file and return a plain dict of arrays."""
        data = np.load(str(path), allow_pickle=True)
        return dict(data)

    # ── Quick summary ─────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary string (call after rollout)."""
        n = len(self._t_tick)
        if n == 0:
            return "DiagnosticLogger: 0 ticks recorded"

        dur = self._t_tick[-1] - self._t_tick[0]
        actual_hz = (n - 1) / dur if dur > 0 else float("nan")
        mean_overrun = float(np.mean(self._tick_overrun_ms))
        n_infer = sum(self._infer_recv)
        mean_sched = float(np.mean(self._n_scheduled)) if n_infer > 0 else float("nan")
        n_stale = sum(self._all_stale)

        return (
            f"DiagnosticLogger: {n} ticks over {dur:.1f}s "
            f"(actual freq={actual_hz:.1f}Hz, "
            f"mean_overrun={mean_overrun:.1f}ms, "
            f"inference_cycles={n_infer}, "
            f"mean_scheduled={mean_sched:.1f}, "
            f"all_stale={n_stale})"
        )
