#!/usr/bin/env python3
"""Preflight checks for the pi0 real-robot evaluation pipeline.

Verifies the full stack WITHOUT motion: camera timestamps, t_obs calibration,
HighFreqController state buffer population, and state history timing.

Requires:
  - NUC running DROID server  (python scripts/server/run_server.py)
  - Cameras connected
  - Mechanical arm at rest (no movement commands sent)

Run:
    cd ~/yingxi/dsrl_pi0
    conda activate dsrl_pi0
    python3 examples/tests/test_preflight.py \\
        --wrist_camera_id 17396664 \\
        --wrist_obs_latency 0.125
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


def check_camera_timestamps(env, args) -> None:
    """[1/4] Verify camera timestamp keys exist in obs['timestamp']['cameras']."""
    logging.info("\n[1/4] Camera timestamp structure")

    from examples.utils.real_robot_common import _find_camera_ts_ms

    obs = env.get_observation()
    cam_ts = obs.get("timestamp", {}).get("cameras", {})
    all_keys = sorted(cam_ts.keys())
    logging.info("All camera timestamp keys: %s", all_keys)

    for cam_id, label in [
        (args.wrist_camera_id,    "wrist (ZedMini)"),
        (args.exterior_camera_id, "exterior (RealSense)"),
    ]:
        if not cam_id:
            continue
        ts_ms = _find_camera_ts_ms(cam_ts, cam_id)
        if ts_ms is not None:
            age_ms = time.time() * 1000.0 - ts_ms
            logging.info("  %-25s (%s): read_end age = %.1f ms", label, cam_id, age_ms)
        else:
            logging.warning(
                "  %-25s (%s): NO hardware timestamp found.\n"
                "    → t_obs will use time.time() - latency fallback.\n"
                "    → This is expected if RealSense is not in DROID camera_readers.",
                label, cam_id,
            )
    logging.info("  [OK] camera timestamp check complete")


def check_t_obs_calibration(env, args) -> float:
    """[2/4] Verify t_obs drift matches configured obs_latency."""
    logging.info("\n[2/4] t_obs calibration")

    from examples.utils.real_robot_common import extract_observation_eval

    obs = env.get_observation()
    _, t_obs = extract_observation_eval(
        args.wrist_camera_id,
        args.exterior_camera_id,
        obs,
        wrist_obs_latency=args.wrist_obs_latency,
        exterior_obs_latency=args.exterior_obs_latency,
    )
    drift_ms = (time.time() - t_obs) * 1000.0
    expected_ms = args.wrist_obs_latency * 1000.0

    logging.info("t_obs drift from now: %.1f ms", drift_ms)
    logging.info("Expected (wrist_obs_latency): %.0f ms", expected_ms)

    if abs(drift_ms - expected_ms) <= 100.0:
        logging.info("  [OK] drift within ±100ms of configured latency")
    else:
        logging.warning(
            "  [WARN] drift deviates >100ms from configured latency.\n"
            "    Measured: %.1f ms  Configured: %.0f ms\n"
            "    → Increase wrist_obs_latency if drift > expected.\n"
            "    → Decrease wrist_obs_latency if drift < expected.",
            drift_ms, expected_ms,
        )

    return drift_ms


def check_state_history(env, args) -> None:
    """[3/4] Start HighFreqController and verify state history is populated."""
    logging.info("\n[3/4] HighFreqController state history")

    logging.info("Starting trajectory controller at 200 Hz...")
    env._robot.start_trajectory_controller(200.0)
    time.sleep(0.5)   # wait 500ms → expect ~100 entries at 200Hz

    times_h, joints_h, gripper_h = env._robot.get_state_history(n=100)
    n = len(times_h)
    logging.info("Entries returned: %d / 100 requested", n)

    if n < 50:
        logging.warning(
            "  [WARN] Only %d entries — HighFreqController may not be running at 200Hz", n
        )
    else:
        logging.info("  [OK] Sufficient entries (%d ≥ 50)", n)

    # Validate joint positions
    if joints_h:
        last_joints = joints_h[-1]
        if all(v == -1.0 for v in last_joints):
            logging.warning(
                "  [WARN] All joint positions are -1.0 sentinel — "
                "Polymetis get_joint_positions() may be failing"
            )
        else:
            logging.info(
                "  Current joint positions: %s",
                [round(v, 3) for v in last_joints],
            )

    # Validate gripper
    if gripper_h:
        g = gripper_h[-1]
        if g < 0:
            logging.warning(
                "  [WARN] Gripper position is -1 sentinel — "
                "Polymetis GripperInterface.get_state() may be failing"
            )
        else:
            logging.info("  Current gripper position: %.3f (0=open, 1=closed)", g)


def check_state_history_timing(env, args) -> None:
    """[4/4] Verify state history timestamps are consistent with 200Hz."""
    logging.info("\n[4/4] State history timing")

    times_h, joints_h, gripper_h = env._robot.get_state_history(n=50)

    if len(times_h) < 10:
        logging.warning("  Not enough entries for timing analysis (%d < 10)", len(times_h))
        return

    dts_ms = np.diff(times_h) * 1000.0   # inter-entry intervals in ms
    avg_dt = float(np.mean(dts_ms))
    std_dt = float(np.std(dts_ms))
    max_dt = float(np.max(dts_ms))
    expected_dt_ms = 1000.0 / 200.0      # 5ms at 200Hz

    logging.info(
        "Inter-entry dt: mean=%.2f ms, std=%.2f ms, max=%.2f ms  (expected ~%.1f ms @ 200Hz)",
        avg_dt, std_dt, max_dt, expected_dt_ms,
    )

    if abs(avg_dt - expected_dt_ms) <= 2.0:
        logging.info("  [OK] mean dt consistent with 200Hz (within ±2ms)")
    else:
        logging.warning(
            "  [WARN] mean dt=%.2f ms deviates >2ms from 200Hz=%.1f ms\n"
            "    → System may be under CPU load, or NUC loop is slower than expected.",
            avg_dt, expected_dt_ms,
        )

    if max_dt > 20.0:
        logging.warning(
            "  [WARN] max inter-entry dt=%.1f ms — occasional scheduling jitter detected", max_dt
        )

    # Check monotonically increasing timestamps
    non_mono = np.sum(np.diff(times_h) < 0)
    if non_mono > 0:
        logging.warning("  [WARN] %d non-monotonic timestamps in history", non_mono)
    else:
        logging.info("  [OK] timestamps monotonically increasing")


def run_preflight(args: argparse.Namespace) -> None:
    from droid.robot_env import RobotEnv

    logging.info("=== Pi0 Eval Preflight Checks ===")
    logging.info("Connecting to DROID RobotEnv...")
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    logging.info("Connected.\n")

    try:
        check_camera_timestamps(env, args)
        drift_ms = check_t_obs_calibration(env, args)
        check_state_history(env, args)
        check_state_history_timing(env, args)
    finally:
        try:
            env._robot.stop_trajectory_controller()
        except Exception:
            pass

    logging.info("\n=== Preflight complete ===")
    logging.info(
        "\nSummary: t_obs drift = %.1f ms  "
        "(target: wrist_obs_latency=%.0f ms)",
        drift_ms, args.wrist_obs_latency * 1000,
    )
    logging.info(
        "If drift differs from target by >50ms, adjust --wrist_camera_obs_latency "
        "when running evaluate_pi0_real.py."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preflight checks for the pi0 real-robot evaluation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python3 examples/tests/test_preflight.py \\
      --wrist_camera_id 17396664 \\
      --wrist_obs_latency 0.125
        """,
    )
    parser.add_argument(
        "--wrist_camera_id", default="17396664",
        help="ZedMini wrist camera serial number. Default: 17396664.",
    )
    parser.add_argument(
        "--exterior_camera_id", default=None,
        help="RealSense exterior camera serial number. Default: None (skip).",
    )
    parser.add_argument(
        "--wrist_obs_latency", default=0.125, type=float,
        help="Configured wrist camera obs latency (s) to compare against measured drift. "
             "Default: 0.125.",
    )
    parser.add_argument(
        "--exterior_obs_latency", default=0.175, type=float,
        help="Configured exterior camera obs latency (s). Default: 0.175.",
    )
    args = parser.parse_args()
    run_preflight(args)
