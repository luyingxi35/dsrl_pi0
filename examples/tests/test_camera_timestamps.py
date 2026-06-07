#!/usr/bin/env python3
"""Unit tests for camera timestamp lookup (_find_camera_ts_ms).

Tests that _find_camera_ts_ms correctly finds DROID camera timestamps
under various key formats (bare serial, prefixed serial, missing camera).

Run without a robot:
    cd ~/yingxi/dsrl_pi0
    python3 examples/tests/test_camera_timestamps.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.utils.real_robot_common import _find_camera_ts_ms


# ── Key format tests ──────────────────────────────────────────────────────────

def test_bare_serial() -> None:
    """Bare serial number as key prefix."""
    ts = {"17396664_read_end": 1718000000000.0}
    result = _find_camera_ts_ms(ts, "17396664")
    assert result == 1718000000000.0, f"Expected 1718000000000.0, got {result}"
    print("  [PASS] bare serial")


def test_zedmini_prefixed_serial() -> None:
    """ZED Mini prefixed key (e.g. zedmini_<serial>_read_end)."""
    ts = {"zedmini_17396664_read_end": 999.0}
    result = _find_camera_ts_ms(ts, "17396664")
    assert result == 999.0, f"Expected 999.0, got {result}"
    print("  [PASS] zedmini_ prefixed serial")


def test_zed_prefixed_serial() -> None:
    """Short 'zed_' prefix variant."""
    ts = {"zed_17396664_read_end": 777.0}
    result = _find_camera_ts_ms(ts, "17396664")
    assert result == 777.0, f"Expected 777.0, got {result}"
    print("  [PASS] zed_ prefixed serial")


def test_realsense_prefixed_serial() -> None:
    """RealSense prefix variant."""
    ts = {"realsense_241122302552_read_end": 42.5}
    result = _find_camera_ts_ms(ts, "241122302552")
    assert result == 42.5, f"Expected 42.5, got {result}"
    print("  [PASS] realsense_ prefixed serial")


def test_lookup_by_full_prefixed_id() -> None:
    """Looking up by the prefixed ID itself should also work."""
    ts = {"zedmini_17396664_read_end": 111.0}
    result = _find_camera_ts_ms(ts, "zedmini_17396664")
    assert result == 111.0, f"Expected 111.0, got {result}"
    print("  [PASS] lookup by prefixed ID")


def test_not_found_returns_none() -> None:
    """Missing camera should return None, not raise."""
    ts = {"other_cam_read_end": 1.0}
    result = _find_camera_ts_ms(ts, "17396664")
    assert result is None, f"Expected None, got {result}"
    print("  [PASS] not found returns None")


def test_empty_dict_returns_none() -> None:
    """Empty camera timestamp dict should return None."""
    result = _find_camera_ts_ms({}, "17396664")
    assert result is None
    print("  [PASS] empty dict returns None")


def test_custom_suffix_read_start() -> None:
    """Custom suffix '_read_start' instead of '_read_end'."""
    ts = {
        "17396664_read_start": 100.0,
        "17396664_read_end":   117.0,
    }
    r_start = _find_camera_ts_ms(ts, "17396664", suffix="_read_start")
    r_end   = _find_camera_ts_ms(ts, "17396664", suffix="_read_end")
    assert r_start == 100.0, f"Expected 100.0, got {r_start}"
    assert r_end   == 117.0, f"Expected 117.0, got {r_end}"
    print(f"  [PASS] custom suffix (read_start={r_start}, read_end={r_end})")


# ── t_obs calculation ─────────────────────────────────────────────────────────

def test_t_obs_calculation() -> None:
    """Verify t_obs = read_end_ms / 1000 - latency produces past timestamp."""
    import time

    # Simulate a real read_end_ms (current wall clock in ms)
    read_end_ms = time.time() * 1000.0
    wrist_obs_latency = 0.125  # 125ms

    ts = {"17396664_read_end": read_end_ms}
    ts_ms = _find_camera_ts_ms(ts, "17396664")
    assert ts_ms is not None

    t_obs = ts_ms / 1000.0 - wrist_obs_latency
    drift_ms = (time.time() - t_obs) * 1000.0

    # drift should be approximately wrist_obs_latency (125ms)
    assert abs(drift_ms - 125.0) < 5.0, (
        f"t_obs drift should be ~125ms, got {drift_ms:.1f}ms"
    )
    print(f"  [PASS] t_obs calculation: drift={drift_ms:.1f}ms ≈ 125ms")


def test_realsense_no_droid_reader() -> None:
    """RealSense with no DROID camera reader: timestamp should not be found.

    In a typical DROID setup the RealSense is not captured by DROID's
    camera_readers, so its timestamp will be absent from obs['timestamp']['cameras'].
    The code falls back to time.time() - exterior_obs_latency in that case.
    This test documents that behaviour.
    """
    ts = {}   # RealSense not in DROID camera_readers → no entry
    result = _find_camera_ts_ms(ts, "241122302552")
    assert result is None
    print("  [INFO] RealSense timestamp absent → t_obs fallback will be used "
          "(time.time() - exterior_obs_latency)")


if __name__ == "__main__":
    print("=== Camera Timestamp Tests ===\n")
    test_bare_serial()
    test_zedmini_prefixed_serial()
    test_zed_prefixed_serial()
    test_realsense_prefixed_serial()
    test_lookup_by_full_prefixed_id()
    test_not_found_returns_none()
    test_empty_dict_returns_none()
    test_custom_suffix_read_start()
    test_t_obs_calculation()
    test_realsense_no_droid_reader()
    print("\nAll tests completed.")
