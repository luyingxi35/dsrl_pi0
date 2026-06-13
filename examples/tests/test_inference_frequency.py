import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.evaluate_policy_real import (
    _count_future_waypoints,
    _future_waypoint_horizon_s,
    build_parser as build_dsrl_parser,
)


def test_dsrl_eval_no_longer_accepts_inference_frequency():
    parser = build_dsrl_parser()

    try:
        parser.parse_args([
            "--restore_path", "dummy",
            "--inference_frequency_hz", "3",
        ])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("DSRL eval should reject --inference_frequency_hz")


def test_dsrl_eval_accepts_both_timing_modes():
    parser = build_dsrl_parser()

    train_args = parser.parse_args([
        "--restore_path", "dummy",
        "--dsrl_eval_timing_mode", "train",
    ])
    low_watermark_args = parser.parse_args([
        "--restore_path", "dummy",
        "--dsrl_eval_timing_mode", "low_watermark",
        "--min_future_actions", "2",
        "--min_future_horizon_s", "0.25",
    ])

    assert train_args.dsrl_eval_timing_mode == "train"
    assert low_watermark_args.dsrl_eval_timing_mode == "low_watermark"
    assert low_watermark_args.min_future_actions == 2
    assert low_watermark_args.min_future_horizon_s == 0.25


def test_future_waypoint_low_watermark_helpers():
    now = 10.0
    timestamps = [9.5, 10.1, 10.3, 10.6]

    assert _count_future_waypoints(timestamps, now) == 3
    assert abs(_future_waypoint_horizon_s(timestamps, now) - 0.6) < 1e-9
    assert _count_future_waypoints([9.0, 9.5], now) == 0
    assert _future_waypoint_horizon_s([9.0, 9.5], now) == 0.0
