## 1. 单元测试（无需机器人）
```text
(dsrl_pi0) robot@robot:~/yingxi/dsrl_pi0$ cd ~/yingxi/dsrl_pi0
python3 examples/tests/test_state_interpolator.py
python3 examples/tests/test_action_logic.py
python3 examples/tests/test_camera_timestamps.py
=== StateInterpolator Unit Tests ===

  [PASS] midpoint interpolation
  [PASS] clamp before start
  [PASS] clamp after end
  [PASS] proprioceptive_latency shift
  [PASS] gripper_latency shift
  [PASS] empty history returns None
  [PASS] single entry clamped correctly
  [PASS] four-point accuracy
  [PASS] duplicate timestamps handled (result j[0]=0.0)

All tests completed.
=== Action Logic Tests ===

  is_new mask: [False, False, True, True, True, True, True, True]
  6/8 actions pass is_new filter
  [PASS] is_new filter with past t_obs
Traceback (most recent call last):
  File "/home/robot/yingxi/dsrl_pi0/examples/tests/test_action_logic.py", line 186, in <module>
    test_is_new_all_new_when_t_obs_is_now()
  File "/home/robot/yingxi/dsrl_pi0/examples/tests/test_action_logic.py", line 68, in test_is_new_all_new_when_t_obs_is_now
    assert is_new[0],  "action[0] should be new when t_obs ≈ now"
           ~~~~~~^^^
AssertionError: action[0] should be new when t_obs ≈ now
=== Camera Timestamp Tests ===

  [PASS] bare serial
  [PASS] zedmini_ prefixed serial
  [PASS] zed_ prefixed serial
  [PASS] realsense_ prefixed serial
  [PASS] lookup by prefixed ID
  [PASS] not found returns None
  [PASS] empty dict returns None
  [PASS] custom suffix (read_start=100.0, read_end=117.0)
  [PASS] t_obs calculation: drift=125.0ms ≈ 125ms
  [INFO] RealSense timestamp absent → t_obs fallback will be used (time.time() - exterior_obs_latency)

All tests completed.
```

## 2. 预飞检查（需机器人连接，不运动）
```text
(dsrl_pi0) robot@robot:~/yingxi/dsrl_pi0$ python3 examples/tests/test_preflight.py \
    --wrist_camera_id 17396664 \
    --wrist_obs_latency 0.125
INFO  === Pi0 Eval Preflight Checks ===
INFO  Connecting to DROID RobotEnv...
Opening Zed:  17396664
Opening RealSense:  241122302552
[2026-06-08 09:46:42 UTC][ZED][INFO] Logging level INFO
[2026-06-08 09:46:44 UTC][ZED][INFO] [Init]  Camera successfully opened.
[2026-06-08 09:46:44 UTC][ZED][INFO] [Init]  Camera FW version: 1523
[2026-06-08 09:46:44 UTC][ZED][INFO] [Init]  Video mode: HD720@60
[2026-06-08 09:46:44 UTC][ZED][INFO] [Init]  Serial Number: S/N 17396664
[2026-06-08 09:46:44 UTC][ZED][INFO] [Init]  Depth mode selected: NEURAL. Ensure this mode matches your application's performance and accuracy requirements. See https://www.stereolabs.com/docs/depth-sensing/depth-modes for help.
INFO  Connected.

INFO  
[1/4] Camera timestamp structure
INFO  All camera timestamp keys: ['17396664_estimated_capture', '17396664_frame_received', '17396664_read_end', '17396664_read_start', 'realsense_241122302552_estimated_capture', 'realsense_241122302552_frame_received', 'realsense_241122302552_read_end', 'realsense_241122302552_read_start']
INFO    wrist (ZedMini)           (17396664): read_end age = 10.3 ms
INFO    [OK] camera timestamp check complete
INFO  
[2/4] t_obs calibration
INFO  t_obs drift from now: 128.1 ms
INFO  Expected (wrist_obs_latency): 125 ms
INFO    [OK] drift within ±100ms of configured latency
INFO  
[3/4] HighFreqController state history
INFO  Starting trajectory controller at 200 Hz...
INFO  Entries returned: 100 / 100 requested
INFO    [OK] Sufficient entries (100 ≥ 50)
INFO    Current joint positions: [0.036, -0.603, -0.026, -2.498, -0.021, 1.898, 0.041]
INFO    Current gripper position: 0.000 (0=open, 1=closed)
INFO  
[4/4] State history timing
INFO  Inter-entry dt: mean=5.00 ms, std=0.21 ms, max=5.88 ms  (expected ~5.0 ms @ 200Hz)
INFO    [OK] mean dt consistent with 200Hz (within ±2ms)
INFO    [OK] timestamps monotonically increasing
INFO  
=== Preflight complete ===
INFO  
Summary: t_obs drift = 128.1 ms  (target: wrist_obs_latency=125 ms)
INFO  If drift differs from target by >50ms, adjust --wrist_camera_obs_latency when running evaluate_pi0_real.py.
```

## 3. 冒烟测试（需机器人，极慢运动）

```text
(dsrl_pi0) robot@robot:~/yingxi/dsrl_pi0$ python3 examples/evaluate_pi0_real.py \
    --instruction "test" \
    --eval_episodes 1 \
    --max_duration_s 5 \
    --action_scale 0.1 \
    --execution_steps 4 \
    --control_frequency_hz 10 \
    --controller_frequency 200 \
    --use_wrist_camera 1 \
    --use_exterior_camera 0 \
    --policy_host 127.0.0.1 \
    --policy_port 8000 \
    --outputdir ./logs/smoke_test
INFO:root:ExecutionConfig: execution_steps=4 robot_action_latency=0.200s gripper_action_latency=0.200s action_exec_latency=0.010s controller_frequency=200Hz action_scale=0.10 (max_joint_delta=0.020 rad/step)
INFO:root:Camera obs latencies: wrist=0.080s exterior=0.080s | State latencies: proprioceptive=0.000s gripper=0.000s
INFO:root:Waiting for server at ws://127.0.0.1:8000...
INFO:root:OpenPI policy server metadata: {'action_horizon': 8, 'action_dim': 32}
Opening Zed:  17396664
Opening RealSense:  241122302552
[2026-06-08 09:49:15 UTC][ZED][INFO] Logging level INFO
[2026-06-08 09:49:17 UTC][ZED][INFO] [Init]  Camera successfully opened.
[2026-06-08 09:49:17 UTC][ZED][INFO] [Init]  Camera FW version: 1523
[2026-06-08 09:49:17 UTC][ZED][INFO] [Init]  Video mode: HD720@60
[2026-06-08 09:49:17 UTC][ZED][INFO] [Init]  Serial Number: S/N 17396664
[2026-06-08 09:49:17 UTC][ZED][INFO] [Init]  Depth mode selected: NEURAL. Ensure this mode matches your application's performance and accuracy requirements. See https://www.stereolabs.com/docs/depth-sensing/depth-modes for help.
INFO:root:Writing pi0 evaluation outputs to logs/smoke_test
INFO:root:Resetting DROID environment (before episode 0)...
pi0 eval episode 0: 0step [00:00, ?step/s]INFO:root:t_obs drift: 83.7 ms (camera frame age = obs_latency + any extra delay; tune *_camera_obs_latency if this deviates from expected latency)
pi0 eval episode 0: 50step [00:04, 10.00step/s]
INFO:root:Episode 0 done: success=False reason=timeout steps=50 duration=5.01s rate=0.000
INFO:root:Resetting DROID environment (after episode 0)...
INFO:root:Pi0 evaluation complete. Results: logs/smoke_test/eval_results.csv
Pi0 evaluation complete. Results: logs/smoke_test/eval_results.csv
```

## 4. 深度诊断（需机器人，产生轨迹数据 + 可视化）

