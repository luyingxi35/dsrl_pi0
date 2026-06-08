# Evaluation Code Tests & Diagnostics

测试分三个级别，逐级递进：

---

## 1. 单元测试（无需机器人）

在 dsrl_pi0 根目录运行（`conda activate dsrl_pi0`）：

```bash
cd ~/yingxi/dsrl_pi0
python3 examples/tests/test_state_interpolator.py
python3 examples/tests/test_action_logic.py
python3 examples/tests/test_camera_timestamps.py
```

**预期输出：** 所有 `[PASS]` 行出现，无 `AssertionError`。

---

## 2. 预飞检查（需机器人连接，不运动）

需要：NUC 运行 DROID server，相机已连接，机械臂静止。

```bash
cd ~/yingxi/dsrl_pi0
python3 examples/tests/test_preflight.py \
    --wrist_camera_id 17396664 \
    --wrist_obs_latency 0.125
```

**预期结果：**

| 检查项 | 期望值 |
|--------|--------|
| Camera timestamp keys | 包含 `*_read_end` 字段 |
| t_obs drift | ≈ wrist_obs_latency（±100ms 以内） |
| State history entries | ≥ 50 / 100 requested |
| State history dt | ≈ 5ms（±2ms），符合 200Hz |
| Joint positions | 非 −1.0 的合理关节角 |
| Gripper position | 0～1 之间的值 |

用 `t_obs drift` 来**标定 wrist_camera_obs_latency**：
- 如果 drift 比配置的 latency 大 50ms 以上，说明 latency 偏小，需增大
- 如果 drift 比配置的 latency 小 50ms 以上，说明 latency 偏大，需减小

---

## 3. 冒烟测试（需机器人，极慢运动）

`action_scale=0.1` → 最大 0.02 rad/step，极慢且安全。

```bash
cd ~/yingxi/dsrl_pi0
python3 examples/evaluate_pi0_real.py \
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
```

**预期 log 输出：**
- `t_obs drift: X ms` — X 应 ≈ `wrist_camera_obs_latency × 1000`
- `State history: N entries` — N ≥ 50
- 5秒后 `reason=timeout`，episode 正常结束，无报错

---

---

## 4. 深度诊断（需机器人，产生轨迹数据 + 可视化）

### 4a. 采集诊断数据

在 `--diagnostic_dir` 目录保存每个 episode 的 `.npz` 诊断文件：

```bash
python3 examples/evaluate_pi0_real.py \
    --instruction "test" \
    --eval_episodes 1 \
    --max_duration_s 10 \
    --action_scale 0.5 \
    --execution_steps 8 \
    --control_frequency_hz 10 \
    --use_wrist_camera 1 \
    --policy_host 127.0.0.1 --policy_port 8000 \
    --diagnostic_dir ./logs/diagnostics
```

### 4b. 生成可视化 PDF

```bash
python3 examples/tests/visualize_rollout.py \
    --npz ./logs/diagnostics/episode_000.npz \
    --output ./logs/diagnostics/episode_000_analysis.pdf
```

**生成 6 张图：**

| 图 | 内容 | 验证什么 |
|----|------|---------|
| Fig 1 | 7关节轨迹（planned vs actual） | 机器人是否跟随命令；积分有无突变 |
| Fig 2 | Action chunk 利用率柱状图 | is_new 过滤多少；execution_steps 截断多少 |
| Fig 3 | Tick 时长直方图 + 频率漂移 | 控制频率是否稳定 10Hz |
| Fig 4 | t_obs 漂移 + state buffer 覆盖 | 相机时间戳是否正确；buffer 是否充足 |
| Fig 5 | 跨 chunk 的位置连续性 | 相邻 chunk 无跳跃（积分基准正确） |
| Fig 6 | 夹爪指令时序 | 夹爪是否按预期开关 |

**Fig 1 解读指南：**
- Blue（planned）和 Red（actual）紧密重合 → 机器人执行正常
- Blue 有大幅跳跃（chunk 边界处）→ 积分起点错误（已知 bug 检测）
- Red 滞后 Blue 固定时间 → robot_action_latency 可调

---

## 已知限制

| 限制 | 影响 | 严重程度 |
|------|------|---------|
| RealSense 若不在 DROID `camera_readers` 中，无硬件时间戳 | t_obs 使用 `time.time() - latency` 近似 | 低（可接受的近似） |
| `StateInterpolator` 重复时间戳无除零保护 | 200Hz 下概率极低 | 低 |
| HighFreqController gRPC 错误静默 pass | 掉线时不报错，下 tick 自动重试 | 中（可加 log） |
