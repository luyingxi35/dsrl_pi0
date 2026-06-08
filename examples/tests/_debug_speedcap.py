#!/usr/bin/env python3
"""Debug script: simulate update_waypoints speed cap to understand why robot doesn't move."""
import numpy as np

# Parameters matching the real test
action_scale = 0.1
MAX_JOINT_DELTA = 0.2 * action_scale  # 0.02 rad/step
dt_step = 0.1  # 10 Hz
robot_action_latency = 0.20

# Simulate pi0 output: constant velocity for joint 0
new_actions = [np.array([0.5] * 7 + [0.0]) for _ in range(8)]

# Current joint position
curr_joints = np.zeros(7)

# Bug1 fix: integrate full chunk
running = curr_joints.copy()
all_abs = []
for a in new_actions:
    vel = np.clip(a[:-1], -1.0, 1.0)
    running = running + vel * MAX_JOINT_DELTA
    all_abs.append(running.copy())
all_abs = np.array(all_abs)

print("All absolute positions (joint 0):")
for i, p in enumerate(all_abs):
    print(f"  action[{i}]: j0 = {p[0]:.4f} rad")

# Timestamps
t_obs = 100.0
action_timestamps = t_obs + np.arange(8) * dt_step
print(f"\naction_timestamps: {action_timestamps}")

# Assume 3 actions are stale
curr_time = t_obs + 0.3
is_new = action_timestamps > (curr_time + 0.01)
print(f"is_new: {is_new}  (sum={np.sum(is_new)})")

# Compute arm_times and arm_positions (what gets sent to NUC)
new_t = action_timestamps[is_new]
arm_times = new_t - robot_action_latency
arm_positions = all_abs[is_new]
print(f"\narm_times (wall-clock): {arm_times}")
print(f"arm_positions (joint 0): {arm_positions[:, 0]}")

# === Simulate update_waypoints on NUC ===
mono_times = arm_times.copy()  # simplified: offset = 0
curr_time_mono = curr_time

print(f"\n=== update_waypoints analysis ===")
print(f"curr_time (mono): {curr_time_mono}")
print(f"first waypoint time: {mono_times[0]}")
print(f"Gap from curr_time to first waypoint: {mono_times[0] - curr_time_mono:.4f}s")

# C0 continuity: interpolate current position
# Assume old trajectory ended at all_abs[2] (last stale position)
curr_pos = all_abs[2]
print(f"curr_pos (j0): {curr_pos[0]:.4f}")

times_in = mono_times.copy()
positions_in = arm_positions.copy()

# Step 1: C0 prepend
if curr_time_mono < times_in[0]:
    times_in = np.concatenate([[curr_time_mono], times_in])
    positions_in = np.vstack([curr_pos[None], positions_in])
    print(f"\nAfter C0 prepend: {len(times_in)} waypoints")
    print(f"  times: {times_in}")
    print(f"  positions j0: {positions_in[:, 0]}")

# Step 2: Speed cap
max_joint_speed = 3.0
print(f"\nSpeed cap analysis (max_joint_speed={max_joint_speed} rad/s):")
for i in range(1, len(times_in)):
    dt = times_in[i] - times_in[i - 1]
    max_delta = float(np.max(np.abs(positions_in[i] - positions_in[i - 1])))
    required_dt = max_delta / max_joint_speed
    old_time = times_in[i]
    if required_dt > dt:
        times_in[i:] = times_in[i:] + (required_dt - dt)
    print(f"  seg[{i-1}->{i}]: dt={dt:.4f}s, max_delta={max_delta:.6f}rad, "
          f"required_dt={required_dt:.6f}s, extended={required_dt > dt}, "
          f"time was {old_time:.4f} -> {times_in[i]:.4f}")

print(f"\nAfter speed cap:")
print(f"  times:  {times_in}")
print(f"  pos j0: {positions_in[:, 0]}")
print(f"  Last waypoint time: {times_in[-1]:.4f}")
print(f"  Total duration: {times_in[-1] - times_in[0]:.4f}s")

# === What does the 200Hz controller see? ===
print(f"\n=== 200Hz controller view ===")
for t_query in np.arange(curr_time_mono, curr_time_mono + 1.5, 0.1):
    if t_query <= times_in[0]:
        pos = positions_in[0]
        status = "CLAMP to first"
    elif t_query >= times_in[-1]:
        pos = positions_in[-1]
        status = "CLAMP to last (HOLD)"
    else:
        idx = int(np.searchsorted(times_in, t_query, side="right")) - 1
        t0, t1 = times_in[idx], times_in[idx + 1]
        alpha = (t_query - t0) / (t1 - t0)
        pos = (1 - alpha) * positions_in[idx] + alpha * positions_in[idx + 1]
        status = f"interp [{idx}->{idx+1}]"
    print(f"  t={t_query:.2f}: j0={pos[0]:.6f}  ({status})")

# === Now simulate the REAL scenario: new chunk arrives every ~0.8s ===
print(f"\n\n=== MULTI-CHUNK SIMULATION ===")
print(f"Simulating 3 consecutive chunks, each 8 actions at 10Hz...")
print(f"Each chunk: t_obs = previous_chunk_start + 0.8s")
print()

for chunk_idx in range(3):
    t_obs_chunk = 100.0 + chunk_idx * 0.8
    action_ts = t_obs_chunk + np.arange(8) * dt_step
    
    # When inference result arrives, curr_time is typically t_obs + ~0.7s
    curr_time_chunk = t_obs_chunk + 0.7
    is_new_mask = action_ts > (curr_time_chunk + 0.01)
    
    # Integrate
    running = np.zeros(7)  # simplified
    abs_pos = []
    for a in new_actions:
        vel = np.clip(a[:-1], -1.0, 1.0)
        running = running + vel * MAX_JOINT_DELTA
        abs_pos.append(running.copy())
    abs_pos = np.array(abs_pos)
    
    n_new = int(np.sum(is_new_mask))
    arm_t = action_ts[is_new_mask] - robot_action_latency
    arm_p = abs_pos[is_new_mask]
    
    print(f"Chunk {chunk_idx}: t_obs={t_obs_chunk:.1f}, curr_time={curr_time_chunk:.1f}, "
          f"is_new_count={n_new}")
    print(f"  arm_times: {arm_t}")
    print(f"  arm_pos j0: {arm_p[:, 0]}")
    
    # Check: are arm_times in the PAST relative to curr_time?
    for i, t in enumerate(arm_t):
        delta = t - curr_time_chunk
        print(f"    waypoint[{i}]: time={t:.3f}, delta_from_now={delta:.3f}s "
              f"{'PAST' if delta < 0 else 'FUTURE'}")