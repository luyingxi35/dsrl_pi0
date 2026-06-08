#!/usr/bin/env python3
"""Debug: trace the EXACT flow for infer_latency=0.3s where n_future=2."""
import numpy as np

action_scale = 0.1
MAX_JOINT_DELTA = 0.2 * action_scale  # 0.02
dt_step = 0.1
robot_action_latency = 0.20
action_exec_latency = 0.01
execution_steps = 8

t_obs = 100.0
infer_latency = 0.3
curr_time = t_obs + infer_latency  # 100.3

action_timestamps = t_obs + np.arange(8) * dt_step
is_new = action_timestamps > (curr_time + action_exec_latency)
print(f"action_timestamps: {action_timestamps}")
print(f"curr_time: {curr_time}")
print(f"is_new: {is_new} (n_new={np.sum(is_new)})")

# Bug1 fix: integrate full chunk
curr_joints = np.array([0.0]*7)
new_actions = [np.array([0.5]*7 + [0.0]) for _ in range(8)]
running = curr_joints.copy()
all_abs = []
for a in new_actions:
    vel = np.clip(a[:-1], -1.0, 1.0)
    running = running + vel * MAX_JOINT_DELTA
    all_abs.append(running.copy())
all_abs = np.array(all_abs)

# is_new filter
arm_positions = all_abs[is_new][:execution_steps]
new_t = action_timestamps[is_new][:execution_steps]
arm_times = new_t - robot_action_latency

print(f"\narm_times: {arm_times}")
print(f"arm_times - curr_time: {arm_times - curr_time}")
print(f"arm_positions j0: {arm_positions[:, 0]}")

# What happens in update_waypoints:
# Step 1: C0 continuity
# interp is currently empty (first chunk), so curr_pos = None
# → C0 prepend is SKIPPED
# OR: interp has old trajectory ending at some position
# Let's trace both cases

print("\n=== Case 1: First chunk (interp is empty) ===")
print("curr_pos = None → C0 prepend skipped")
print("Speed cap: max_delta = 0.01 rad between consecutive waypoints")
print("dt = 0.1s between waypoints")
print("required_dt = 0.01/3.0 = 0.0033s << 0.1s → no extension")
print()
print("Times after update_waypoints: [100.1, 100.2, 100.3, 100.4]")
print()
print("200Hz controller at curr_time=100.3:")
print("  t=100.30 → interp [1->2] → j0=0.06")
print("  t=100.35 → interp [2->3] → j0=0.065")
print("  t=100.40 → CLAMP to last → j0=0.08")
print("  t=100.50 → CLAMP to last → j0=0.08 (HOLD)")
print()
print("Result: Robot moves from 0.06 to 0.08 in 0.1s, then HOLDS.")
print("The first two waypoints (100.1, 100.2) are ALREADY IN THE PAST")
print("and are instantly clamped through.")
print()

print("=== Case 2: Second chunk arrives (interp has old trajectory) ===")
print()
print("The REAL problem: update_waypoints calls self.__call__(curr_time)")
print("to get curr_pos. If the old trajectory's times[-1] < curr_time,")
print("it returns the LAST position of the OLD trajectory.")
print("But if times[-1] > curr_time, it interpolates correctly.")
print()

# The critical insight: when we call update_waypoints, we REPLACE
# the entire trajectory. The old trajectory's tail (future waypoints)
# is DISCARDED. So the C0 continuity tries to bridge from the old
# position to the new waypoints.

# Let's trace what happens when the second chunk arrives
print("=== Simulating chunk 2 arrival ===")
# Chunk 2: t_obs = 100.8 (execution_steps=8 → next inference at t=100+8*0.1=100.8)
t_obs_2 = 100.8
infer_latency_2 = 0.5  # second inference might take different time
curr_time_2 = t_obs_2 + infer_latency_2  # 101.3

action_ts_2 = t_obs_2 + np.arange(8) * dt_step
is_new_2 = action_ts_2 > (curr_time_2 + action_exec_latency)
n_new_2 = int(np.sum(is_new_2))

print(f"t_obs_2: {t_obs_2}, curr_time_2: {curr_time_2}")
print(f"action_ts_2: {action_ts_2}")
print(f"is_new_2: {is_new_2} (n_new={n_new_2})")

arm_times_2 = action_ts_2[is_new_2][:execution_steps] - robot_action_latency
print(f"arm_times_2: {arm_times_2}")
print(f"arm_times_2 - curr_time_2: {arm_times_2 - curr_time_2}")

print()
print("ALL arm_times for chunk 2 are in the PAST!")
print(f"  arm_times_2 max = {arm_times_2[-1]:.1f}, curr_time_2 = {curr_time_2:.1f}")
print("  → The 200Hz controller will INSTANTLY jump to the last position")
print("  → Then HOLD until the next chunk")
print()
print("This explains the 'jump' behavior: each chunk causes an instant")
print("position change, with no smooth interpolation in between.")

print()
print("========================================")
print("ROOT CAUSE SUMMARY")
print("========================================")
print()
print("The fundamental problem is that robot_action_latency=0.20s is")
print("SUBTRACTED from already-past timestamps, pushing arm_times")
print("even further into the past. Combined with inference latency")
print("(0.3-0.7s), by the time waypoints arrive at the NUC, most")
print("waypoint times are already past.")
print()
print("The 200Hz controller then:")
print("1. Sees all waypoint times <= curr_time")
print("2. Clamps to positions[-1] (last waypoint) immediately")
print("3. Holds at that position until next chunk arrives")
print()
print("Result: Robot appears to 'jump' to each chunk's final position")
print("instead of smoothly interpolating through intermediate waypoints.")