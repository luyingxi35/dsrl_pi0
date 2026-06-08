#!/usr/bin/env python3
"""Debug: simulate the actual timing to understand why only first action executes."""
import numpy as np

# Real parameters from output.txt
action_scale = 0.1
MAX_JOINT_DELTA = 0.2 * action_scale  # 0.02 rad/step
dt_step = 0.1  # 10 Hz control loop
robot_action_latency = 0.20
action_exec_latency = 0.01
execution_steps = 8
action_horizon = 8

# Typical pi0 inference latency: ~0.3-0.8s
# The key question: how many actions pass is_new filter?

print("=== is_new filter analysis ===")
print(f"dt_step={dt_step}, robot_action_latency={robot_action_latency}")
print(f"action_exec_latency={action_exec_latency}")
print(f"action_horizon={action_horizon}, execution_steps={execution_steps}")
print()

for infer_latency in [0.3, 0.5, 0.7, 0.8, 1.0]:
    t_obs = 100.0
    curr_time = t_obs + infer_latency
    action_timestamps = t_obs + np.arange(action_horizon) * dt_step
    is_new = action_timestamps > (curr_time + action_exec_latency)
    n_new = int(np.sum(is_new))
    
    if n_new > 0:
        arm_times = action_timestamps[is_new][:execution_steps] - robot_action_latency
        # Check how many arm_times are still in the future relative to curr_time
        n_future = int(np.sum(arm_times > curr_time))
        n_past = n_new - n_future
    else:
        arm_times = np.array([])
        n_future = 0
        n_past = 0
    
    print(f"infer_latency={infer_latency:.1f}s: "
          f"is_new={is_new.tolist()} (n_new={n_new}), "
          f"n_future={n_future}, n_past={n_past}")
    if n_new > 0:
        print(f"  arm_times relative to now: {(arm_times - curr_time).tolist()}")

print()
print("=== The REAL issue: arm_times in the past ===")
print()
print("After subtracting robot_action_latency (0.20s),")
print("many waypoint times fall in the PAST.")
print("The 200Hz controller will clamp to those positions IMMEDIATELY,")
print("causing jumps, then HOLD at the last position until next chunk.")
print()

# Simulate the actual scenario: inference takes ~0.5s
print("=== Detailed simulation: infer_latency=0.5s ===")
t_obs = 100.0
infer_latency = 0.5
curr_time = t_obs + infer_latency  # 100.5
action_timestamps = t_obs + np.arange(8) * dt_step
is_new = action_timestamps > (curr_time + 0.01)
print(f"action_timestamps: {action_timestamps}")
print(f"curr_time: {curr_time}")
print(f"is_new: {is_new} (n_new={np.sum(is_new)})")

new_t = action_timestamps[is_new][:execution_steps]
arm_times = new_t - robot_action_latency
print(f"arm_times: {arm_times}")
print(f"arm_times - curr_time: {arm_times - curr_time}")
print()

# Now simulate update_waypoints on NUC
# The first arm_time is 100.30, but curr_time is 100.50
# So curr_time > times[0] → C0 prepend does NOT happen
# The interpolator just gets these times directly
print("=== update_waypoints behavior ===")
print(f"curr_time_mono ≈ {curr_time}")
print(f"First waypoint time: {arm_times[0]}")
print(f"curr_time > first waypoint time? {curr_time > arm_times[0]}")
print()
print("If curr_time > times[0]:")
print("  - C0 prepend is SKIPPED (condition: curr_time < times[0])")
print("  - The interpolator replaces the trajectory with these past-future waypoints")
print("  - The 200Hz controller at t=100.50 sees:")
print()

times_in = arm_times.copy()
positions_in = np.array([[0.03 + i*0.01]*7 for i in range(len(times_in))])

for t_query in [100.50, 100.55, 100.60, 100.65, 100.70, 100.80, 101.0]:
    if t_query <= times_in[0]:
        pos_val = positions_in[0, 0]
        status = "CLAMP to first (JUMP!)"
    elif t_query >= times_in[-1]:
        pos_val = positions_in[-1, 0]
        status = "CLAMP to last (HOLD)"
    else:
        idx = int(np.searchsorted(times_in, t_query, side="right")) - 1
        t0, t1 = times_in[idx], times_in[idx + 1]
        alpha = (t_query - t0) / (t1 - t0)
        pos_val = (1 - alpha) * positions_in[idx, 0] + alpha * positions_in[idx + 1, 0]
        status = f"interp [{idx}->{idx+1}]"
    print(f"  t={t_query:.2f}: j0={pos_val:.4f}  ({status})")

print()
print("=== KEY INSIGHT ===")
print("When arm_times[0] < curr_time (first waypoint is in the past),")
print("the 200Hz controller IMMEDIATELY jumps through all past waypoints")
print("(they are clamped or interpolated instantly since they're all in the past)")
print("and then HOLDS at the last waypoint position.")
print("This means the robot instantly snaps to the LAST action's position,")
print("skipping all intermediate motion → appears as 'jump'.")
print()
print("Then it holds there until the next chunk arrives → no smooth motion.")
print("Each chunk: instant jump to final position, then hold.")