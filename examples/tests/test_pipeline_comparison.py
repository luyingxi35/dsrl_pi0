#!/usr/bin/env python3
"""
examples/tests/test_pipeline_comparison.py

Compare sim vs real action-processing pipelines for pi0_droid end-to-end.

Flow
────
1. ManiSkillRemoteEnv.reset() → raw obs  (qpos 9-D, ext 128×128, wrist 128×128)
   OR --mock mode for a quick offline run with synthetic data
2. LocalPi0Policy.infer()      → raw action chunk  (H, 8)  normalised velocities
   OR mock chunk sampled from Uniform(-0.3, 0.3) with gripper transition at step 4
3. SIM pipeline  (mirrors evaluate_pi0_sim.py):
     pi0_velocity_chunk_to_sim_actions(qpos, chunk, scale)
       → integrated & re-normalised delta targets  (H, 8)
     RealTimeActionChunker.step(sim_actions)
       → single blended action  (8,)  ← one env.step() call
4. REAL pipeline  (mirrors the action-integration logic in evaluate_pi0_real.py,
   WITHOUT is_new filtering — this test runs entirely in simulation where all
   inference is synchronous; wall-clock stale-action filtering is not applicable):
     integrate_joint_velocity_actions(joint_pos, chunk, max_delta)
       → abs joint angles  (H, 7)  rad  (full chunk integrated)
     arm_positions = all_abs[:execution_steps]      (N_sched, 7) rad
     gripper_vals  = binarize_and_clip_action(...)  for each step
5. Print both results and analyse differences.

Usage
─────
  # With real env + real model (slow, ~3 min to load both):
  conda run -n dsrl_pi0 python3 examples/tests/test_pipeline_comparison.py

  # Mock mode (fast, no subprocess required):
  conda run -n dsrl_pi0 python3 examples/tests/test_pipeline_comparison.py --mock

  # Custom checkpoint / robofac python:
  conda run -n dsrl_pi0 python3 examples/tests/test_pipeline_comparison.py \\
      --checkpoint /opt/yingxi/pi0_droid \\
      --robofac_python /opt/yingxi/envs/robofac/bin/python3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.sim.action_utils import RealTimeActionChunker, pi0_velocity_chunk_to_sim_actions
from examples.real.utils.real_robot_common import (
    binarize_and_clip_action,
    integrate_joint_velocity_actions,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants  (mirror evaluate_pi0_sim.py / evaluate_pi0_real.py defaults)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT       = "/opt/yingxi/pi0_droid"
DEFAULT_ROBOFAC_PY       = "/opt/yingxi/envs/robofac/bin/python3"
DEFAULT_INSTRUCTION      = "pick up the blue peg and insert it into the hole"
DEFAULT_ACTION_HORIZON   = 8
DEFAULT_ACTION_SCALE     = 0.5
DEFAULT_ACTION_CHUNK_DECAY = 0.01   # RealTimeActionChunker m
DEFAULT_EXECUTION_STEPS  = 6       # actions to schedule per inference in real eval
DROID_MAX_JOINT_DELTA    = 0.2     # rad per pi0 step (DROID training constant)
MANISKILL_JOINT_DELTA    = 0.1     # rad per sim step (ManiSkill controller unit)


# ──────────────────────────────────────────────────────────────────────────────
# Mock observation helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_mock_obs() -> dict[str, Any]:
    """Synthetic ManiSkillRemoteEnv observation.

    Joint config: Franka Panda neutral pose matching DROID_RESET_QPOS in
    mani_skill_client.py.  Gripper half-open (fingers at 0.02 m each).
    """
    rng = np.random.default_rng(42)
    qpos = np.array([
        0.0, -np.pi / 5, 0.0, -4 * np.pi / 5,
        0.0,  3 * np.pi / 5, np.pi,
        0.02, 0.02,             # two gripper finger joints (m)
    ], dtype=np.float32)
    ext   = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    return {"qpos": qpos, "ext": ext, "wrist": wrist}


def make_mock_pi0_actions(horizon: int = DEFAULT_ACTION_HORIZON) -> np.ndarray:
    """Synthetic pi0_droid action chunk — normalised joint velocities.

    Arm velocities are drawn from U(-0.3, +0.3), which is a realistic range for
    pi0_droid DROID-trained outputs.  Gripper: open for steps 0-3, close for 4+.
    """
    rng = np.random.default_rng(7)
    chunk = rng.uniform(-0.3, 0.3, (horizon, 8)).astype(np.float32)
    chunk[:4, 7] = 0.1   # open  (≤ 0.5)
    chunk[4:, 7] = 0.8   # close (> 0.5)
    return chunk


# ──────────────────────────────────────────────────────────────────────────────
# Step 1+2: observation and pi0 inference
# ──────────────────────────────────────────────────────────────────────────────

def get_obs_from_env(robofac_python: str) -> dict:
    """Boot ManiSkillRemoteEnv, reset, return raw obs dict."""
    from examples.sim.envs.mani_skill_client import ManiSkillRemoteEnv
    print("[env] Booting ManiSkillRemoteEnv (robofac subprocess)…", flush=True)
    env = ManiSkillRemoteEnv(robofac_python=robofac_python)
    print("[env] Resetting environment…", flush=True)
    env_obs, _ = env.reset()
    env.close()
    return env_obs


def run_pi0_inference(
    checkpoint: str,
    obs_dict: dict,
    instruction: str,
) -> np.ndarray:
    """Load LocalPi0Policy, run one inference, return raw action chunk (H, 8)."""
    from examples.sim.evaluate_pi0 import (
        LocalPi0Policy,
        _extract_sim_obs,
        _obs_to_pi0_input,
    )
    print("[pi0] Booting LocalPi0Policy subprocess (~60 s)…", flush=True)
    policy = LocalPi0Policy(checkpoint)

    qpos, ext_rgb, wrist_rgb = _extract_sim_obs(obs_dict)
    pi0_input = _obs_to_pi0_input(qpos, ext_rgb, wrist_rgb, instruction)

    print("[pi0] Running inference…", flush=True)
    raw = policy.infer(pi0_input)
    policy.close()

    return np.asarray(raw["actions"], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: SIM pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_sim_pipeline(
    qpos: np.ndarray,
    raw_actions: np.ndarray,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    action_scale: float = DEFAULT_ACTION_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply sim action processing.

    Parameters
    ----------
    qpos        : (9,) ManiSkill joint state (7 arm + 2 gripper fingers)
    raw_actions : (H, 8) normalised joint velocities from pi0

    Returns
    -------
    sim_chunk : (action_horizon, 8)  in ManiSkill normalised-delta space
    blended   : (8,)  single step output after RealTimeActionChunker
    """
    chunk = raw_actions[:action_horizon]
    # pi0 velocity → cumulative delta → normalise by MANISKILL_JOINT_DELTA
    #   running_joints[k] = running + clip(vel[k]) * DROID_MAX * scale
    #   sim_action[k,:7]  = (running_joints[k] - qpos[:7]) / MANISKILL_JOINT_DELTA
    sim_chunk = pi0_velocity_chunk_to_sim_actions(
        qpos, chunk, action_scale=action_scale
    )
    chunker = RealTimeActionChunker(
        action_horizon=action_horizon,
        action_dim=8,
        m=DEFAULT_ACTION_CHUNK_DECAY,
    )
    blended = chunker.step(sim_chunk)
    return sim_chunk, blended


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: REAL pipeline  (no is_new filter — synchronous sim context)
# ──────────────────────────────────────────────────────────────────────────────

def run_real_pipeline(
    joint_position: np.ndarray,
    raw_actions: np.ndarray,
    action_scale: float  = DEFAULT_ACTION_SCALE,
    execution_steps: int = DEFAULT_EXECUTION_STEPS,
) -> dict:
    """Simulate real-eval action integration without is_new filtering.

    In the real robot eval (evaluate_pi0_real.py), actions whose wall-clock
    target timestamps have already passed are discarded via is_new before being
    sent as waypoints.  That guard is a real-time concern only — it prevents
    the NUC's HighFreqController from receiving commands that are already stale.

    This test script runs entirely in simulation where inference is synchronous
    and there is no controller to protect, so we skip is_new and simply take
    the first execution_steps actions from the integrated chunk.

    Parameters
    ----------
    joint_position : (7,) joint angles at the time of observation (rad)
    raw_actions    : (H, 8) normalised joint velocities from pi0

    Returns (dict with intermediate values for analysis and printing)
    """
    max_joint_delta = DROID_MAX_JOINT_DELTA * action_scale  # rad/step = 0.10

    # Integrate the full chunk to get absolute joint angles (H, 7)
    all_abs = integrate_joint_velocity_actions(
        source_joint_position=joint_position,
        actions=raw_actions,
        max_joint_delta=max_joint_delta,
    )  # (H, 7) radians

    # Take the first execution_steps waypoints (no stale filtering in sim)
    n_scheduled   = min(execution_steps, len(all_abs))
    arm_positions = all_abs[:n_scheduled]                    # (N, 7) rad
    gripper_vals  = np.array([
        binarize_and_clip_action(raw_actions[i])[-1]
        for i in range(n_scheduled)
    ])

    return {
        "H":              len(raw_actions),
        "n_scheduled":    n_scheduled,
        "all_abs":        all_abs,
        "arm_positions":  arm_positions,
        "gripper_vals":   gripper_vals,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-printing
# ──────────────────────────────────────────────────────────────────────────────

def _sep(title: str = "") -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def _arm_row(vals: np.ndarray) -> str:
    return "  ".join(f"{v:+.4f}" for v in vals)


def print_raw_obs(qpos: np.ndarray, mock: bool) -> None:
    _sep("Initial observation  (from " + ("MOCK data" if mock else "ManiSkillRemoteEnv") + ")")
    print(f"  qpos arm    (7, rad)  : {np.round(qpos[:7], 4).tolist()}")
    print(f"  qpos fingers (2, m)   : {np.round(qpos[7:9], 4).tolist()}")
    gripper_norm = float(np.clip(np.mean(qpos[7:9]) / 0.04, 0.0, 1.0))
    print(f"  gripper_norm → pi0    : {gripper_norm:.4f}   (mean(fingers)/0.04, clip [0,1])")


def print_raw_actions(raw: np.ndarray, mock: bool) -> None:
    _sep("RAW pi0_droid output  (from " + ("MOCK chunk" if mock else "LocalPi0Policy") + ")")
    print(f"  shape : {raw.shape}    dtype : {raw.dtype}")
    print(f"  arm  cols 0-6 : range [{raw[:,:7].min():+.4f}, {raw[:,:7].max():+.4f}]"
          f"  mean-abs {np.abs(raw[:,:7]).mean():.4f}")
    print(f"  grip col   7  : {np.round(raw[:,7], 3).tolist()}")
    print()
    hdr = "  step │ " + "  ".join(f"  j{i+1} " for i in range(7)) + "  grip"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for i, row in enumerate(raw):
        print(f"  {i:4d} │ " + "  ".join(f"{v:+.3f}" for v in row))


def print_sim_results(qpos: np.ndarray, sim_chunk: np.ndarray, blended: np.ndarray) -> None:
    _sep("SIM pipeline  (evaluate_pi0_sim.py)")
    print("  Step A — pi0_velocity_chunk_to_sim_actions(qpos, chunk, scale=0.5)")
    print("    running_joints[k] = running + clip(vel[k]) * 0.10 rad")
    print("    sim_action[k,:7]  = (running_joints[k] − qpos[:7]) / 0.10")
    print("    → normalised delta targets for ManiSkill's joint-delta controller")
    print()
    hdr = "  step │ " + "  ".join(f"  j{i+1} " for i in range(7)) + " │ grip"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for i, row in enumerate(sim_chunk):
        grip_sym = "OPEN (+1)" if row[7] > 0 else "CLOS (−1)"
        print(f"  {i:4d} │ " + "  ".join(f"{v:+.4f}" for v in row[:7])
              + f" │ {row[7]:+.4f}  {grip_sym}")
    print()
    print("  Step B — RealTimeActionChunker.step(sim_chunk)  [m=0.01, first call]")
    print("    On the first call the buffer contains only this chunk;")
    print("    blended = sim_chunk[0]  (age=0 → weight=1, no prior chunks)")
    print()
    grip_sym = "OPEN (+1)" if blended[7] > 0 else "CLOS (−1)"
    print("  ★ FINAL SIM ACTION  (8,) — single vector passed to env.step():")
    print(f"    arm  : {_arm_row(blended[:7])}")
    print(f"    grip : {blended[7]:+.4f}  ({grip_sym})")
    print()
    sim_target_rad = qpos[:7] + blended[:7] * MANISKILL_JOINT_DELTA
    print("  Equivalent absolute joint target for later comparison:")
    print(f"    qpos[:7] + blended[:7]*0.10 = {np.round(sim_target_rad, 4).tolist()}")
    print()
    print("  Units: arm ∈ normalised joint-delta space  (value × 0.10 = Δrad)")
    print("         grip ∈ {+1.0 open, −1.0 closed}  (ManiSkill convention)")


def print_real_results(r: dict, joint_position: np.ndarray) -> None:
    _sep("REAL pipeline  (evaluate_pi0_real.py action integration — sim context)")
    print("  is_new filtering is OMITTED: this test runs in simulation where")
    print("  inference is synchronous and no NUC controller needs protecting.")
    print("  In real eval, is_new discards actions whose wall-clock timestamp")
    print("  has already passed by the time the inference result is drained.")
    print()
    print("  Step A — integrate_joint_velocity_actions(joint_pos, chunk, max_delta=0.10)")
    print("    running_joints[k] = running + clip(vel[k]) * 0.10 rad   (same as SIM arm)")
    print("    → absolute joint angles in radians  (H, 7)")
    print()
    print("  Full integrated chunk:")
    hdr = "  step │ " + "  ".join(f"  j{i+1}  " for i in range(7))
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for i, row in enumerate(r["all_abs"]):
        sched = "  ← scheduled" if i < r["n_scheduled"] else ""
        print(f"  {i:4d} │ " + "  ".join(f"{v:+.4f}" for v in row) + sched)
    print()
    print(f"  Step B — take first execution_steps={DEFAULT_EXECUTION_STEPS} waypoints")
    print(f"    (in real eval these are sent as timestamped waypoints to the NUC)")
    print()
    print("  ★ FINAL REAL WAYPOINTS  (N={n}, 7) rad  +  gripper per waypoint:".format(
        n=r["n_scheduled"]))
    for i, (pos, g) in enumerate(zip(r["arm_positions"], r["gripper_vals"])):
        g_sym = "CLOS (1.0)" if g > 0.5 else "OPEN (0.0)"
        print(f"    [{i}]  arm  : {_arm_row(pos)}")
        print(f"          grip : {g:.1f}  ({g_sym})")
    print()
    print("  Units: arm ∈ absolute joint angles [rad]")
    print("         grip ∈ {0.0 open, 1.0 closed}  (DROID position convention)")


def print_analysis(
    qpos: np.ndarray,
    raw_actions: np.ndarray,
    sim_blended: np.ndarray,
    r: dict,
) -> None:
    _sep("ANALYSIS — Key differences between SIM and REAL pipelines")

    print("""
  ┌────────────────────────┬──────────────────────────────┬──────────────────────────────┐
  │ Dimension              │ SIM  pipeline                │ REAL pipeline                │
  ├────────────────────────┼──────────────────────────────┼──────────────────────────────┤
  │ Output quantity        │ 1 blended vector per tick    │ N_sched timestamped waypoints│
  │ Arm action space       │ normalised delta (÷ 0.10)    │ absolute joint angles [rad]  │
  │ Reference joint state  │ qpos at env.step() call      │ joint_pos at obs time        │
  │ Chunk selection        │ exp-weighted ensemble of     │ first execution_steps items  │
  │                        │ overlapping past chunks      │ (no blending, raw waypoints) │
  │ Stale action handling  │ blended in (older=lower wt)  │ is_new filter in real robot  │
  │                        │                              │ (skipped in this sim test)   │
  │ Gripper encoding       │ +1.0=open, −1.0=closed       │ 0.0=open, 1.0=closed         │
  │ Gripper threshold      │ vel > 0.5 → closed (−1)      │ action > 0.5 → closed (1.0)  │
  │ Arm command rate       │ 1× per sim step              │ 200 Hz NUC HighFreqCtrl      │
  │ Latency compensation   │ none                         │ robot_action_latency 0.20 s  │
  └────────────────────────┴──────────────────────────────┴──────────────────────────────┘
""")

    # ── Joint target: first step ──────────────────────────────────────────────
    print("  ── Are the arm targets numerically consistent? ───────────────────────")
    sim_target_rad  = qpos[:7] + sim_blended[:7] * MANISKILL_JOINT_DELTA
    real_target_rad = r["arm_positions"][0]
    diff = real_target_rad - sim_target_rad
    print(f"    SIM  target step 0 (rad): {np.round(sim_target_rad,  4).tolist()}")
    print(f"    REAL target step 0 (rad): {np.round(real_target_rad, 4).tolist()}")
    print(f"    Δ real − sim            : {np.round(diff, 6).tolist()}")
    if np.abs(diff).max() < 1e-5:
        print("    ✓ Targets match to numerical precision.")
        print("      Both pipelines integrate the same formula:")
        print("        pos[k] = joint_pos + cumsum(clip(vel) * 0.10 rad)")
        print("      SIM re-normalises the result (÷0.10) for ManiSkill, then")
        print("      un-normalises it here (×0.10) — the round-trip is exact.")
    else:
        print(f"    ✗ Targets differ by up to {np.abs(diff).max():.6f} rad.")
        print("      Unexpected — check whether qpos[:7] == joint_position.")

    print()

    # ── Gripper convention ────────────────────────────────────────────────────
    print("  ── Gripper convention ────────────────────────────────────────────────")
    raw_grip_0 = float(raw_actions[0, 7])
    sim_grip   = float(sim_blended[7])
    real_grip  = float(r["gripper_vals"][0])

    print(f"    pi0 raw output step 0 : {raw_grip_0:+.4f}")
    print(f"    SIM  result           : {sim_grip:+.4f}   (+1.0=open, −1.0=closed)")
    print(f"    REAL result           : {real_grip:.1f}      ( 0.0=open,  1.0=closed)")
    sim_closed  = sim_grip  < 0
    real_closed = real_grip > 0.5
    if sim_closed == real_closed:
        state = "closed" if sim_closed else "open"
        print(f"    ✓ Both agree: gripper intent = {state}.")
    else:
        print("    ✗ SIM and REAL DISAGREE on gripper state!")

    print()
    print("    Both pipelines binarize at the same threshold (> 0.5 → close),")
    print("    but map to different numeric conventions:")
    print()
    print("      pi0 raw   │ sim  binarize_sim_gripper   │ real binarize_and_clip")
    print("      ──────────┼─────────────────────────────┼────────────────────────")
    print("      > 0.5     │  −1.0  closed  (ManiSkill)  │  1.0  closed  (DROID)  ")
    print("      ≤ 0.5     │  +1.0  open    (ManiSkill)  │  0.0  open    (DROID)  ")
    print()
    print("    ManiSkill's joint-delta controller treats +1 as open and −1 as closed.")
    print("    DROID's gripper position interface uses 0 (open) and 1 (closed).")

    print()

    # ── Chunk shape ───────────────────────────────────────────────────────────
    print("  ── How many actions are consumed per inference? ──────────────────────")
    print(f"    Chunk horizon : {r['H']} steps")
    print(f"    SIM uses      : 1 step  (blended scalar, rest discarded until next infer)")
    print(f"    REAL uses     : {r['n_scheduled']} steps  (waypoints queued in NUC controller)")
    print()
    print("    SIM re-infers every step  → 1 pi0 call per env.step().")
    print("    REAL re-infers every execution_steps ticks (default 6) → the NUC")
    print("    runs waypoints at 200 Hz between calls; main loop at 10 Hz.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mock", action="store_true",
                   help="Use synthetic obs + actions (no subprocesses required).")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--robofac_python", default=DEFAULT_ROBOFAC_PY)
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    p.add_argument("--action_horizon", default=DEFAULT_ACTION_HORIZON, type=int)
    p.add_argument("--action_scale",   default=DEFAULT_ACTION_SCALE,   type=float)
    p.add_argument("--execution_steps", default=DEFAULT_EXECUTION_STEPS, type=int)
    return p


def main() -> None:
    args = build_parser().parse_args()

    print("=" * 70)
    print("  pi0 pipeline comparison: SIM vs REAL action processing")
    print("=" * 70)
    print(f"  mode          : {'MOCK (synthetic data)' if args.mock else 'LIVE (real env + model)'}")
    print(f"  instruction   : {args.instruction}")
    print(f"  action_horizon: {args.action_horizon}   action_scale: {args.action_scale}")
    print(f"  execution_steps (real): {args.execution_steps}")

    # ── 1. Observation ────────────────────────────────────────────────────────
    if args.mock:
        env_obs = make_mock_obs()
    else:
        env_obs = get_obs_from_env(args.robofac_python)

    qpos = np.asarray(env_obs["qpos"], dtype=np.float32)
    joint_position = qpos[:7].astype(np.float64)
    print_raw_obs(qpos, mock=args.mock)

    # ── 2. pi0 inference ──────────────────────────────────────────────────────
    if args.mock:
        raw_actions = make_mock_pi0_actions(horizon=args.action_horizon)
    else:
        raw_full    = run_pi0_inference(args.checkpoint, env_obs, args.instruction)
        raw_actions = raw_full[:args.action_horizon]

    if raw_actions.ndim != 2 or raw_actions.shape[1] < 8:
        print(f"[ERROR] Unexpected pi0 action shape: {raw_actions.shape}", file=sys.stderr)
        sys.exit(1)

    print_raw_actions(raw_actions, mock=args.mock)

    # ── 3. SIM pipeline ───────────────────────────────────────────────────────
    sim_chunk, sim_blended = run_sim_pipeline(
        qpos, raw_actions,
        action_horizon=args.action_horizon,
        action_scale=args.action_scale,
    )
    print_sim_results(qpos, sim_chunk, sim_blended)

    # ── 4. REAL pipeline ──────────────────────────────────────────────────────
    real_r = run_real_pipeline(
        joint_position, raw_actions,
        action_scale=args.action_scale,
        execution_steps=args.execution_steps,
    )
    print_real_results(real_r, joint_position)

    # ── 5. Analysis ───────────────────────────────────────────────────────────
    print_analysis(qpos, raw_actions, sim_blended, real_r)

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
