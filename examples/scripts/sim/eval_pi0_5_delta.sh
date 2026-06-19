#!/bin/bash
# Evaluate pi0.5_droid using pd_joint_delta_pos control mode.
#
# Compared to eval_pi0_5.sh (which integrates velocity to absolute joint
# positions via pd_joint_pos), this script sends the pi0 velocity output
# directly to ManiSkill as per-step joint-position deltas:
#
#   pi0 vel ∈ [-1,1]
#     → normalised_delta = clip(vel × action_scale × 2.0, -1, 1)
#     → ManiSkill pd_joint_delta_pos (±1 maps to ±0.1 rad/step)
#
# With action_scale=0.5 (default): normalised_delta = pi0_vel (identity).
# No accumulation across the chunk — each step is independent.
#
# Usage:
#   bash examples/scripts/sim/eval_pi0_5_delta.sh
#   bash examples/scripts/sim/eval_pi0_5_delta.sh --eval_episodes 3

set -euo pipefail

cd "$(dirname "$0")/../../.."   # → repo root

DSRL_PYTHON=/opt/yingxi/envs/dsrl_pi0/bin/python
ROBOFAC_PYTHON=/opt/yingxi/envs/robofac/bin/python3
PI05_DROID_CKPT=/opt/yingxi/pi05_droid
CONFIG_NAME=pi05_droid
DEVICE_ID=0

EVAL_EPISODES=10
MAX_ROLLOUT_STEPS=600
ACTION_SCALE=1.0          # 0.5 → identity mapping in delta mode
OUTPUTDIR=./logs/pi05_delta_eval_sim
INSTRUCTION="pick up the blue peg and insert it into the hole"
WORKSPACE_BOUNDS_PATH="/home/gpu4/yingxi/dsrl_pi0/workspace_bounds.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --eval_episodes)         EVAL_EPISODES="$2";          shift 2 ;;
        --max_rollout_steps)     MAX_ROLLOUT_STEPS="$2";      shift 2 ;;
        --action_scale)          ACTION_SCALE="$2";            shift 2 ;;
        --outputdir)             OUTPUTDIR="$2";               shift 2 ;;
        --instruction)           INSTRUCTION="$2";             shift 2 ;;
        --checkpoint_path)       PI05_DROID_CKPT="$2";        shift 2 ;;
        --robofac_python)        ROBOFAC_PYTHON="$2";          shift 2 ;;
        --dsrl_python)           DSRL_PYTHON="$2";             shift 2 ;;
        --device_id)             DEVICE_ID="$2";               shift 2 ;;
        --workspace_bounds_path) WORKSPACE_BOUNDS_PATH="$2";  shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEVICE_ID}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_triton_gemm_any=True"
export TF_CPP_MIN_LOG_LEVEL=1

DSRL_SITE_PACKAGES="$("${DSRL_PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
ALL_NVIDIA=$(find "${DSRL_SITE_PACKAGES}/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${ALL_NVIDIA}:${LD_LIBRARY_PATH:-}"

"${DSRL_PYTHON}" examples/sim/evaluate_pi0_delta.py \
    --checkpoint_path "${PI05_DROID_CKPT}" \
    --eval_episodes "${EVAL_EPISODES}" \
    --max_rollout_steps "${MAX_ROLLOUT_STEPS}" \
    --action_scale "${ACTION_SCALE}" \
    --robofac_python "${ROBOFAC_PYTHON}" \
    --instruction "${INSTRUCTION}" \
    --outputdir "${OUTPUTDIR}" \
    --config_name "${CONFIG_NAME}" \
    ${WORKSPACE_BOUNDS_PATH:+--workspace_bounds_path "${WORKSPACE_BOUNDS_PATH}"}
