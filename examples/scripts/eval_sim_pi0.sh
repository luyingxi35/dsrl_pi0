#!/bin/bash
# Evaluate pi0_droid only in ManiSkill simulation.
#
# Usage:
#   bash examples/scripts/eval_sim_pi0.sh
#   bash examples/scripts/eval_sim_pi0.sh --eval_episodes 3 --max_rollout_steps 100

set -euo pipefail

cd "$(dirname "$0")/../.."

DSRL_PYTHON=/opt/yingxi/envs/dsrl_pi0/bin/python
ROBOFAC_PYTHON=/opt/yingxi/envs/robofac/bin/python3
PI0_DROID_CKPT=/opt/yingxi/pi0_droid
DEVICE_ID=0

EVAL_EPISODES=10
MAX_ROLLOUT_STEPS=600
QUERY_FREQ=8
ACTION_SCALE=0.5
OUTPUTDIR=./logs/pi0_eval_sim
INSTRUCTION="pick up the peg and insert it vertically"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --eval_episodes) EVAL_EPISODES="$2"; shift 2 ;;
        --max_rollout_steps) MAX_ROLLOUT_STEPS="$2"; shift 2 ;;
        --query_freq) QUERY_FREQ="$2"; shift 2 ;;
        --action_scale) ACTION_SCALE="$2"; shift 2 ;;
        --outputdir) OUTPUTDIR="$2"; shift 2 ;;
        --instruction) INSTRUCTION="$2"; shift 2 ;;
        --checkpoint_path) PI0_DROID_CKPT="$2"; shift 2 ;;
        --robofac_python) ROBOFAC_PYTHON="$2"; shift 2 ;;
        --dsrl_python) DSRL_PYTHON="$2"; shift 2 ;;
        --device_id) DEVICE_ID="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${DEVICE_ID}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_triton_gemm_any=True"
export TF_CPP_MIN_LOG_LEVEL=1
DSRL_SITE_PACKAGES="$("${DSRL_PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
ALL_NVIDIA=$(find "${DSRL_SITE_PACKAGES}/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${ALL_NVIDIA}:${LD_LIBRARY_PATH:-}"

"${DSRL_PYTHON}" examples/evaluate_pi0_sim.py \
    --checkpoint_path "${PI0_DROID_CKPT}" \
    --eval_episodes "${EVAL_EPISODES}" \
    --max_rollout_steps "${MAX_ROLLOUT_STEPS}" \
    --query_freq "${QUERY_FREQ}" \
    --action_scale "${ACTION_SCALE}" \
    --robofac_python "${ROBOFAC_PYTHON}" \
    --instruction "${INSTRUCTION}" \
    --outputdir "${OUTPUTDIR}"
