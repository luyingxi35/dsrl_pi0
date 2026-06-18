#!/bin/bash
# Evaluate DSRL StateSAC + pi0_droid in ManiSkill simulation.
#
# Usage:
#   bash examples/scripts/eval_sim_dino.sh --restore_path ./logs/DSRL_pi0_SimDino/<run_dir>
#   bash examples/scripts/eval_sim_dino.sh --restore_path <run_dir> --eval_episodes 3

set -euo pipefail

cd "$(dirname "$0")/../.."
export HF_ENDPOINT=https://hf-mirror.com

DSRL_PYTHON=/opt/yingxi/envs/dsrl_pi0/bin/python
ROBOFAC_PYTHON=/opt/yingxi/envs/robofac/bin/python3
PI0_DROID_CKPT=/opt/yingxi/pi0_droid
DEVICE_ID=0

RESTORE_PATH=""
EVAL_EPISODES=10
MAX_ROLLOUT_STEPS=600
QUERY_FREQ=8
RL_NOISE_HORIZON=8
ACTION_SCALE=0.5
OUTPUTDIR=./logs/dino_eval_sim
INSTRUCTION="pick up the peg and insert it vertically"
DINO_MODEL=facebook/dinov2-small
DINO_DEVICE=auto

while [[ $# -gt 0 ]]; do
    case "$1" in
        --restore_path) RESTORE_PATH="$2"; shift 2 ;;
        --eval_episodes) EVAL_EPISODES="$2"; shift 2 ;;
        --max_rollout_steps) MAX_ROLLOUT_STEPS="$2"; shift 2 ;;
        --query_freq) QUERY_FREQ="$2"; shift 2 ;;
        --rl_noise_horizon) RL_NOISE_HORIZON="$2"; shift 2 ;;
        --action_scale) ACTION_SCALE="$2"; shift 2 ;;
        --outputdir) OUTPUTDIR="$2"; shift 2 ;;
        --instruction) INSTRUCTION="$2"; shift 2 ;;
        --checkpoint_path) PI0_DROID_CKPT="$2"; shift 2 ;;
        --robofac_python) ROBOFAC_PYTHON="$2"; shift 2 ;;
        --dsrl_python) DSRL_PYTHON="$2"; shift 2 ;;
        --device_id) DEVICE_ID="$2"; shift 2 ;;
        --dino_model) DINO_MODEL="$2"; shift 2 ;;
        --dino_device) DINO_DEVICE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${RESTORE_PATH}" ]]; then
    echo "Error: --restore_path is required." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${DEVICE_ID}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_triton_gemm_any=True"
export TF_CPP_MIN_LOG_LEVEL=1
DSRL_SITE_PACKAGES="$("${DSRL_PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
ALL_NVIDIA=$(find "${DSRL_SITE_PACKAGES}/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${ALL_NVIDIA}:${LD_LIBRARY_PATH:-}"

"${DSRL_PYTHON}" examples/evaluate_policy_sim_dino.py \
    --restore_path "${RESTORE_PATH}" \
    --checkpoint_path "${PI0_DROID_CKPT}" \
    --eval_episodes "${EVAL_EPISODES}" \
    --max_rollout_steps "${MAX_ROLLOUT_STEPS}" \
    --query_freq "${QUERY_FREQ}" \
    --rl_noise_horizon "${RL_NOISE_HORIZON}" \
    --action_scale "${ACTION_SCALE}" \
    --robofac_python "${ROBOFAC_PYTHON}" \
    --instruction "${INSTRUCTION}" \
    --dino_model "${DINO_MODEL}" \
    --dino_device "${DINO_DEVICE}" \
    --outputdir "${OUTPUTDIR}"
