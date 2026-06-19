#!/bin/bash
# Multi-seed DSRL simulation sweep — parallel (one seed per GPU).
#
# Architecture (aligned with real-robot training):
#   - ManiSkill PegInsertionVertical-v1 with panda_wristcam (robofac subprocess)
#   - pi0_droid policy (local, /opt/yingxi/pi0_droid)
#   - StateSACLearner + Transformer (same as run_real_dino.sh)
#   - STATE_DIM = 2440: proprio(8) + VLM_embed(2048) + DINOv2_wrist(384)
#   - DINOv2 on WRIST camera (hand_camera), matching real-robot setup
#
# Usage:
#   bash examples/scripts/sim/run_dino.sh
#   bash examples/scripts/sim/run_dino.sh --seeds "0 1 2 3"
#   bash examples/scripts/sim/run_dino_dense.sh --seeds "0 1 2" --gpus "4 5 6"  # 1:1
#   bash examples/scripts/sim/run_dino_dense.sh --seeds "0 1 2" --gpus "7"      # all on GPU 7
#   bash examples/scripts/sim/run_dino_dense.sh --seeds "0 1 2 3" --gpus "6 7"  # round-robin
#
# Seeds run in parallel. GPU assignment is round-robin:
#   seed[i] → gpus[i % len(gpus)] so fewer GPUs than seeds is fine.
# Default GPU assignment: seed index maps to GPU index (seed 0 → GPU 0, etc.).

set -euo pipefail

SEEDS="0 1 2"
GPUS=""          # empty = auto-assign by seed index

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds) SEEDS="$2"; shift 2 ;;
        --gpus)  GPUS="$2";  shift 2 ;;
        *)       echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

read -ra SEEDS_ARR <<< "$SEEDS"
if [[ -z "$GPUS" ]]; then
    # default: GPU i for seed i
    read -ra GPUS_ARR <<< "$(seq 0 $(( ${#SEEDS_ARR[@]} - 1 )))"
else
    read -ra GPUS_ARR <<< "$GPUS"
fi


# ── Paths ──────────────────────────────────────────────────────────────────────
ROBOFAC_PYTHON=/opt/yingxi/envs/robofac/bin/python3
PI0_DROID_CKPT=/opt/yingxi/pi0_droid

# ── Robometer dense reward config ─────────────────────────────────────────────
ROBOMETER_PYTHON=/opt/yingxi/envs/robometer/bin/python3
ROBOMETER_CKPT=/opt/yingxi/checkpoint-400
ROBOMETER_BASE_MODEL=/opt/caoyuhang/Pretrained_models/Qwen3-VL-4B-Instruct
PROGRESS_REWARD_SCALE=1.0

# ── Shared environment ─────────────────────────────────────────────────────────
proj_name=DSRL_pi0_SimDinoDense

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EXP=./logs/$proj_name
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# HuggingFace: use cached weights offline; if re-download needed set TRANSFORMERS_OFFLINE=0
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

ALL_NVIDIA=$(find /home/gpu4/miniconda3/envs/dsrl_pi0/lib/python3.11/site-packages/nvidia \
    -name 'lib' -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${ALL_NVIDIA}:${LD_LIBRARY_PATH}"

mkdir -p "$EXP"

# ── Workspace bounds ─────────────────────────────────────────────────────────
WORKSPACE_BOUNDS_PATH="/home/gpu4/yingxi/dsrl_pi0/workspace_bounds.json"

echo "=== SimDino Sweep (parallel) ==="
echo "  Seeds: ${SEEDS_ARR[*]}"
echo "  GPUs:  ${GPUS_ARR[*]}"
echo "  pi0:   $PI0_DROID_CKPT"
echo "  out:   $EXP"
echo ""

# ── Launch all seeds in parallel, one per GPU ──────────────────────────────────
PIDS=()

cleanup() {
    local code=$?
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        echo ""
        echo "Stopping launched SimDino jobs: ${PIDS[*]}" >&2
        for pid in "${PIDS[@]}"; do
            kill -INT "$pid" 2>/dev/null || true
        done
        sleep 2
        for pid in "${PIDS[@]}"; do
            kill -TERM "$pid" 2>/dev/null || true
        done
        sleep 2
        for pid in "${PIDS[@]}"; do
            kill -KILL "$pid" 2>/dev/null || true
        done
    fi
    exit "$code"
}
trap cleanup INT TERM

for idx in "${!SEEDS_ARR[@]}"; do
    SEED=${SEEDS_ARR[$idx]}
    GPU=${GPUS_ARR[$(( idx % ${#GPUS_ARR[@]} ))]}
    LOG="$EXP/seed${SEED}_gpu${GPU}.log"

    echo "  Seed $SEED → GPU $GPU  (log: $LOG)"

    # CUDA_VISIBLE_DEVICES restricts this process to one GPU;
    # from the process's view that GPU appears as device 0,
    # so MUJOCO_EGL_DEVICE_ID is always 0.
    CUDA_VISIBLE_DEVICES=$GPU \
    MUJOCO_EGL_DEVICE_ID=0 \
        python3 -m examples.sim.launch_train_dino_dense \
            --algorithm state_sac \
            --env peg_insertion_vertical_v2 \
            --prefix "dsrl_pi0_sim_dino_s${SEED}" \
            --wandb_project ${proj_name} \
            --batch_size 256 \
            --discount 0.99 \
            --seed "$SEED" \
            --max_steps 500000 \
            --max_timesteps 600 \
            --eval_episodes 10 \
            --eval_interval 999999 \
            --log_interval 100 \
            --multi_grad_step 5 \
            --num_initial_traj_collect 5 \
            --action_magnitude 2.0 \
            --action_scale 1.0 \
            --instruction 'pick up the peg and insert it into the hole' \
            --query_freq 8 \
            --rl_noise_horizon 8 \
            --network_type transformer \
            --transformer_dim 256 \
            --transformer_depth 3 \
            --transformer_heads 4 \
            --transformer_mlp_dim 1024 \
            --transformer_dropout 0.0 \
            --num_qs 2 \
            --checkpoint_path "${PI0_DROID_CKPT}" \
            --robofac_python "${ROBOFAC_PYTHON}" \
            --dino_model facebook/dinov2-small \
            --dino_device auto \
            --eval_env_step_interval 1000 \
            --stop_success_rate 0.95 \
            --stop_window 2 \
            ${WORKSPACE_BOUNDS_PATH:+--workspace_bounds_path "${WORKSPACE_BOUNDS_PATH}"} \
            --robometer_python       "${ROBOMETER_PYTHON}" \
            --robometer_checkpoint_path "${ROBOMETER_CKPT}" \
            --robometer_base_model_id   "${ROBOMETER_BASE_MODEL}" \
            --progress_reward_scale     "${PROGRESS_REWARD_SCALE}" \
        > "$LOG" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} seeds launched in background."
echo "Monitor:  tail -f $EXP/seed*.log"
echo ""

# ── Wait for all seeds; collect failures without early-exit ───────────────────
set +e
FAILED=()
for idx in "${!PIDS[@]}"; do
    PID=${PIDS[$idx]}
    SEED=${SEEDS_ARR[$idx]}
    wait "$PID"
    CODE=$?
    if [[ $CODE -eq 0 ]]; then
        echo "  Seed $SEED [PID $PID]: OK"
    else
        echo "  Seed $SEED [PID $PID]: FAILED (exit $CODE)" >&2
        FAILED+=("$SEED")
    fi
done
set -e

[[ ${#FAILED[@]} -gt 0 ]] && \
    echo "" && echo "WARNING: seeds ${FAILED[*]} failed — check $EXP/seed*.log" >&2

# ── Aggregate plot ─────────────────────────────────────────────────────────────
echo ""
echo "All seeds done. Plotting..."
python3 examples/sim/plot_curve.py \
    --log_dir  "$EXP" \
    --output   "$EXP/sim_dino_curve.png" \
    --stop_line 0.95 \
    --title    "PegInsertionVertical — DSRL Dense Reward (wrist-aligned, $(echo $SEEDS | wc -w) seeds)"

echo ""
echo "=== Sweep complete ==="
echo "  Curve: $EXP/sim_dino_curve.png"
echo "  CSVs:  $EXP/*/eval_curve.csv"
echo "  Logs:  $EXP/seed*.log"
