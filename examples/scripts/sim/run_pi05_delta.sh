#!/bin/bash
# Delta-action variant: uses pd_joint_delta_pos instead of pd_joint_pos.
# Multi-seed DSRL simulation sweep for pi0.5 — parallel (one seed per GPU).
#
# Architecture:
#   - ManiSkill PegInsertionVertical-v1 with panda_wristcam (robofac subprocess)
#   - pi05_droid policy (local, /opt/yingxi/pi05_droid)
#   - StateSACLearner + Transformer (same as run_dino.sh)
#   - STATE_DIM = 2440: proprio(8) + VLM_embed(2048) + DINOv2_wrist(384)
#   - rl_noise_horizon = 15  (matches pi0.5 action_horizon=15)
#
# Usage:
#   bash examples/scripts/sim/run_pi05.sh
#   bash examples/scripts/sim/run_pi05.sh --seeds "0 1 2 3"
#   bash examples/scripts/sim/run_pi05.sh --seeds "0 1 2" --gpus "4 5 6"

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
    read -ra GPUS_ARR <<< "$(seq 0 $(( ${#SEEDS_ARR[@]} - 1 )))"
else
    read -ra GPUS_ARR <<< "$GPUS"
fi

if [[ ${#SEEDS_ARR[@]} -ne ${#GPUS_ARR[@]} ]]; then
    echo "Error: --seeds has ${#SEEDS_ARR[@]} values but --gpus has ${#GPUS_ARR[@]}" >&2
    exit 1
fi

# ── Paths ──────────────────────────────────────────────────────────────────────
ROBOFAC_PYTHON=/opt/yingxi/envs/robofac/bin/python3
PI05_DROID_CKPT=/opt/yingxi/pi05_droid

# ── Shared environment ─────────────────────────────────────────────────────────
proj_name=DSRL_pi05_SimDino_Delta

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EXP=./logs/$proj_name
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

ALL_NVIDIA=$(find /opt/yingxi/envs/dsrl_pi0/lib/python3.11/site-packages/nvidia \
    -name 'lib' -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${ALL_NVIDIA}:${LD_LIBRARY_PATH:-}"

mkdir -p "$EXP"

WORKSPACE_BOUNDS_PATH="/home/gpu4/yingxi/dsrl_pi0/workspace_bounds.json"

echo "=== pi0.5 SimDino Sweep (sparse reward, parallel) ==="
echo "  Seeds: ${SEEDS_ARR[*]}"
echo "  GPUs:  ${GPUS_ARR[*]}"
echo "  pi05:  $PI05_DROID_CKPT"
echo "  out:   $EXP"
echo ""

PIDS=()

cleanup() {
    local code=$?
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        echo ""
        echo "Stopping launched pi05 SimDino jobs: ${PIDS[*]}" >&2
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
    GPU=${GPUS_ARR[$idx]}
    LOG="$EXP/seed${SEED}_gpu${GPU}.log"

    echo "  Seed $SEED → GPU $GPU  (log: $LOG)"

    CUDA_VISIBLE_DEVICES=$GPU \
    MUJOCO_EGL_DEVICE_ID=0 \
        python3 -m examples.sim.launch_train_dino_pi05_delta \
            --algorithm state_sac \
            --env peg_insertion_vertical_v2 \
            --prefix "dsrl_pi05_sim_dino_s${SEED}" \
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
            --action_scale 0.5  # delta mode: 0.5→identity (vel=delta) \
            --instruction 'pick up the peg and insert it into the hole' \
            --query_freq 8 \
            --rl_noise_horizon 15 \
            --network_type transformer \
            --transformer_dim 256 \
            --transformer_depth 3 \
            --transformer_heads 4 \
            --transformer_mlp_dim 1024 \
            --transformer_dropout 0.0 \
            --num_qs 2 \
            --checkpoint_path "${PI05_DROID_CKPT}" \
            --robofac_python "${ROBOFAC_PYTHON}" \
            --dino_model facebook/dinov2-small \
            --dino_device auto \
            --eval_env_step_interval 1000 \
            --stop_success_rate 0.95 \
            --stop_window 2 \
            ${WORKSPACE_BOUNDS_PATH:+--workspace_bounds_path "${WORKSPACE_BOUNDS_PATH}"} \
        > "$LOG" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} seeds launched in background."
echo "Monitor:  tail -f $EXP/seed*.log"
echo ""

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
    echo "" && echo "WARNING: seeds ${FAILED[*]} failed -- check $EXP/seed*.log" >&2

echo ""
echo "All seeds done. Plotting..."
python3 examples/sim/plot_curve.py \
    --log_dir  "$EXP" \
    --output   "$EXP/sim_dino_pi05_curve.png" \
    --stop_line 0.95 \
    --title    "PegInsertionVertical -- DSRL pi0.5 (wrist-aligned, $(echo $SEEDS | wc -w) seeds)"

echo ""
echo "=== Sweep complete ==="
echo "  Curve: $EXP/sim_dino_pi05_curve.png"
echo "  CSVs:  $EXP/*/eval_curve.csv"
echo "  Logs:  $EXP/seed*.log"
