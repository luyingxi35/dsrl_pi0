#!/bin/bash
# Multi-seed DSRL simulation sweep for PegInsertionVertical-v1.
#
# Runs SEEDS sequentially on the same GPU (single H100).
# Each seed:
#   - saves a checkpoint + evaluates every EVAL_ENV_STEP_INTERVAL env steps
#   - stops automatically when success_rate >= STOP_SUCCESS_RATE for STOP_WINDOW
#     consecutive eval rounds (or when MAX_STEPS gradient steps are reached)
#   - writes per-seed eval_curve.csv to its output directory
#
# After all seeds finish, plots the averaged success-rate curve.
#
# Usage:
#   bash examples/scripts/run_sim_dino_sweep.sh
#   bash examples/scripts/run_sim_dino_sweep.sh --seeds "0 1 2 3"   # custom seeds

set -e

# ── Parse optional --seeds argument ───────────────────────────────────────────
SEEDS="0 1 2"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds) SEEDS="$2"; shift 2 ;;
        *)       echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Experiment settings ────────────────────────────────────────────────────────
proj_name=DSRL_pi0_SimDino_Sweep
device_id=0

# Every N env steps: evaluate + save checkpoint
EVAL_ENV_STEP_INTERVAL=1000
# Stop when last STOP_WINDOW evals are all >= STOP_SUCCESS_RATE
STOP_SUCCESS_RATE=0.95
STOP_WINDOW=2

# ManiSkill2 / MuJoCo EGL headless rendering
export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "$EXP"
echo "=== SimDino Sweep ==="
echo "  Seeds:                 $SEEDS"
echo "  Eval interval:         ${EVAL_ENV_STEP_INTERVAL} env steps"
echo "  Stop threshold:        ${STOP_SUCCESS_RATE} × ${STOP_WINDOW} consecutive evals"
echo "  Output directory:      $EXP"
echo "  WandB project:         $proj_name"
echo ""

# ── Run each seed sequentially ─────────────────────────────────────────────────
for SEED in $SEEDS; do
    echo "────────────────────────────────────────────────"
    echo "  Starting seed $SEED  ($(date))"
    echo "────────────────────────────────────────────────"

    python3 -m examples.launch_train_sim_dino \
        --algorithm state_sac \
        --env peg_insertion_vertical \
        --prefix "dsrl_pi0_sim_dino_seed${SEED}" \
        --wandb_project "${proj_name}" \
        --batch_size 256 \
        --discount 0.99 \
        --seed "$SEED" \
        --max_steps 500000 \
        --max_timesteps 300 \
        --eval_episodes 10 \
        --eval_interval 999999 \
        --log_interval 100 \
        --multi_grad_step 5 \
        --num_initial_traj_collect 5 \
        --action_magnitude 2.0 \
        --instruction 'pick up the peg and insert it vertically' \
        --query_freq 8 \
        --rl_noise_horizon 8 \
        --network_type transformer \
        --transformer_dim 256 \
        --transformer_depth 3 \
        --transformer_heads 4 \
        --transformer_mlp_dim 1024 \
        --transformer_dropout 0.0 \
        --num_qs 2 \
        --dino_model facebook/dinov2-small \
        --dino_device auto \
        --checkpoint_path /opt/yingxi/pi0_droid \
        --checkpoint_interval 999999 \
        --eval_env_step_interval "$EVAL_ENV_STEP_INTERVAL" \
        --stop_success_rate "$STOP_SUCCESS_RATE" \
        --stop_window "$STOP_WINDOW"

    echo "  Seed $SEED finished  ($(date))"
    echo ""
done

# ── Plot aggregated curve ──────────────────────────────────────────────────────
echo "All seeds done. Plotting..."
python3 examples/plot_sim_dino_curve.py \
    --log_dir  "$EXP" \
    --output   "$EXP/sim_dino_curve.png" \
    --stop_line "$STOP_SUCCESS_RATE" \
    --title    "PegInsertionVertical — DSRL sim (${#SEEDS} seeds)" \
    --smoothing 3

echo ""
echo "=== Sweep complete ==="
echo "  Results: $EXP"
echo "  Curve:   $EXP/sim_dino_curve.png"
