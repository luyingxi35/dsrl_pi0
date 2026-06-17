#!/bin/bash
# Multi-seed DSRL simulation sweep v2.
#
# Architecture (completely aligned with real-robot training):
#   - ManiSkill PegInsertionVertical-v1 with panda_wristcam (robofac subprocess)
#   - pi0_droid policy (local, /opt/yingxi/pi0_droid)
#   - StateSACLearner + Transformer (same as run_real_dino.sh)
#   - STATE_DIM = 2440: proprio(8) + VLM_embed(2048) + DINOv2_wrist(384)
#   - DINOv2 on WRIST camera (hand_camera), matching real-robot setup
#
# Usage:
#   bash examples/scripts/run_sim_dino_v2.sh
#   bash examples/scripts/run_sim_dino_v2.sh --seeds "0 1 2 3"

set -e

SEEDS="0 1 2"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds) SEEDS="$2"; shift 2 ;;
        *)       echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Paths ──────────────────────────────────────────────────────────────────────
ROBOFAC_PYTHON=/opt/yingxi/envs/robofac/bin/python3
PI0_DROID_CKPT=/opt/yingxi/pi0_droid

# ── Environment ────────────────────────────────────────────────────────────────
proj_name=DSRL_pi0_SimDinoV2
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ALL_NVIDIA=$(find /home/gpu4/miniconda3/envs/dsrl_pi0/lib/python3.11/site-packages/nvidia \
    -name 'lib' -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${ALL_NVIDIA}:${LD_LIBRARY_PATH}"

mkdir -p "$EXP"

echo "=== SimDino V2 Sweep ==="
echo "  Seeds:         $SEEDS"
echo "  robofac env:   $ROBOFAC_PYTHON"
echo "  pi0 ckpt:      $PI0_DROID_CKPT"
echo "  output dir:    $EXP"
echo ""

# ── Per-seed training ──────────────────────────────────────────────────────────
for SEED in $SEEDS; do
    echo "── Seed $SEED ($(date)) ──────────────────"
    python3 -m examples.launch_train_sim_dino_v2 \
        --algorithm state_sac \
        --env peg_insertion_vertical_v2 \
        --prefix "dsrl_pi0_sim_dino_v2_s${SEED}" \
        --wandb_project ${proj_name} \
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
        --checkpoint_path "${PI0_DROID_CKPT}" \
        --robofac_python "${ROBOFAC_PYTHON}" \
        --dino_model facebook/dinov2-small \
        --dino_device auto \
        --eval_env_step_interval 1000 \
        --stop_success_rate 0.95 \
        --stop_window 2
    echo "  Seed $SEED done ($(date))"
    echo ""
done

# ── Aggregate plot ─────────────────────────────────────────────────────────────
echo "All seeds done. Plotting..."
python3 examples/plot_sim_dino_curve.py \
    --log_dir  "$EXP" \
    --output   "$EXP/sim_dino_v2_curve.png" \
    --stop_line 0.95 \
    --title    "PegInsertionVertical v2 — DSRL (wrist-aligned, $(echo $SEEDS | wc -w) seeds)"

echo ""
echo "=== Sweep complete ==="
echo "  Curve: $EXP/sim_dino_v2_curve.png"
echo "  CSVs:  $EXP/*/eval_curve.csv"
