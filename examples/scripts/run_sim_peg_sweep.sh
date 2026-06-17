#!/bin/bash
# Multi-seed LIBERO PegInsertionVertical training sweep.
# Each seed logs eval_curve.csv; afterwards plots averaged env_steps vs success_rate.
#
# Usage:
#   bash examples/scripts/run_sim_peg_sweep.sh
#   bash examples/scripts/run_sim_peg_sweep.sh --seeds "0 1 2"

set -e
SEEDS="0 1 2"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds) SEEDS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

proj_name=DSRL_pi0_PegLibero_Sweep
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id
export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ALL_NVIDIA=$(find /home/gpu4/miniconda3/envs/dsrl_pi0/lib/python3.11/site-packages/nvidia \
    -name 'lib' -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${ALL_NVIDIA}:${LD_LIBRARY_PATH}"

mkdir -p "$EXP"
echo "=== PegLibero Sweep ==="
echo "  Seeds: $SEEDS"
echo "  Output: $EXP"
echo ""

for SEED in $SEEDS; do
    echo "── Seed $SEED ($(date)) ──"
    python3 examples/launch_train_sim_peg.py \
        --algorithm pixel_sac \
        --env libero_peg \
        --prefix "dsrl_pi0_peg_libero_seed${SEED}" \
        --wandb_project ${proj_name} \
        --batch_size 256 \
        --discount 0.999 \
        --seed "$SEED" \
        --max_steps 200000 \
        --eval_interval 2000 \
        --log_interval 500 \
        --eval_episodes 10 \
        --multi_grad_step 20 \
        --start_online_updates 500 \
        --resize_image 64 \
        --action_magnitude 1.0 \
        --query_freq 20 \
        --hidden_dims 128 \
        --target_entropy 0.0
    echo "  Seed $SEED done ($(date))"
    echo ""
done

echo "All seeds done. Plotting..."
python3 examples/plot_sim_dino_curve.py \
    --log_dir  "$EXP" \
    --output   "$EXP/peg_libero_curve.png" \
    --stop_line 0.8 \
    --title    "PegInsertionVertical LIBERO — DSRL (${SEEDS} seeds)"

echo ""
echo "=== Sweep complete ==="
echo "  Curve: $EXP/peg_libero_curve.png"
echo "  CSVs:  $EXP/*/eval_curve.csv"
