#!/bin/bash
# Launch DSRL training on PegInsertionVertical in LIBERO (robosuite/MuJoCo).
# Architecture: PixelSAC + pi0_libero (pi05_base ckpt), identical to run_libero.sh
# except env='libero_peg' and our custom BDDL + problem class.
#
# Usage:
#   bash examples/scripts/run_sim_peg_libero.sh

proj_name=DSRL_pi0_PegLibero
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

pip install mujoco==3.3.1 2>/dev/null

python3 examples/launch_train_sim_peg.py \
  --algorithm pixel_sac \
  --env libero_peg \
  --prefix dsrl_pi0_peg_libero \
  --wandb_project ${proj_name} \
  --batch_size 256 \
  --discount 0.999 \
  --seed 0 \
  --max_steps 500000 \
  --eval_interval 10000 \
  --log_interval 500 \
  --eval_episodes 10 \
  --multi_grad_step 20 \
  --start_online_updates 500 \
  --resize_image 64 \
  --action_magnitude 1.0 \
  --query_freq 20 \
  --hidden_dims 128 \
  --target_entropy 0.0
