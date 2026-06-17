#!/bin/bash
# Launch DSRL simulation training:
#   PegInsertionVertical-v1 (ManiSkill2/SAPIEN) + pi0_droid (local) + StateSAC + DINOv2
#
# Usage:
#   bash examples/scripts/run_sim_dino.sh

proj_name=DSRL_pi0_SimDino
device_id=0

# ManiSkill2 / MuJoCo rendering via EGL (headless GPU)
export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python3 examples/launch_train_sim_dino.py \
  --algorithm state_sac \
  --env peg_insertion_vertical \
  --prefix dsrl_pi0_sim_dino \
  --wandb_project ${proj_name} \
  --batch_size 256 \
  --discount 0.99 \
  --seed 0 \
  --max_steps 500000 \
  --max_timesteps 300 \
  --eval_interval 2000 \
  --log_interval 100 \
  --eval_episodes 5 \
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
  --checkpoint_path /opt/yingxi/pi0_droid
