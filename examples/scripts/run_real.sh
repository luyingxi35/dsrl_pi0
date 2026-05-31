#!/bin/bash
proj_name=DSRL_pi0_FrankaDroid
device_id=0

export EXP=./logs/$proj_name; 
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# DROID-aligned runtime topology:
#   NUC                -> Franka + Polymetis + DROID ZeroRPC robot server
#   laptop/workstation -> DROID client + cameras + DSRL steering loop
#   GPU server         -> OpenPI websocket policy server

# Fill in Franka DROID camera IDs on the laptop/workstation.
LEFT_CAMERA_ID=""
RIGHT_CAMERA_ID=""
WRIST_CAMERA_ID=""

# Fill in OpenPI policy server host and port.
POLICY_HOST=""
POLICY_PORT="8000"


python3 examples/launch_train_real.py \
--algorithm pixel_sac \
--env franka_droid \
--prefix dsrl_pi0_real \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.99 \
--seed 0 \
--max_steps 500000  \
--eval_interval 2000 \
--log_interval 100 \
--multi_grad_step 30 \
--resize_image 128 \
--action_magnitude 2.5 \
--query_freq 10 \
--hidden_dims 1024 \
--num_qs 2 \
--external_camera right \
--left_camera_id "${LEFT_CAMERA_ID}" \
--right_camera_id "${RIGHT_CAMERA_ID}" \
--wrist_camera_id "${WRIST_CAMERA_ID}" \
--policy_host "${POLICY_HOST}" \
--policy_port "${POLICY_PORT}"
