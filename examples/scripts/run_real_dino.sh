#!/bin/bash
proj_name=DSRL_pi0_FrankaDroid
device_id=0

export EXP=/home/robot/yingxi/dsrl_pi0/logs/$proj_name;
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# DROID-aligned runtime topology:
#   NUC                -> Franka + Polymetis + DROID ZeroRPC robot server
#   laptop/workstation -> DROID client + cameras + DSRL steering loop
#   GPU server         -> OpenPI websocket policy server

# Select which camera images are sent to the OpenPI policy request.
# Wrist-only is the default for the 2440-D Wrist-DINO RL state.
USE_WRIST_CAMERA=1
USE_EXTERIOR_CAMERA=0

# Fill in Franka DROID camera IDs on the laptop/workstation.
# LEFT_CAMERA_ID/RIGHT_CAMERA_ID are only required when USE_EXTERIOR_CAMERA=1.
# RealSense IDs can be either the raw serial or realsense_<serial>.
LEFT_CAMERA_ID=""
RIGHT_CAMERA_ID=""
WRIST_CAMERA_ID="17396664"

# Fill in OpenPI policy server host and port.
POLICY_HOST="127.0.0.1"
POLICY_PORT="8000"

# Action execution parameters (aligned with evaluate_pi0_real.py).
# action_scale: speed of arm motion. 0.5 = half of DROID training speed (safe default).
# max_joint_speed_rad_s: NUC-side safety cap. 0.1 is very conservative; increase to
#   1.5 for normal training speed. Match action_scale: e.g. action_scale=0.5 with
#   max_joint_speed_rad_s=1.5 allows up to 1.0 rad/step at 10 Hz without capping.
ACTION_SCALE="0.5"
MAX_JOINT_SPEED="0.5"


python3 examples/launch_train_real_dino.py \
--resume_from /home/robot/yingxi/dsrl_pi0/logs/DSRL_pi0_FrankaDroid/dsrl_pi0_real_dino_2026_06_09_17_06_08_0000--s-0 \
--algorithm state_sac \
--env franka_droid \
--prefix dsrl_pi0_real_dino \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.99 \
--seed 0 \
--max_steps 500000 \
--eval_interval 2000 \
--log_interval 100 \
--multi_grad_step 30 \
--action_magnitude 2.0 \
--instruction 'pick up the blue peg' \
--query_freq 8 \
--rl_noise_horizon 8 \
--hidden_dims 1024 \
--network_type transformer \
--transformer_dim 256 \
--transformer_depth 3 \
--transformer_heads 4 \
--transformer_mlp_dim 1024 \
--transformer_dropout 0.0 \
--num_qs 2 \
--dino_model facebook/dinov2-small \
--dino_device auto \
--external_camera right \
--use_wrist_camera "${USE_WRIST_CAMERA}" \
--use_exterior_camera "${USE_EXTERIOR_CAMERA}" \
--left_camera_id "${LEFT_CAMERA_ID}" \
--right_camera_id "${RIGHT_CAMERA_ID}" \
--wrist_camera_id "${WRIST_CAMERA_ID}" \
--policy_host "${POLICY_HOST}" \
--policy_port "${POLICY_PORT}" \
--action_scale "${ACTION_SCALE}" \
--max_joint_speed_rad_s "${MAX_JOINT_SPEED}"
