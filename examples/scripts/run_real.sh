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
# If you only have one side/external camera, set LEFT_CAMERA_ID and
# RIGHT_CAMERA_ID to the same side camera serial. RealSense IDs can be
# either the raw serial or realsense_<serial>.
LEFT_CAMERA_ID="241122302552"
RIGHT_CAMERA_ID="241122302552"
WRIST_CAMERA_ID="17396664"

# Fill in OpenPI policy server host and port.
POLICY_HOST="127.0.0.1"
POLICY_PORT="8000"

# Action execution parameters (aligned with evaluate_pi0_real.py).
# action_scale: speed of arm motion. 0.5 = half of DROID training speed (safe default).
# max_joint_speed_rad_s: NUC-side safety cap. 0.1 is very conservative; increase to
#   1.5 for normal training speed. Match action_scale: e.g. action_scale=0.5 with
#   max_joint_speed_rad_s=1.5 allows up to 1.0 rad/step at 10 Hz without capping.
ACTION_SCALE="1.0"
MAX_JOINT_SPEED="0.3"


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
--multi_grad_step 20 \
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
--policy_port "${POLICY_PORT}" \
--action_scale "${ACTION_SCALE}" \
--max_joint_speed_rad_s "${MAX_JOINT_SPEED}"
