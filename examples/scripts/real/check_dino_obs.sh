#!/bin/bash
proj_name=DSRL_pi0_FrankaDroid
device_id=0

export EXP=./logs/$proj_name;
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

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

PYTHON_BIN="${PYTHON_BIN:-/home/robot/miniconda3/envs/dsrl_pi0/bin/python}"

"${PYTHON_BIN}" tests/check_real_wrist_dino_obs.py \
--policy_host "${POLICY_HOST}" \
--policy_port "${POLICY_PORT}" \
--external_camera right \
--left_camera_id "${LEFT_CAMERA_ID}" \
--right_camera_id "${RIGHT_CAMERA_ID}" \
--wrist_camera_id "${WRIST_CAMERA_ID}" \
--dino_model facebook/dinov2-small \
--dino_device auto
