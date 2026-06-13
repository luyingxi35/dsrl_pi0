#!/bin/bash

cd ~/yingxi/dsrl_pi0
export HF_ENDPOINT=https://hf-mirror.com
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Select which camera images are sent to the OpenPI policy request.
# Wrist is still used by the DSRL DINO state builder.
USE_WRIST_CAMERA=1
USE_EXTERIOR_CAMERA=0

# Fill in Franka DROID camera IDs on the laptop/workstation.
# LEFT_CAMERA_ID/RIGHT_CAMERA_ID are only required when USE_EXTERIOR_CAMERA=1.
LEFT_CAMERA_ID=""
RIGHT_CAMERA_ID=""
WRIST_CAMERA_ID="17396664"
EXTERNAL_CAMERA="right"

POLICY_HOST="127.0.0.1"
POLICY_PORT="8000"

ACTION_SCALE="0.5"
MAX_JOINT_SPEED="0.3"
# DSRL eval timing modes:
#   low_watermark: Version 1, refill waypoints before the controller runs dry.
#   train:         Version 2, strict train-style fixed query_freq cadence.
DSRL_EVAL_TIMING_MODE="low_watermark"
MIN_FUTURE_ACTIONS="2"
MIN_FUTURE_HORIZON_S="0.25"

python3 examples/evaluate_policy_real.py \
--restore_path ./logs/DSRL_pi0_FrankaDroid/dsrl_pi0_real_dino_2026_06_13_15_37_49_0000--s-0 \
--instruction "pick up the blue peg" \
--eval_episodes 10 \
--max_duration_s 60.0 \
--max_rollout_steps 600 \
--control_frequency_hz 10 \
--query_freq 8 \
--dsrl_eval_timing_mode "${DSRL_EVAL_TIMING_MODE}" \
--min_future_actions "${MIN_FUTURE_ACTIONS}" \
--min_future_horizon_s "${MIN_FUTURE_HORIZON_S}" \
--external_camera "${EXTERNAL_CAMERA}" \
--use_wrist_camera "${USE_WRIST_CAMERA}" \
--use_exterior_camera "${USE_EXTERIOR_CAMERA}" \
--left_camera_id "${LEFT_CAMERA_ID}" \
--right_camera_id "${RIGHT_CAMERA_ID}" \
--wrist_camera_id "${WRIST_CAMERA_ID}" \
--policy_host "${POLICY_HOST}" \
--policy_port "${POLICY_PORT}" \
--outputdir ./logs/policy_eval_real \
--seed 0 \
--hidden_dims 1024 \
--network_type transformer \
--rl_noise_horizon 8 \
--action_scale "${ACTION_SCALE}" \
--max_joint_speed_rad_s "${MAX_JOINT_SPEED}" \
--dsrl_eval_timing_mode train \
--diagnostic_dir ./logs/diagnostics
