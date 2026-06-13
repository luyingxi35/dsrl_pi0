cd ~/yingxi/dsrl_pi0
export HF_ENDPOINT=https://hf-mirror.com
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python3 examples/evaluate_pi0_real.py \
--instruction "pick up the blue peg" \
--eval_episodes 10 \
--max_duration_s 60 \
--execution_steps 4 \
--action_scale 1.0 \
--max_joint_speed_rad_s 0.2 \
--control_frequency_hz 10 \
--use_wrist_camera 1 \
--use_exterior_camera 0 \
--policy_host 127.0.0.1 \
--policy_port 8000 \
--outputdir ./logs/pi0_eval_real