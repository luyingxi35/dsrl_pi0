cd ~/yingxi/dsrl_pi0
export HF_ENDPOINT=https://hf-mirror.com
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python3 examples/evaluate_policy_real.py \
--restore_path ./logs/DSRL_pi0_FrankaDroid/dsrl_pi0_real_dino_2026_06_09_21_46_52_0000--s-0 \
--instruction "pick up the peg" \
--eval_episodes 10 \
--max_duration_s 60.0 \
--control_frequency_hz 10 \
--use_wrist_camera 1 \
--use_exterior_camera 0 \
--policy_host 127.0.0.1 \
--policy_port 8000 \
--outputdir ./logs/policy_eval_real \
--seed 0 \
--hidden_dims 1024 \
--network_type transformer \
--rl_noise_horizon 8 \
--action_scale 0.5 \
--max_joint_speed_rad_s 0.5 \
--diagnostic_dir ./logs/diagnostics
