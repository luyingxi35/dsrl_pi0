<div align="center">

# DSRL for π₀: Diffusion Steering via Reinforcement Learning

## [[website](https://diffusion-steering.github.io)]      [[paper](https://arxiv.org/abs/2506.15799)]

</div>


## Overview
This repository provides the official implementation for our paper: [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799) (CoRL 2025).

Specifically, it contains a JAX-based implementation of DSRL (Diffusion Steering via Reinforcement Learning) for steering a pre-trained generalist policy, [π₀](https://github.com/Physical-Intelligence/openpi), across various environments, including:

- **Simulation:** Libero, Aloha  
- **Real Robot:** Franka

If you find this repository useful for your research, please cite:

```
@article{wagenmaker2025steering,
  author    = {Andrew Wagenmaker and Mitsuhiko Nakamoto and Yunchu Zhang and Seohong Park and Waleed Yagoub and Anusha Nagabandi and Abhishek Gupta and Sergey Levine},
  title     = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```

## Installation
1. Create a conda environment:
```
conda create -n dsrl_pi0 python=3.11.11
conda activate dsrl_pi0
```

2. Clone this repo with all submodules
```
git clone git@github.com:nakamotoo/dsrl_pi0.git --recurse-submodules
cd dsrl_pi0
```

3. Install all packages and dependencies
```
pip install -e .
pip install -r requirements.txt
pip install "jax[cuda12]==0.5.0"

# install openpi
pip install -e openpi
pip install -e openpi/packages/openpi-client

# install Libero
pip install -e LIBERO
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu # needed for libero
```

## Training (Simulation)
Libero
```
bash examples/scripts/run_libero.sh
```
Aloha
```
bash examples/scripts/run_aloha.sh
```
### Training Logs
We provide sample W&B runs and logs: https://wandb.ai/mitsuhiko/DSRL_pi0_public

## Training (Real)
For real-world experiments, we use the remote hosting feature from pi0 (see [here](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)) which enables us to host the pi0 model on a higher-spec remote server, in case the robot's client machine is not powerful enough. 

0. Setup Franka robot and install DROID package [[link](https://github.com/droid-dataset/droid.git)].
   The aligned runtime is:
   - NUC: Franka + Polymetis + DROID robot server
   - Laptop/workstation: DROID client, cameras, and the DSRL steering loop
   - Remote GPU server: OpenPI policy server

1. [On the NUC] Start the DROID robot server so robot control stays on the NUC as in DROID.
```
cd ~/yingxi/droid
conda activate polymetis-local
python scripts/server/run_server.py
```

2. [On the remote GPU server] Host the pi0 DROID model:
```
cd openpi && python scripts/serve_policy.py --env=DROID
```

3. [On the robot laptop/workstation] Fill in camera IDs and remote policy host/port in `examples/scripts/run_real.sh`, then run DSRL:
```
export HF_ENDPOINT=https://hf-mirror.com
bash examples/scripts/run_real.sh
```

For the Wrist-DINO state-only real-world variant, fill in the camera IDs and remote policy host/port in `examples/scripts/run_real_dino.sh`, then run:
```
export HF_ENDPOINT=https://hf-mirror.com
bash examples/scripts/run_real_dino.sh [--resume_from [RESUME_DIR]]
```
This variant uses only the wrist camera for the RL steering policy image feature, featurized by `facebook/dinov2-small` into a 384-D CLS embedding. The full RL state is 2440-D: 7 joint positions, 1 gripper position, 2048-D pi0 VLM embedding, and 384-D DINO feature. The pi0 policy request still keeps its expected DROID inputs. The first run may download/cache the DINO-v2-small model through HuggingFace Transformers.

#### Resuming a Wrist-DINO training run

Every `--checkpoint_interval` gradient steps (default 10 000) the training loop automatically saves three files to `outputdir`:
- `checkpoint_<step>` — Flax agent checkpoint (actor / critic / temp params+optimizer state / `_rng`)
- `training_state.json` — gradient-step counter, episode counts, success counts, temperature scalar
- `replay_buffer.pkl` — full replay buffer snapshot

To resume from the latest checkpoint, pass `--resume_from <outputdir>` in place of the normal run:
```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 examples/launch_train_real_dino.py \
  --resume_from $EXP/DSRL_pi0_FrankaDroid/<your_run_name> \
  --algorithm state_sac \
  --env franka_droid \
  --prefix dsrl_pi0_real_dino \
  --wandb_project DSRL_pi0_FrankaDroid \
  --batch_size 256 \
  --max_steps 500000 \
  --multi_grad_step 30 \
  --query_freq 8 \
  --rl_noise_horizon 8 \
  --network_type transformer \
  --instruction 'pick up the blue peg' \
  --wrist_camera_id "<WRIST_CAM_ID>" \
  --policy_host "<GPU_SERVER_IP>" \
  --policy_port 8000
```
`--resume_from` reuses the existing output directory, restores the agent weights from the latest checkpoint, reloads the training counters, and refills the replay buffer. It is mutually exclusive with `--restore_path`.

### Action Execution Parameters

The training loop uses a HighFreqController (200 Hz, on the NUC) for smooth joint trajectory execution, matching the eval-time setup. Two key parameters control arm speed:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--action_scale` | `0.5` | Speed multiplier on DROID's training max joint delta (0.2 rad/step). `1.0` = full training speed, `0.5` = half speed (safer default). |
| `--max_joint_speed_rad_s` | `0.5` | NUC-side per-joint speed cap (rad/s). Conservative default; increase to `1.5` or higher for faster execution. |

**Recommended starting point**: `--action_scale 0.5 --max_joint_speed_rad_s 1.5` allows arm motion up to 1.0 rad/step at 10 Hz without the NUC cap triggering.

The default `--max_joint_speed_rad_s 0.5` is intentionally conservative for initial safety validation. At `action_scale=0.5` the arm moves up to 0.5 rad/step; the NUC cap at 0.5 rad/s will extend execution time by ~10× unless raised.

## Real Policy Evaluation
After training, evaluate real-world policies with the same three-machine runtime.

1. [On the NUC] Start the DROID robot server:
```
cd ~/yingxi/droid
conda activate polymetis-local

python scripts/server/run_server.py
```

2. [On the remote GPU server] Start the OpenPI/pi0 policy server:
```
cd ~/yingxi/dsrl_pi0/openpi
conda activate dsrl_pi0

python scripts/serve_policy.py --env=DROID --port=8000
```

3. [On the robot laptop/workstation] Evaluate a trained DSRL checkpoint with the standalone GUI-based evaluator:
```
cd ~/yingxi/dsrl_pi0
conda activate dsrl_pi0

python3 examples/evaluate_policy_real.py \
--restore_path ./logs/DSRL_pi0_FrankaDroid/<exp_name_with_checkpoints> \
--instruction "put the spoon on the plate" \
--eval_episodes 10 \
--max_rollout_steps 200 \
--query_freq 8 \
--control_frequency_hz 10 \
--external_camera right \
--max_joint_speed_rad_s 0.5 \
--use_wrist_camera 1 \
--use_exterior_camera 0 \
--left_camera_id "" \
--right_camera_id "" \
--wrist_camera_id 17396664 \
--policy_host <GPU_SERVER_IP_OR_127.0.0.1> \
--policy_port 8000 \
--outputdir ./logs/policy_eval_real \
--seed 0 \
--hidden_dims 1024 \
--network_type transformer \
--rl_noise_horizon 8
```
The evaluator opens a Tkinter GUI with live wrist and selected exterior-camera previews. Click `Start next` to begin each rollout, click `Success` or `Failure` to label the trajectory, or let the rollout timeout to mark it as failure automatically. Each labeled rollout triggers `env.reset()` and then waits for the next `Start next`. Results are written to `eval_results.csv`, and videos are saved as `eval_video_<episode_id>.mp4` in `--outputdir`.

`--control_frequency_hz` controls the main rollout loop and action timestamp spacing. DSRL eval uses train-aligned synchronous inference: every `--query_freq` control steps it builds the DSRL state, predicts RL noise, calls the pi0 server, integrates the returned chunk, and schedules at most `query_freq` targets. With the default `--control_frequency_hz 10 --query_freq 8`, DSRL/pi0 inference runs about every 0.8 seconds. `--inference_frequency_hz` is intentionally not supported by the DSRL evaluator.

To evaluate the pi0 policy alone with wrist-camera observations only, keep the NUC and GPU server commands above running and use:
```
cd ~/yingxi/dsrl_pi0
conda activate dsrl_pi0

python3 examples/evaluate_pi0_real.py \
--instruction "pick up the blue peg" \
--eval_episodes 10 \
--max_duration_s 60 \
--execution_steps 8 \
--action_scale 0.5 \
--control_frequency_hz 10 \
--inference_frequency_hz 3 \
--controller_frequency 200 \
--max_joint_speed_rad_s 0.5 \
--use_wrist_camera 1 \
--use_exterior_camera 0 \
--policy_host 127.0.0.1 \
--policy_port 8000 \
--outputdir ./logs/pi0_eval_real
```
This pi0-only evaluator does not load a DSRL checkpoint or send RL noise. With `--use_wrist_camera 1 --use_exterior_camera 0`, it sends one real wrist image to OpenPI and leaves the other model image slots masked out. To evaluate with a Zed Mini wrist camera plus a RealSense exterior view, run with:
```
--use_wrist_camera 1 \
--use_exterior_camera 1
```
For exterior-only evaluation, use `--use_wrist_camera 0 --use_exterior_camera 1`. In all cases, only enabled real cameras are sent to OpenPI: RealSense exterior maps to `observation/exterior_image_1_left`, Zed wrist maps to `observation/wrist_image_left`, and absent DROID image slots remain masked out. Restart the OpenPI policy server after updating this repository so the empty-slot transform is loaded.

Both evaluators accept `--max_joint_speed_rad_s` and pass it to the NUC-side `add_waypoints` safety cap. Keep it conservative for first validation; increase it only when the commanded motion should not be slowed by the NUC cap.

## Test
1. Test observation:
On the NUC: 
```
cd ~/yingxi/droid
conda activate polymetis-local
python scripts/server/run_server.py
```
On GPU server:
```
cd ~/yingxi/dsrl_pi0/openpi
conda activate dsrl_pi0
python scripts/serve_policy.py --env=DROID --port=8000
```
On the workstation:
```
bash examples/scripts/check_real_dino_obs.sh
```

## Credits
This repository is built upon [jaxrl2](https://github.com/ikostrikov/jaxrl2) and [PTR](https://github.com/Asap7772/PTR) repositories. 
In case of any questions, bugs, suggestions or improvements, please feel free to contact me at nakamoto\[at\]berkeley\[dot\]edu 
