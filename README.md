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

# install ManiSkill and sapien (needed for PegInsertionVertical sim)
pip install mani-skill sapien
pip install -e RoboFPE
# DINOv2 weights are auto-downloaded by HuggingFace Transformers on first run;
# set HF_ENDPOINT if needed (e.g. export HF_ENDPOINT=https://hf-mirror.com)
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
### PegInsertionVertical Simulation — Wrist-DINO state (multi-seed sweep)

This variant runs **DSRL on a simulated Franka peg-insertion task** with an observation
space that is **architecturally identical to the real-robot Wrist-DINO setup**:

```
state = [ proprio (8-D) | pi0 VLM embed (2048-D) | DINOv2-small CLS (384-D) ] = 2440-D
```

- **Policy**: `pi0_droid` (local, `/opt/yingxi/pi0_droid`; DroidInputs format, action_horizon=8)
- **Environment**: `PegInsertionVertical-v1` from [RoboFPE](https://github.com/luyingxi35/RoboFPE), running in an **isolated subprocess** (robofac conda env) to avoid ManiSkill/sapien dependency conflicts with the JAX training process
- **Reset distribution**: robot joints use the DROID reset pose; the hole stays fixed, and only the peg is randomized in a small ring around the hole by the dsrl_pi0 remote-env wrapper (RoboFPE source logic is unchanged)
- **Cameras**: 224×224 `base_camera` (exterior) + `hand_camera` (wrist, `panda_wristcam`); DINOv2 runs on the **wrist** image, mirroring `WristDinoObservationBuilder` in `run_real_dino.sh`
- **RL agent**: StateSAC + Transformer — identical hyperparameters to `run_real_dino.sh`
- **Success**: rule-based `has_peg_inserted()` geometry check, no human labelling needed

**Multi-seed sweep with automatic early stopping and curve plotting:**
```bash
# Runs seeds 0 / 1 / 2 in PARALLEL, one seed per GPU (GPU 0 / 1 / 2).
# Every 1000 env steps: eval (10 episodes) + save checkpoint.
# Stops each seed once success rate >= 95% for 2 consecutive evals.
# Plots mean +/- std curve after all seeds finish.
bash examples/scripts/run_sim_dino.sh

# Custom seeds:
bash examples/scripts/run_sim_dino.sh --seeds "0 1 2 3 4"

# Custom GPU assignment (e.g. use GPUs 4-6 instead of 0-2):
bash examples/scripts/run_sim_dino.sh --seeds "0 1 2" --gpus "4 5 6"
```

Each seed's stdout/stderr is redirected to `logs/DSRL_pi0_SimDino/seed<N>_gpu<G>.log`.
Monitor live progress with:
```bash
tail -f logs/DSRL_pi0_SimDino/seed*.log
```

Output layout:
```
logs/DSRL_pi0_SimDino/
├── dsrl_pi0_sim_dino_s0_<hash>/
│   ├── checkpoint_<grad_step>   <- checkpoints at each eval milestone
│   └── eval_curve.csv           <- columns: env_steps, success_rate
├── dsrl_pi0_sim_dino_s1_<hash>/  ...
├── dsrl_pi0_sim_dino_s2_<hash>/  ...
└── sim_dino_curve.png        <- aggregated mean +/- std curve
```

**Simulation policy evaluation** (run from the `dsrl_pi0` environment; ManiSkill
rollouts are spawned in the `robofac` subprocess environment):
```bash
bash examples/scripts/eval_sim_pi0.sh

bash examples/scripts/eval_sim_dino.sh \
    --restore_path ./logs/DSRL_pi0_SimDino/dsrl_pi0_sim_dino_s0_<hash>
```
Both scripts default to `--device_id 0`, which sets `CUDA_VISIBLE_DEVICES=0` for
the dsrl/OpenPI process so evaluation does not claim every GPU. Use
`--device_id <GPU_ID>` to select a different card.

Use `Ctrl-C` to stop these eval scripts. If a test eval is stuck, kill only the
eval parent process group, not every `mani_skill_server.py` on the machine:
```bash
pgrep -af 'evaluate_pi0_sim.py|evaluate_policy_sim_dino.py|pi0_eval_sim|dino_eval_sim'
ps -o pid,ppid,pgid,sid,stat,etime,cmd -p <PID>
kill -INT -<PGID>
kill -TERM -<PGID>
kill -KILL -<PGID>   # only if TERM does not exit after a few seconds
```

**Plot only** (re-plot from existing CSVs without re-running training):
```bash
python3 examples/plot_sim_dino_curve.py \
    --log_dir  ./logs/DSRL_pi0_SimDino \
    --output   ./logs/DSRL_pi0_SimDino/sim_dino_curve.png \
    --stop_line 0.95 \
    --title    "PegInsertionVertical — DSRL (wrist-aligned)"
```

**Live curve during training** — each seed appends a row to `eval_curve.csv` after every eval, so you can re-run the plot command at any point during training to inspect progress.

### PegInsertionVertical — Dense Reward (Robometer progress estimation)

This variant adds a **dense Robometer progress reward** on top of the sparse binary signal:

```
dense_reward[t] = binary_reward[t]  +  progress_reward_scale × Robometer_progress(t)
```

- `binary_reward`: −1 at every intermediate step, 0 at the final step of a successful rollout (unchanged from `run_sim_dino.sh`)
- `Robometer_progress(t)` ∈ [0, 1]: per-query-step progress score from a fine-tuned
  **Qwen3-VL-4B Robometer model** (discrete 20-bin head, checkpoint `/opt/yingxi/checkpoint-400`)
- The **exterior camera** (`base_camera`, 224×224) is buffered at each policy-query step
  (every 8 env steps); all `query_steps` frames are sent to Robometer at rollout end
- Robometer runs in a **dedicated subprocess** (`/opt/yingxi/envs/robometer/bin/python3`,
  GPU PyTorch 2.8+cu128) to avoid conflicts with the JAX training process
- Reward scale default: `progress_reward_scale = 1.0`

```bash
# Runs seeds 0 / 1 / 2 in PARALLEL on GPUs 0 / 1 / 2.
bash examples/scripts/run_sim_dino_dense.sh

# Custom seeds / GPUs:
bash examples/scripts/run_sim_dino_dense.sh --seeds "0 1 2" --gpus "4 5 6"
```

**Smoke test** (verify Robometer does not OOM on a 75-frame timeout rollout).
The `--gpu` flag is required — Robometer needs ~8 GB VRAM on a free GPU:
```bash
# Check free GPU memory first:
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader

/opt/yingxi/envs/dsrl_pi0/bin/python3 examples/tests/test_smoke_robometer.py \
    --video-dir logs/pi0_eval_sim_20260618_221150 \
    --max-frames 75 \
    --gpu <FREE_GPU_ID>
```
Expected output: `PASS: No OOM, output length correct` in ~5 s.

**Integration test** (one ManiSkill rollout with random actions + Robometer reward calibration,
saves `reward_frames.mp4` and `reward_progress_viz.png`):
```bash
/opt/yingxi/envs/dsrl_pi0/bin/python3 examples/tests/test_dense_reward_rollout.py \
    --gpu <FREE_GPU_ID> \
    --output-dir /tmp/dense_test \
    --max-timesteps 80
```
The visualization shows exterior-camera frames (top row, one per query step) with
per-frame `progress` labels, and a Robometer progress line chart (bottom).

**WandB reward metrics** (logged to `rollout/reward/*` after every rollout):

| Metric | Description |
|--------|-------------|
| `rollout/reward/dense_mean` | Mean dense reward across all query steps in the rollout |
| `rollout/reward/dense_max` | Max dense reward in the rollout |
| `rollout/reward/dense_min` | Min dense reward in the rollout |
| `rollout/reward/dense_last` | Dense reward of the final query step (0 on success, <0 otherwise) |
| `rollout/reward/dense_sum` | Sum of dense rewards (≈ episode return) |
| `rollout/reward/binary_mean` | Mean binary component (always −1, or 0 at success final step) |
| `rollout/reward/binary_last` | 0 on success, −1 on failure |
| `rollout/reward/progress_mean` | Mean Robometer progress score across query steps |
| `rollout/reward/progress_max` | Max Robometer progress score |
| `rollout/reward/progress_min` | Min Robometer progress score |
| `rollout/reward/progress_last` | Robometer progress at the final query step |
| `rollout/reward/query_steps` | Number of policy queries in this rollout |
| `rollout/reward/env_steps` | Total env steps in this rollout |

> **Note — one-time robometer code patches** (already applied in this repo):
> The three patches below are required because the robometer checkpoint was saved with
> HuggingFace `Trainer` + FSDP and the env uses `transformers 5.5.0`:
>
> | File | Change |
> |------|--------|
> | `RoboFPE/robometer/robometer/utils/setup_utils.py` | Non-PEFT checkpoint loading: use `_load_checkpoint_weights_from_safetensors` instead of `from_pretrained` (avoids embed_tokens shape mismatch caused by re-initialisation without special tokens) |
> | `RoboFPE/robometer/robometer/evals/eval_server.py` | Pass `mm_token_type_ids` to Qwen3-VL model (`transformers ≥ 5.x` requirement for multimodal RoPE) |
> | `examples/robometer_reward_server.py` | Resolve `${loss.*}` OmegaConf interpolations from `config.yaml` before building `ExperimentConfig` (prevents progress head from being initialised with 1 bin instead of 20) |

Key differences across training variants:

| | `run_libero.sh` | `run_real_dino.sh` | `run_sim_dino.sh` | `run_sim_dino_dense.sh` |
|---|---|---|---|---|
| Environment | LIBERO (MuJoCo) | Franka DROID (real) | PegInsertionVertical (ManiSkill, subprocess) | PegInsertionVertical (ManiSkill, subprocess) |
| SAC | PixelSAC + CNN | StateSAC + Transformer | StateSAC + Transformer | StateSAC + Transformer |
| Observation | 64x64 pixels | proprio+VLM+DINO (2440-D) | proprio+VLM+DINO (2440-D) | proprio+VLM+DINO (2440-D) |
| Camera for DINOv2 | — | wrist (RealSense) | wrist (`hand_camera`, 224x224) | wrist (`hand_camera`, 224x224) |
| pi0 inference | local | remote server | local | local |
| Reward | sparse -1/0 (env) | sparse -1/0 (human) | sparse -1/0 (env) | binary -1/0 + Robometer progress |
| Success signal | env reward | human GUI label | rule-based geometry | rule-based geometry |
| Multi-seed sweep | — | — | `run_sim_dino.sh` | `run_sim_dino_dense.sh` |

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
