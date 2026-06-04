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
bash examples/scripts/run_real.sh
```

For the Wrist-DINO state-only real-world variant, fill in the camera IDs and remote policy host/port in `examples/scripts/run_real_dino.sh`, then run:
```
bash examples/scripts/run_real_dino.sh
```
This variant uses only the wrist camera for the RL steering policy image feature, featurized by `facebook/dinov2-small` into a 384-D CLS embedding. The full RL state is 2440-D: 7 joint positions, 1 gripper position, 2048-D pi0 VLM embedding, and 384-D DINO feature. The pi0 policy request still keeps its expected DROID inputs. The first run may download/cache the DINO-v2-small model through HuggingFace Transformers.

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
--resize_image 128 \
--control_frequency_hz 15 \
--use_wrist_camera 1 \
--use_exterior_camera 1 \
--policy_host <GPU_SERVER_IP_OR_127.0.0.1> \
--policy_port 8000 \
--outputdir ./logs/policy_eval_real \
--seed 0 \
--add_states 1 \
--hidden_dims 1024 \
--num_qs 2 \
--action_magnitude 2.0
```
The evaluator opens a Tkinter GUI with live wrist and exterior-camera previews. Camera serial IDs are hardcoded in the evaluator (`17396664` for Zed Mini wrist, `241122302552` for RealSense exterior); the command only chooses whether each camera is used. Click `Start next` to begin each rollout, click `Success` or `Failure` to label the trajectory, or let the rollout timeout to mark it as failure automatically. Each labeled rollout triggers `env.reset()` and then waits for the next `Start next`. Results are written to `eval_results.csv`, and videos are saved as `eval_video_<episode_id>.mp4` in `--outputdir`.

To evaluate the pi0 policy alone with wrist-camera observations only, keep the NUC and GPU server commands above running and use:
```
cd ~/yingxi/dsrl_pi0
conda activate dsrl_pi0

python3 examples/evaluate_pi0_real.py \
--instruction "Insert the peg into the hole." \
--eval_episodes 15 \
--max_rollout_steps 400 \
--query_freq 8 \
--control_frequency_hz 10 \
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
