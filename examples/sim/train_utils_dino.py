"""Simulation training utilities for train_sim_dino.py.

PegInsertionVertical-v1 (ManiSkill2/SAPIEN) + pi0_droid (local) + DINOv2 + StateSAC.

Observation space mirrors train_real_dino.py:
    state = [proprio(8) | pi0_VLM_embed(2048) | DINOv2_CLS(384)]  → STATE_DIM = 2440

Key simplifications vs. train_utils_real.py (no hardware):
  - No HumanEvalUI: success judged automatically by env.evaluate()
  - No HighFreqController / waypoint scheduling / latency compensation
  - Direct env.step(); timeout = env truncation or max_timesteps reached
"""
import csv
from pathlib import Path
import os
from collections import deque

import numpy as np
import jax
from tqdm import tqdm

from openpi_client import image_tools
from jaxrl2.utils.noise_utils import make_full_horizon_noise

# Reuse buffer insertion from train_utils_sim (identical logic, no modification needed)
from examples.sim.train_utils import add_online_data_to_buffer  # noqa: F401
from examples.sim.action_utils import pi0_vel_chunk_to_joint_pos_actions


# ── Video helper ──────────────────────────────────────────────────────────────

def _save_traj_video(
    frames: list,
    path: Path,
    fps: float = 15.0,
) -> None:
    """Save a list of uint8 RGB frames as an mp4 video."""
    if not frames:
        return
    try:
        import numpy as _np
        from moviepy.editor import ImageSequenceClip
        from moviepy.video.io.ffmpeg_writer import ffmpeg_write_video
        arr = _np.stack([_np.asarray(f, dtype=_np.uint8) for f in frames])
        clip = ImageSequenceClip(list(arr), fps=fps)
        ffmpeg_write_video(clip, str(path), fps, codec='libx264',
                           audiofile=None, logger=None)
        print(f'Saved rollout video: {path}')
    except Exception as exc:
        print(f'Warning: could not save rollout video {path}: {exc}')


# ── Constants (mirrors train_real_dino.py) ─────────────────────────────────────
PROPRIO_DIM           = 8
PI0_VLM_EMBED_DIM     = 2048
DINO_V2_SMALL_CLS_DIM = 384
PI0_NOISE_DIM         = 32
STATE_DIM             = PROPRIO_DIM + PI0_VLM_EMBED_DIM + DINO_V2_SMALL_CLS_DIM  # 2440


# ── DINOv2 feature extractor ───────────────────────────────────────────────────
# Copied from train_real_dino.py to avoid importing robot hardware dependencies.
class WristDinoFeatureExtractor:
    """Runs DINOv2-small on a single RGB image and returns the CLS token (384-D)."""

    def __init__(self, model_name: str, device: str):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "SimDino training requires torch and transformers. "
                "Install them or activate the correct conda environment."
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._device = torch.device(device)
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device)
        self._model.eval()

    @property
    def feature_dim(self) -> int:
        return DINO_V2_SMALL_CLS_DIM

    def encode(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        feature = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()[0].astype(np.float32)
        if feature.shape != (self.feature_dim,):
            raise RuntimeError(
                f"Expected DINO CLS feature shape ({self.feature_dim},), got {feature.shape}"
            )
        return feature


# ── Sim observation builder ────────────────────────────────────────────────────

class SimDinoObservationBuilder:
    """Builds the RL state vector from sim obs: proprio + pi0 VLM embed + DINOv2 CLS.

    Mirrors WristDinoObservationBuilder from train_real_dino.py.
    STATE_DIM = 8 + 2048 + 384 = 2440.
    """

    def __init__(self, dino_extractor: WristDinoFeatureExtractor):
        self._dino = dino_extractor

    def build(
        self,
        qpos: np.ndarray,
        ext_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        pi0_obs: dict,
        agent_dp,
    ) -> dict:
        """
        Args:
            qpos:     (9,) robot joint positions [7 arm + 2 gripper fingers]
            ext_rgb:   (H, W, 3) uint8 base_camera (exterior) - used for pi0 VLM
            wrist_rgb: (H, W, 3) uint8 hand_camera (wrist)    - used for DINOv2
            pi0_obs:   dict in DroidInputs format (already built by _obs_to_pi0_input)
            agent_dp:  local pi0 policy exposing get_prefix_rep()

        Returns:
            {"state": ndarray of shape (1, STATE_DIM, 1)}
        """
        # 1. pi0 VLM embedding (PaliGemma prefix forward pass, cheap vs full infer)
        prefix_rep, _ = agent_dp.get_prefix_rep(pi0_obs)       # (1, S_prefix, 2048)
        vlm_embed = np.asarray(prefix_rep[:, -1, :]).reshape(-1).astype(np.float32)  # (2048,)
        if vlm_embed.shape != (PI0_VLM_EMBED_DIM,):
            raise RuntimeError(
                f"Expected VLM embed shape ({PI0_VLM_EMBED_DIM},), got {vlm_embed.shape}"
            )

        # 2. DINOv2 CLS token from WRIST camera (mirrors real-robot WristDinoObservationBuilder)
        dino_cls = self._dino.encode(wrist_rgb)                 # (384,)

        # 3. Proprioception: 7 joint positions + 1 gripper (average of 2 fingers)
        proprio = np.concatenate([
            qpos[:7].astype(np.float32),
            [float(np.mean(qpos[7:9]))],
        ])  # (8,)

        state = np.concatenate([proprio, vlm_embed, dino_cls]).astype(np.float32)  # (2440,)
        if state.shape != (STATE_DIM,):
            raise RuntimeError(f"Expected state shape ({STATE_DIM},), got {state.shape}")
        return {"state": state[np.newaxis, ..., np.newaxis]}


# ── Helper: extract raw obs from ManiSkill env obs dict ───────────────────────

def _extract_sim_obs(env_obs: dict):
    """Return (qpos, ext_rgb, wrist_rgb) from env obs dict.

    Supports two formats:
    1. ManiSkillRemoteEnv: {"qpos":(9,), "ext":(H,W,3), "wrist":(H,W,3)}
    2. Legacy ManiSkill gym: {"agent":{"qpos":...}, "sensor_data":{...}}
    """
    if "ext" in env_obs and "wrist" in env_obs:
        # ManiSkillRemoteEnv format (subprocess client)
        return (np.asarray(env_obs["qpos"],  dtype=np.float32),
                np.asarray(env_obs["ext"],   dtype=np.uint8),
                np.asarray(env_obs["wrist"], dtype=np.uint8))

    # Legacy: direct gym obs from ManiSkill
    qpos = env_obs["agent"]["qpos"]
    if hasattr(qpos, "cpu"): qpos = qpos.cpu().numpy()
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)       # (9,)

    def _rgb(key):
        r = env_obs["sensor_data"][key]["rgb"]
        if hasattr(r, "cpu"): r = r.cpu().numpy()
        r = np.asarray(r, dtype=np.uint8)
        return r[0] if r.ndim == 4 else r

    ext   = _rgb("base_camera")
    wrist = (_rgb("hand_camera")
             if "hand_camera" in env_obs.get("sensor_data", {})
             else np.zeros_like(ext))
    return qpos, ext, wrist


def _obs_to_pi0_input(qpos: np.ndarray, ext_rgb: np.ndarray,
                      wrist_rgb: np.ndarray, instruction: str) -> dict:
    """Build pi0 input dict in DroidInputs format.

    Mirrors real-robot get_pi0_input_train() exactly:
      exterior (base_camera)  -> observation/exterior_image_1_left
      wrist    (hand_camera)  -> observation/wrist_image_left  (real image!)
    """
    ext_224   = image_tools.convert_to_uint8(ext_rgb)
    wrist_224 = image_tools.convert_to_uint8(wrist_rgb)
    return {
        "observation/exterior_image_1_left": ext_224,
        "observation/wrist_image_left":      wrist_224,
        "observation/joint_position":        qpos[:7].astype(np.float32),
        "observation/gripper_position":      np.array([np.clip(np.mean(qpos[7:9]) / 0.04, 0.0, 1.0)], dtype=np.float32),  # normalize panda finger (0~0.04m) to DROID [0,1]
        "prompt":                            instruction,
    }


# ── Trajectory collection ──────────────────────────────────────────────────────

def collect_traj(variant, agent, env, i, agent_dp, obs_builder, video_dir=None):
    """Collect one trajectory in simulation.

    Mirrors train_utils_real.py:collect_traj() but simplified for sim:
      - No HumanEvalUI: success judged by env.evaluate() / has_peg_inserted()
      - No HighFreqController: direct env.step()
      - No latency compensation
      - Timeout failure: failure_reason = "timeout" when max_timesteps exceeded
    """
    query_frequency = variant.query_freq
    max_timesteps   = variant.max_timesteps
    instruction     = variant.instruction

    agent._rng, rng = jax.random.split(agent._rng)

    is_success     = False
    failure_reason = "timeout"
    env_steps      = 0

    env_obs, _ = env.reset()

    # Video frame buffers (populated only when video_dir is set)
    _side_frames:  list = []
    _wrist_frames: list = []

    action_list = []
    obs_list    = []
    actions     = None
    sim_actions = None

    for t in tqdm(range(max_timesteps)):
        qpos, ext_rgb, wrist_rgb = _extract_sim_obs(env_obs)
        if video_dir is not None:
            _side_frames.append(ext_rgb.copy())
            _wrist_frames.append(wrist_rgb.copy())
        pi0_obs                  = _obs_to_pi0_input(qpos, ext_rgb, wrist_rgb, instruction)

        if t % query_frequency == 0:
            rng, key = jax.random.split(rng)

            obs_dict = obs_builder.build(qpos, ext_rgb, wrist_rgb, pi0_obs, agent_dp)

            if i == 0:
                # First iteration: random Gaussian noise → evaluates base pi0 policy
                initial_noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                actions_noise, noise = make_full_horizon_noise(
                    initial_noise[0], agent.action_chunk_shape
                )
            else:
                # SAC predicts noise that steers pi0 toward higher-reward regions
                actions_noise = agent.sample_actions(obs_dict)
                actions_noise, noise = make_full_horizon_noise(
                    actions_noise, agent.action_chunk_shape
                )

            action_list.append(actions_noise)
            obs_list.append(obs_dict)

            # pi0 denoises with the RL-predicted noise → executable action chunk
            actions = agent_dp.infer(pi0_obs, noise=noise)["actions"]
            sim_actions = pi0_vel_chunk_to_joint_pos_actions(
                qpos,
                actions,
                action_scale=getattr(variant, "action_scale", 0.5),
                execution_steps=query_frequency,
            )

        action_8d = np.asarray(sim_actions[t % query_frequency], dtype=np.float32)
        env_obs, _reward, terminated, truncated, info = env.step(action_8d)
        done      = bool(terminated) or bool(truncated)
        env_steps = t + 1

        # Workspace constraint check (pre/post-grasp bounding box)
        if bool(info.get("workspace_violated", False)):
            failure_reason = "workspace_" + info.get("phase", "pre_grasp")
            break

        # Rule-based success check (has_peg_inserted)
        if bool(info["success"]):
            is_success     = True
            failure_reason = ""
            break

        if done:
            # env truncated due to max_episode_steps → timeout failure
            break

    # Save first-rollout diagnostic videos
    if video_dir is not None:
        vdir = Path(video_dir)
        vdir.mkdir(parents=True, exist_ok=True)
        _save_traj_video(_side_frames,  vdir / 'traj_0_side.mp4')
        _save_traj_video(_wrist_frames, vdir / 'traj_0_wrist.mp4')

    # Append final observation (after last step)
    qpos_last, ext_last, wrist_last = _extract_sim_obs(env_obs)
    pi0_obs_last  = _obs_to_pi0_input(qpos_last, ext_last, wrist_last, instruction)
    obs_dict_last = obs_builder.build(qpos_last, ext_last, wrist_last, pi0_obs_last, agent_dp)
    obs_list.append(obs_dict_last)

    print(f"Rollout Done: success={is_success}, "
          f"reason={failure_reason or 'success'}, steps={env_steps}")

    # Sparse -1/0 reward (same as train_real_dino.py)
    query_steps = len(action_list)
    if query_steps == 0:
        rewards_arr = np.array([], dtype=np.float32)
        masks_arr   = np.array([], dtype=np.float32)
    elif is_success:
        rewards_arr = np.concatenate([-np.ones(query_steps - 1), [0.0]])
        masks_arr   = np.concatenate([np.ones(query_steps - 1), [0.0]])
    else:
        rewards_arr = -np.ones(query_steps, dtype=np.float32)
        masks_arr   =  np.ones(query_steps, dtype=np.float32)

    return {
        "observations":   obs_list,
        "actions":        action_list,
        "rewards":        rewards_arr,
        "masks":          masks_arr,
        "is_success":     is_success,
        "failure_reason": failure_reason,
        "env_steps":      env_steps,
    }


# ── Training loop ──────────────────────────────────────────────────────────────

def trajwise_alternating_training_loop(
    variant,
    agent,
    env,
    eval_env,
    online_replay_buffer,
    replay_buffer,
    wandb_logger,
    shard_fn=None,
    agent_dp=None,
    obs_builder=None,
    # ── Sweep / early-stop parameters ──────────────────────────────────────────
    eval_env_step_interval: int   = -1,
    stop_success_rate:      float = 0.95,
    stop_window:            int   = 2,
    eval_csv_path:          str   = None,
):
    """Sim training loop: collect trajectory → gradient updates, repeat.

    Mirrors train_utils_sim.py:trajwise_alternating_training_loop but:
      - Uses num_initial_traj_collect (traj count) like train_real_dino.py
      - Logs failure_reason to WandB like train_real_dino.py
      - No GUI, no robot_io, no timing

    Sweep / early-stop extensions (active when eval_env_step_interval > 0):
      eval_env_step_interval: trigger eval + checkpoint every N env steps
      stop_success_rate:      stop when last `stop_window` evals all >= this value
      stop_window:            number of consecutive evals that must hit the threshold
      eval_csv_path:          if set, append (env_steps, success_rate) rows to this CSV
    """
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    i               = 0
    total_env_steps = 0
    total_num_traj  = 0
    successes       = 0

    # ── Sweep state ────────────────────────────────────────────────────────────
    _sweep_active        = eval_env_step_interval > 0
    _next_eval_env_step  = eval_env_step_interval if _sweep_active else float("inf")
    _recent_success_rate = deque(maxlen=stop_window)  # sliding window for early stop

    if _sweep_active and eval_csv_path is not None:
        os.makedirs(os.path.dirname(eval_csv_path) if os.path.dirname(eval_csv_path) else ".",
                    exist_ok=True)
        # Write CSV header (overwrite any stale file from a previous run)
        with open(eval_csv_path, "w", newline="") as _f:
            csv.writer(_f).writerow(["env_steps", "success_rate"])
        print(f"Eval CSV: {eval_csv_path}")

    wandb_logger.log({"num_online_samples": 0}, step=i)
    wandb_logger.log({"num_online_trajs":   0}, step=i)
    wandb_logger.log({"env_steps":          0}, step=i)

    _converged = False

    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps and not _converged:
            _video_dir = variant.outputdir if total_num_traj == 0 else None
            traj = collect_traj(variant, agent, env, i, agent_dp, obs_builder,
                               video_dir=_video_dir)
            total_num_traj += 1
            successes      += int(traj["is_success"])
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            total_env_steps += traj["env_steps"]
            print(
                f"buffer={len(online_replay_buffer)}, "
                f"traj={total_num_traj}, "
                f"env_steps={total_env_steps}, "
                f"successes={successes}"
            )

            wandb_logger.log({"is_success":     int(traj["is_success"])}, step=i)
            wandb_logger.log({"total_num_traj": total_num_traj},          step=i)
            wandb_logger.log(
                {f'rollout_result/{traj["failure_reason"] or "success"}': 1}, step=i
            )

            # ── Sweep: env-step triggered eval + ckpt + early stop ─────────────
            if _sweep_active and total_env_steps >= _next_eval_env_step:
                print(f"\n[Sweep] env_steps={total_env_steps} "
                      f"→ eval + checkpoint (interval={eval_env_step_interval})")

                sr = _perform_eval(
                    agent, eval_env, i, variant, wandb_logger, agent_dp, obs_builder
                )
                _recent_success_rate.append(sr)

                # Save checkpoint at this env-step milestone
                agent.save_checkpoint(
                    variant.outputdir, i,
                    keep_every_n_steps=-1,   # keep all sweep checkpoints
                )
                print(f"[Sweep] Checkpoint saved at grad_step={i}, "
                      f"env_steps={total_env_steps}, success_rate={sr:.3f}")

                # Log to CSV
                if eval_csv_path is not None:
                    with open(eval_csv_path, "a", newline="") as _f:
                        csv.writer(_f).writerow([total_env_steps, sr])

                # Log to WandB
                wandb_logger.log({"sweep/env_steps":     total_env_steps}, step=i)
                wandb_logger.log({"sweep/success_rate":  sr},              step=i)

                # Early-stop check
                if (len(_recent_success_rate) >= stop_window
                        and all(r >= stop_success_rate for r in _recent_success_rate)):
                    print(f"\n[Sweep] Converged! Last {stop_window} evals all "
                          f">= {stop_success_rate:.0%}.  Saving final checkpoint and stopping.")
                    agent.save_checkpoint(variant.outputdir, i, keep_every_n_steps=-1)
                    _converged = True
                    break

                _next_eval_env_step += eval_env_step_interval

            # First batch: extra gradient steps (data is precious even in sim)
            num_gradsteps = (
                5000
                if i == 0
                else len(traj["rewards"]) * variant.multi_grad_step
            )

            if total_num_traj >= variant.num_initial_traj_collect:
                for _ in range(num_gradsteps):
                    if _converged:
                        break

                    # Initial evaluation before any gradient update
                    if i == 0:
                        print("Evaluating initial checkpoint (base pi0_droid policy)...")
                        _perform_eval(
                            agent, eval_env, i, variant, wandb_logger,
                            agent_dp, obs_builder
                        )

                    batch       = next(replay_buffer_iterator)
                    update_info = agent.update(batch)

                    pbar.update()
                    i += 1

                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f"training/{k}": v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f"training/{k}", v, i)
                        wandb_logger.log(
                            {
                                "replay_buffer_size":       len(online_replay_buffer),
                                "episode_return":           float(np.sum(traj["rewards"])),
                                "is_success (exploration)": int(traj["is_success"]),
                            },
                            step=i,
                        )

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({"num_online_samples": len(online_replay_buffer)}, step=i)
                        wandb_logger.log({"num_online_trajs":   total_num_traj},           step=i)
                        wandb_logger.log({"env_steps":          total_env_steps},          step=i)
                        _perform_eval(
                            agent, eval_env, i, variant, wandb_logger,
                            agent_dp, obs_builder
                        )

                    if (
                        variant.checkpoint_interval != -1
                        and i % variant.checkpoint_interval == 0
                    ):
                        agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)

    if _converged:
        print(f"\nTraining converged at env_steps={total_env_steps}, grad_steps={i}.")
    else:
        print(f"\nTraining finished (max_steps reached). grad_steps={i}.")


def _perform_eval(agent, env, i, variant, wandb_logger, agent_dp, obs_builder) -> float:
    """Run eval_episodes deterministic rollouts and log success rate / avg return.

    Returns:
        success_rate (float in [0, 1])
    """
    query_frequency = variant.query_freq
    max_timesteps   = variant.max_timesteps
    instruction     = variant.instruction

    rng             = jax.random.PRNGKey(variant.seed + 456)
    success_rates   = []
    episode_returns = []

    for rollout_id in range(variant.eval_episodes):
        env_obs, _ = env.reset()
        actions      = None
        sim_actions  = None
        total_reward = 0.0
        is_success   = False

        for t in tqdm(range(max_timesteps)):
            qpos, ext_rgb, wrist_rgb = _extract_sim_obs(env_obs)
            pi0_obs                  = _obs_to_pi0_input(qpos, ext_rgb, wrist_rgb, instruction)

            if t % query_frequency == 0:
                rng, key = jax.random.split(rng)
                obs_dict = obs_builder.build(qpos, ext_rgb, wrist_rgb, pi0_obs, agent_dp)

                if i == 0:
                    noise_raw     = jax.random.normal(key, (1, *agent.action_chunk_shape))
                    _, noise      = make_full_horizon_noise(noise_raw[0], agent.action_chunk_shape)
                else:
                    actions_noise = agent.sample_actions(obs_dict)
                    _, noise      = make_full_horizon_noise(actions_noise, agent.action_chunk_shape)

                actions = agent_dp.infer(pi0_obs, noise=noise)["actions"]
                sim_actions = pi0_vel_chunk_to_joint_pos_actions(
                    qpos,
                    actions,
                    action_scale=getattr(variant, "action_scale", 0.5),
                    execution_steps=query_frequency,
                )

            action_8d = np.asarray(sim_actions[t % query_frequency], dtype=np.float32)
            env_obs, reward, terminated, truncated, info = env.step(action_8d)
            done          = bool(terminated) or bool(truncated)
            total_reward += float(reward) if reward is not None else 0.0

            # Workspace constraint check
            if bool(info.get("workspace_violated", False)):
                break

            if bool(info["success"]):
                is_success = True
                break
            if done:
                break

        success_rates.append(is_success)
        episode_returns.append(total_reward)
        print(f"Eval rollout {rollout_id}: return={total_reward:.2f}, success={is_success}")

    avg_return   = float(np.mean(episode_returns))
    success_rate = float(np.mean(success_rates))
    print(f"Eval summary: success_rate={success_rate:.2f}, avg_return={avg_return:.2f}")

    if wandb_logger is not None:
        wandb_logger.log({"evaluation/success_rate": success_rate}, step=i)
        wandb_logger.log({"evaluation/avg_return":   avg_return},   step=i)

    return success_rate
