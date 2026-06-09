import json
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np
from openpi_client import image_tools
from moviepy.editor import ImageSequenceClip
from moviepy.video.io.ffmpeg_writer import ffmpeg_write_video
from tqdm import tqdm

from jaxrl2.utils.noise_utils import make_full_horizon_noise

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.real_robot_common import (
    HumanEvalUI,
    format_stats,
    extract_observation_train as _extract_observation,
    get_pi0_input_train,
    select_video_frame_train as _select_rollout_video_frame,
    binarize_and_clip_action,
)

EMPTY_IMAGE_SHAPE = (224, 224, 3)


def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       shard_fn=None, policy_service=None, robot_io=None, obs_builder=None,
                                       initial_step=0, initial_total_env_steps=0, initial_total_num_traj=0,
                                       initial_completed=0, initial_successes=0):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    i = initial_step
    total_env_steps = initial_total_env_steps
    total_num_traj = initial_total_num_traj
    wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
    wandb_logger.log({'num_online_trajs': total_num_traj}, step=i)
    wandb_logger.log({'env_steps': total_env_steps}, step=i)

    try:
        ui = HumanEvalUI(
            title="Real Training",
            total_episodes=None,
            preview_names=(("wrist", "Wrist"), ("external", "External")),
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("Failed to initialize real training GUI. Check DISPLAY and Tkinter availability.") from exc

    completed = initial_completed
    successes = initial_successes
    try:
        with tqdm(total=variant.max_steps, initial=i) as pbar:
            while i <= variant.max_steps:
                if not ui.wait_for_start(total_num_traj, completed, successes):
                    print("Training stopped before starting the next trajectory.")
                    break

                traj = collect_traj(
                    variant,
                    agent,
                    env,
                    i,
                    policy_service=policy_service,
                    wandb_logger=wandb_logger,
                    traj_id=total_num_traj,
                    robot_io=robot_io,
                    obs_builder=obs_builder,
                    ui=ui,
                    completed=completed,
                    successes=successes,
                )
                total_num_traj += 1
                completed += 1
                successes += int(traj['is_success'])
                add_online_data_to_buffer(variant, traj, online_replay_buffer)
                total_env_steps += traj['env_steps']
                print('online buffer timesteps length:', len(online_replay_buffer))
                print('online buffer num traj:', total_num_traj)
                print('total env steps:', total_env_steps)

                if traj.get('stop_training'):
                    print("Training stop requested from GUI.")
                    break

                if i == 0:
                    num_gradsteps = 5000
                else:
                    num_gradsteps = len(traj["rewards"]) * variant.multi_grad_step
                print(f'num_gradsteps: {num_gradsteps}')
                if total_num_traj >= variant.num_initial_traj_collect:
                    for _ in range(num_gradsteps):

                        batch = next(replay_buffer_iterator)
                        update_info = agent.update(batch)

                        pbar.update()
                        i += 1

                        if i % variant.log_interval == 0:
                            update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                            for k, v in update_info.items():
                                if v.ndim == 0:
                                    wandb_logger.log({f'training/{k}': v}, step=i)
                                elif v.ndim <= 2:
                                    wandb_logger.log_histogram(f'training/{k}', v, i)
                            wandb_logger.log({
                                'replay_buffer_size': len(online_replay_buffer),
                                'is_success (exploration)': int(traj['is_success']),
                            }, i)

                        if i % variant.eval_interval == 0:
                            wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                            wandb_logger.log({'num_online_trajs': total_num_traj}, step=i)
                            wandb_logger.log({'env_steps': total_env_steps}, step=i)
                            if hasattr(agent, 'perform_eval'):
                                agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                        if variant.checkpoint_interval != -1:
                            if i % variant.checkpoint_interval == 0:
                                agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)
                                # Save training counters so resume can restore them.
                                state_path = os.path.join(variant.outputdir, 'training_state.json')
                                with open(state_path, 'w') as _f:
                                    json.dump({
                                        'i':               i,
                                        'total_env_steps': total_env_steps,
                                        'total_num_traj':  total_num_traj,
                                        'completed':       completed,
                                        'successes':       successes,
                                    }, _f)
                                # Save replay buffer (overwrite; always keep the latest).
                                buf_path = os.path.join(variant.outputdir, 'replay_buffer.pkl')
                                online_replay_buffer.save(buf_path)
    finally:
        ui.close()
            
def add_online_data_to_buffer(variant, traj, online_replay_buffer):
    
    discount_horizon = variant.query_freq
    actions = np.array(traj['actions']) # (T, chunk_size, 14)
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])

    for t in range(episode_len):
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)
        
        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon
        )
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()

def collect_traj(variant, agent, env, i, policy_service=None, wandb_logger=None, traj_id=None, robot_io=None,
                 obs_builder=None, ui=None, completed=0, successes=0):
    if ui is None:
        raise RuntimeError("collect_traj requires a HumanEvalUI instance for real training.")
    query_frequency = variant.query_freq
    instruction = variant.instruction
    runtime_config = robot_io.runtime_config
    max_timesteps = runtime_config.max_timesteps
    agent._rng, rng = jax.random.split(agent._rng)
    is_success = False
    failure_reason = "timeout"
    stop_training = False
    env_steps = 0
    ui.set_resetting(traj_id, completed, successes)
    try:
        env.reset()
    except Exception as e:
        print(f"Environment reset failed")
        import traceback
        traceback.print_exc()
        import pdb; pdb.set_trace()

    # ── Start HighFreqController (aligned with evaluate_pi0_real.py) ──────────
    # Warm-up: triggers joint impedance controller on NUC before HighFreqController.
    try:
        _init_joints = np.array(env.get_observation()["robot_state"]["joint_positions"])
        env._robot.update_joints(_init_joints, velocity=False, blocking=False)
        time.sleep(0.15)
        env._robot.start_trajectory_controller(
            float(getattr(runtime_config, "controller_frequency", 200.0))
        )
        # ── Pre-populate interpolator with a hold-in-place trajectory ──────────
        # Without this, the JointTrajectoryInterpolator is empty on the first
        # add_waypoints call.  When the interpolator is empty, update_waypoints()
        # skips the continuity-bridge logic (curr_pos is None), so a batch of
        # all-stale waypoints is written directly.  The 200 Hz loop then
        # immediately clamps to positions[-1] (the fully-accumulated final target)
        # in a single 5 ms tick → violent first step.
        # Sending a hold trajectory ensures curr_pos is always available, so the
        # bridge fires and the robot transitions smoothly.
        _hold_joints  = np.tile(_init_joints, (4, 1))       # (4, 7)
        _hold_offsets = [0.05, 0.20, 0.50, 1.00]            # seconds, all positive
        _hold_speed   = float(getattr(runtime_config, "max_joint_speed_rad_s", 0.5))
        env._robot.add_waypoints(
            _hold_offsets, _hold_joints.tolist(),
            max_joint_speed_rad_s=_hold_speed,
        )
        time.sleep(0.05)  # let NUC receive and process the hold batch
    except Exception:
        import traceback
        traceback.print_exc()
        import pdb; pdb.set_trace()

    scheduled_gripper_actions: list[tuple[float, float]] = []
    _robot_action_latency   = float(getattr(runtime_config, "robot_action_latency",   0.20))
    _gripper_action_latency = float(getattr(runtime_config, "gripper_action_latency", 0.15))
    _max_joint_speed_rad_s  = float(getattr(runtime_config, "max_joint_speed_rad_s",  0.5))
    _action_scale           = float(getattr(runtime_config, "action_scale", 0.5))
    _MAX_JOINT_DELTA        = 0.2 * _action_scale
    _wrist_obs_latency      = float(getattr(runtime_config, "wrist_camera_obs_latency", 0.084))
    _proprioceptive_latency = float(getattr(runtime_config, "proprioceptive_latency",   0.0003))

    step_time = 1 / runtime_config.control_frequency_hz
    dt_step   = step_time                                   # seconds per control tick
    if query_frequency <= 0:
        raise ValueError(f"query_freq must be positive, got {query_frequency}.")
    if query_frequency > agent.action_chunk_shape[0]:
        raise ValueError(
            f"query_freq ({query_frequency}) must be <= RL noise horizon ({agent.action_chunk_shape[0]})."
        )
    
    rewards = []
    action_list = []
    obs_list = []
    image_list = []

    try:
        ui.set_running(traj_id, completed, successes)
        ui.update_step(traj_id, env_steps, completed, successes)
        action = None
        decision = None
        for t in tqdm(range(max_timesteps)):
            step_started = time.time()
            try:
                _env_obs = env.get_observation()
                state_history = env._robot.get_state_history(n=100)
            except Exception as e:
                print(f"Environment get obs failed")
                import traceback
                traceback.print_exc()
                import pdb; pdb.set_trace()
            curr_obs, t_obs = _extract_observation(
                runtime_config,
                _env_obs,
                state_history=state_history,
                wrist_obs_latency=_wrist_obs_latency,
                proprioceptive_latency=_proprioceptive_latency,
            )
            image_list.append(_select_rollout_video_frame(curr_obs, runtime_config))
            ui.update_camera_previews(
                wrist=curr_obs["wrist_image"],
                external=curr_obs[runtime_config.camera_to_use + "_image"],
            )

            request_data = get_pi0_input_train(curr_obs, runtime_config, instruction)
        
            if t % query_frequency == 0:

                rng, key = jax.random.split(rng)

                obs_dict = build_rl_observation(
                    variant,
                    curr_obs,
                    request_data,
                    policy_service,
                    obs_builder=obs_builder,
                )
                if i == 0:
                    initial_noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                    actions_noise, noise = make_full_horizon_noise(initial_noise[0], agent.action_chunk_shape)
                else:
                    # sac agent predicts the noise for diffusion model
                    actions_noise = agent.sample_actions(obs_dict)
                    actions_noise, noise = make_full_horizon_noise(actions_noise, agent.action_chunk_shape)
                action_list.append(actions_noise)
                obs_list.append(obs_dict)
                # action = pi0 server output; this is the final executable chunk.
                # (actions_noise = RL agent output fed to pi0 as denoising noise)
                action = policy_service.infer(request_data, noise=np.asarray(noise))["actions"]

                # ── Arm: integrate pi0 action chunk → add_waypoints ──────────
                # Integrate from t_obs joint state (aligned with eval).
                _running_joints = curr_obs["joint_position"].copy()
                abs_positions: list[np.ndarray] = []
                for _a in action[:query_frequency]:
                    _vel = np.clip(np.asarray(_a[:-1]), -1.0, 1.0)
                    _running_joints = _running_joints + _vel * _MAX_JOINT_DELTA
                    abs_positions.append(_running_joints.copy())
                abs_positions_arr = np.array(abs_positions)   # (query_frequency, 7)

                arm_target_times = t_obs + np.arange(1, query_frequency + 1) * dt_step

                # ── is_new filter (mirrors evaluate_pi0_real.py) ─────────────
                # Because pi0 inference is synchronous here (~200–400 ms), t_obs
                # is already far in the past by the time we reach this point.
                # Without filtering, ALL waypoints have negative time offsets and
                # the HighFreqController's JointTrajectoryInterpolator clamps to
                # positions[-1] in a single 5 ms tick → violent first step.
                # We integrate the full chunk first (so cumulative positions are
                # correct), then discard waypoints whose wall-clock target has
                # already passed before scheduling them on the controller.
                _action_exec_latency = float(
                    getattr(runtime_config, "action_exec_latency", 0.0)
                )
                _is_new = arm_target_times > (time.time() + _action_exec_latency)
                if np.any(_is_new):
                    arm_offsets = (
                        arm_target_times[_is_new] - _robot_action_latency
                    ) - time.time()
                    try:
                        env._robot.add_waypoints(
                            arm_offsets.tolist(),
                            abs_positions_arr[_is_new].tolist(),
                            max_joint_speed_rad_s=_max_joint_speed_rad_s,
                        )
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        import pdb; pdb.set_trace()
                # If all stale: hold-in-place trajectory (sent at episode start)
                # keeps the robot stationary until a fresh chunk arrives.

                # ── Gripper: schedule timestamped commands ────────────────────
                scheduled_gripper_actions = [
                    (float(ts), float(binarize_and_clip_action(np.asarray(_a))[-1]))
                    for ts, _a in zip(arm_target_times.tolist(), action[:query_frequency])
                ]

            decision = ui.poll()
            if decision is not None:
                break
            if action is None:
                raise RuntimeError("No action chunk available. This should be impossible at t=0.")

            # ── Execute due gripper commands (arm handled by HighFreqController) ──
            curr_time_now = time.time()
            gripper_to_exec = None
            while (scheduled_gripper_actions and
                   scheduled_gripper_actions[0][0] - _gripper_action_latency <= curr_time_now):
                _, gripper_to_exec = scheduled_gripper_actions.pop(0)
            if gripper_to_exec is not None:
                try:
                    env._robot.update_gripper(gripper_to_exec, velocity=False, blocking=False)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    import pdb; pdb.set_trace()

            env_steps = t + 1
            ui.update_step(traj_id, env_steps, completed, successes)
            decision = ui.poll()
            if decision is not None:
                break

            elapsed = time.time() - step_started
            if elapsed < step_time:
                time.sleep(step_time - elapsed)

        if decision is None:
            decision = (False, "timeout")

        is_success, failure_reason = decision
        if failure_reason == "user_quit":
            stop_training = True
        if is_success:
            failure_reason = ""
            print("Trial marked as SUCCESS.")
        else:
            print(f"Trial marked as FAILURE ({failure_reason}).")

        try:
            _env_obs = env.get_observation()
        except Exception as e:
            print(f"Environment get obs failed")
            import traceback
            traceback.print_exc()
            import pdb; pdb.set_trace()
        
        # add last observation
        curr_obs, _ = _extract_observation(
                    runtime_config,
                    _env_obs,
            )
        image_list.append(_select_rollout_video_frame(curr_obs, runtime_config))
        request_data = get_pi0_input_train(curr_obs, runtime_config, instruction)
        obs_dict = build_rl_observation(
            variant,
            curr_obs,
            request_data,
            policy_service,
            obs_builder=obs_builder,
        )
        obs_list.append(obs_dict)
        print(f'Rollout Done')
        
    finally:
        # ── Stop HighFreqController before reset ──────────────────────────────
        try:
            env._robot.stop_trajectory_controller()
        except Exception:
            pass

        query_steps = len(action_list)
        if query_steps == 0:
            rewards = np.array([], dtype=np.float32)
            masks = np.array([], dtype=np.float32)
        elif is_success:
            rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
            masks = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            rewards = -np.ones(query_steps)
            masks = np.ones(query_steps)
            
        if wandb_logger is not None:
            wandb_logger.log({f'is_success': int(is_success)}, step=i)
            wandb_logger.log({f'total_num_traj': traj_id}, step=i)
            wandb_logger.log({f'rollout_result/{failure_reason or "success"}': 1}, step=i)

        if image_list:
            video_path = os.path.join(variant.outputdir, f'video_high_{traj_id}.mp4')
            fps = 15
            video = np.stack(image_list)
            clip = ImageSequenceClip(list(video), fps=fps)
            ffmpeg_write_video(
                clip,
                video_path,
                fps,
                codec="libx264",
                audiofile=None,
                logger=None,
            )
        else:
            print("No rollout frames captured; skipping video save.")
       
        print("Episode Done! Resetting the environment.")
        ui.set_resetting(traj_id, completed, successes)
        try:
            env.reset()
        except Exception as e:
            print(f"Environment reset failed")
            import traceback
            traceback.print_exc()  # This prints the full traceback
            import pdb; pdb.set_trace()
    
    traj = {
        'observations': obs_list,
        'actions': action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'failure_reason': failure_reason,
        'stop_training': stop_training,
        'env_steps': env_steps,
    }
    
    return traj


def build_rl_observation(variant, curr_obs, request_data, policy_service, obs_builder=None):
    if obs_builder is not None:
        return obs_builder.build(curr_obs, request_data, policy_service)

    img_all = process_images(variant, curr_obs)

    # extract the feature from the pi0 VLM backbone and concat with the qpos as states
    img_rep_pi0, _ = policy_service.get_prefix_rep(request_data)
    img_rep_pi0 = img_rep_pi0[:, -1, :] # (1, 2048)
    qpos = np.concatenate([curr_obs["joint_position"], curr_obs["gripper_position"], img_rep_pi0.flatten()])

    return {
        'pixels': img_all,
        'state': qpos[np.newaxis, ..., np.newaxis],
    }
    

def process_images(variant, obs):
    '''
    concat the images from all cameras
    '''
    im1 = image_tools.resize_with_pad(obs["left_image"], variant.resize_image, variant.resize_image)
    im2 = image_tools.resize_with_pad(obs["right_image"], variant.resize_image, variant.resize_image)
    im3 = image_tools.resize_with_pad(obs["wrist_image"], variant.resize_image, variant.resize_image)
    img_all = np.concatenate([im1, im2, im3], axis=2)[np.newaxis, ..., np.newaxis]
    return img_all
