import os
import time

import jax
import numpy as np
from openpi_client import image_tools
from moviepy.editor import ImageSequenceClip
from moviepy.video.io.ffmpeg_writer import ffmpeg_write_video
from tqdm import tqdm

from jaxrl2.utils.noise_utils import make_full_horizon_noise

EMPTY_IMAGE_SHAPE = (224, 224, 3)


class HumanEvalUI:
    PREVIEW_SIZE = (360, 270)
    BUTTON_FONT = ("Arial", 14, "bold")
    BUTTON_WIDTH = 12

    def __init__(self):
        try:
            import tkinter as tk
        except ImportError as exc:
            raise RuntimeError("Real training GUI requires tkinter to be installed.") from exc

        self._tk = tk
        self.root = tk.Tk()
        self.root.title("Real Training")
        self.root.geometry("820x610")
        self.root.protocol("WM_DELETE_WINDOW", self.request_quit)

        self.start_requested = False
        self.quit_requested = False
        self.running = False
        self.decision = None
        self.closed = False
        self._preview_photos = {}

        self.status_var = tk.StringVar(value="Waiting to start.")
        self.stats_var = tk.StringVar(value="")

        title = tk.Label(self.root, text="Real Training", font=("Arial", 16, "bold"))
        title.pack(pady=(14, 4))

        status = tk.Label(self.root, textvariable=self.status_var, font=("Arial", 11))
        status.pack(pady=4)

        stats = tk.Label(self.root, textvariable=self.stats_var, font=("Arial", 10))
        stats.pack(pady=2)

        preview_container = tk.Frame(self.root)
        preview_container.pack(pady=(10, 8))
        self.preview_labels = {}
        for col, (name, title_text) in enumerate((("wrist", "Wrist"), ("external", "External"))):
            column_frame = tk.Frame(preview_container)
            column_frame.grid(row=0, column=col, padx=8)
            tk.Label(column_frame, text=title_text, font=("Arial", 10, "bold")).pack(pady=(0, 4))
            preview_frame = tk.Frame(
                column_frame,
                width=self.PREVIEW_SIZE[0],
                height=self.PREVIEW_SIZE[1],
                bg="black",
            )
            preview_frame.pack()
            preview_frame.pack_propagate(False)
            preview_label = tk.Label(preview_frame, bg="black", bd=0)
            preview_label.pack(expand=True)
            self.preview_labels[name] = preview_label

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=14)

        self.start_button = tk.Button(
            button_frame,
            text="Start next",
            width=self.BUTTON_WIDTH,
            font=self.BUTTON_FONT,
            command=self.request_start,
        )
        self.start_button.grid(row=0, column=0, padx=6)

        self.success_button = tk.Button(
            button_frame,
            text="Success",
            width=self.BUTTON_WIDTH,
            font=self.BUTTON_FONT,
            command=self.mark_success,
        )
        self.success_button.grid(row=0, column=1, padx=6)

        self.failure_button = tk.Button(
            button_frame,
            text="Failure",
            width=self.BUTTON_WIDTH,
            font=self.BUTTON_FONT,
            command=self.mark_failure,
        )
        self.failure_button.grid(row=0, column=2, padx=6)

        self.quit_button = tk.Button(
            self.root,
            text="Quit",
            width=self.BUTTON_WIDTH,
            font=self.BUTTON_FONT,
            command=self.request_quit,
        )
        self.quit_button.pack(pady=(2, 10))

        self.set_idle(traj_id=0, completed=0, successes=0)
        self.update()

    def request_start(self):
        if not self.running:
            self.start_requested = True

    def request_quit(self):
        self.quit_requested = True
        if not self.closed:
            self.status_var.set("Quit requested. Finishing current step...")

    def mark_success(self):
        if self.running and self.decision is None:
            self.decision = (True, "")
            self.status_var.set("Success marked. Stopping rollout...")
            self._set_decision_buttons("disabled")

    def mark_failure(self):
        if self.running and self.decision is None:
            self.decision = (False, "human_failure")
            self.status_var.set("Failure marked. Stopping rollout...")
            self._set_decision_buttons("disabled")

    def set_idle(self, traj_id, completed, successes):
        if self.closed:
            return
        self.running = False
        self.start_requested = False
        self.decision = None
        self.status_var.set(f"Trajectory {traj_id + 1}: waiting for Start next.")
        self.stats_var.set(_format_training_stats(completed, successes))
        self.start_button.config(state="normal")
        self._set_decision_buttons("disabled")
        self.update()

    def set_running(self, traj_id, step, completed, successes):
        if self.closed:
            return
        self.running = True
        self.start_requested = False
        if step == 0:
            self.decision = None
        self.status_var.set(f"Trajectory {traj_id + 1}: running step {step}.")
        self.stats_var.set(_format_training_stats(completed, successes))
        self.start_button.config(state="disabled")
        self._set_decision_buttons("normal")
        self.update()

    def set_resetting(self, traj_id, completed, successes):
        if self.closed:
            return
        self.running = False
        self.status_var.set(f"Trajectory {traj_id + 1}: resetting robot.")
        self.stats_var.set(_format_training_stats(completed, successes))
        self.start_button.config(state="disabled")
        self._set_decision_buttons("disabled")
        self.update()

    def wait_for_start(self, traj_id, completed, successes):
        self.set_idle(traj_id, completed, successes)
        while not self.start_requested and not self.quit_requested:
            self.update()
            time.sleep(0.05)
        return self.start_requested and not self.quit_requested

    def poll(self):
        self.update()
        if self.quit_requested:
            return False, "user_quit"
        return self.decision

    def update_camera_previews(self, wrist_image=None, external_image=None):
        if self.closed:
            return
        try:
            if wrist_image is not None:
                self._set_preview_image("wrist", wrist_image)
            if external_image is not None:
                self._set_preview_image("external", external_image)
            self.update()
        except self._tk.TclError:
            self.closed = True
            self.quit_requested = True

    def _set_preview_image(self, preview_name, image):
        from PIL import Image, ImageTk

        preview_label = self.preview_labels.get(preview_name)
        if preview_label is None:
            return

        frame = np.asarray(image)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        if frame.ndim != 3:
            return
        if frame.shape[2] > 3:
            frame = frame[..., :3]
        if frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        if frame.shape[2] != 3:
            return

        if frame.dtype != np.uint8:
            frame = np.nan_to_num(frame)
            if np.issubdtype(frame.dtype, np.floating) and frame.size and float(frame.max()) <= 1.0:
                frame = frame * 255
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        pil_image = Image.fromarray(np.ascontiguousarray(frame))
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        pil_image.thumbnail(self.PREVIEW_SIZE, resampling)

        canvas = Image.new("RGB", self.PREVIEW_SIZE, "black")
        offset = (
            (self.PREVIEW_SIZE[0] - pil_image.width) // 2,
            (self.PREVIEW_SIZE[1] - pil_image.height) // 2,
        )
        canvas.paste(pil_image, offset)

        photo = ImageTk.PhotoImage(canvas)
        preview_label.config(image=photo)
        self._preview_photos[preview_name] = photo

    def update(self):
        if self.closed:
            return
        try:
            self.root.update()
        except self._tk.TclError:
            self.closed = True
            self.quit_requested = True

    def close(self):
        if self.closed:
            return
        try:
            self.root.destroy()
        except self._tk.TclError:
            pass
        self.closed = True

    def _set_decision_buttons(self, state):
        self.success_button.config(state=state)
        self.failure_button.config(state=state)


def _format_training_stats(completed, successes):
    if completed == 0:
        return "Completed: 0 | Success rate: n/a"
    return f"Completed: {completed} | Successes: {successes} | Success rate: {successes / completed:.3f}"


def _select_rollout_video_frame(obs, robot_config):
    if getattr(robot_config, "use_wrist_camera", True) and obs.get("wrist_image_present", True):
        return obs["wrist_image"]

    external_image_key = robot_config.camera_to_use + "_image"
    external_present_key = robot_config.camera_to_use + "_image_present"
    if getattr(robot_config, "use_exterior_camera", True) and obs.get(external_present_key, True):
        return obs[external_image_key]

    if obs.get("wrist_image_present", False):
        return obs["wrist_image"]
    if obs.get(external_present_key, False):
        return obs[external_image_key]
    return obs["wrist_image"]


def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       shard_fn=None, policy_service=None, robot_io=None, obs_builder=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)
        
    i = 0
    total_env_steps = 0
    total_num_traj = 0
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)

    try:
        ui = HumanEvalUI()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("Failed to initialize real training GUI. Check DISPLAY and Tkinter availability.") from exc

    completed = 0
    successes = 0
    try:
        with tqdm(total=variant.max_steps, initial=0) as pbar:
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
    step_time = 1 / runtime_config.control_frequency_hz
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
        ui.set_running(traj_id, env_steps, completed, successes)
        action = None
        decision = None
        for t in tqdm(range(max_timesteps)):
            step_started = time.time()
            try:
                _env_obs = env.get_observation()
            except Exception as e:
                print(f"Environment get obs failed")
                import traceback
                traceback.print_exc()
                import pdb; pdb.set_trace()
            curr_obs = _extract_observation(
                    runtime_config,
                    _env_obs,
            )
            image_list.append(_select_rollout_video_frame(curr_obs, runtime_config))
            ui.update_camera_previews(
                wrist_image=curr_obs["wrist_image"],
                external_image=curr_obs[runtime_config.camera_to_use + "_image"],
            )

            request_data = get_pi0_input(curr_obs, runtime_config, instruction)
        
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
                action = policy_service.infer(request_data, noise=np.asarray(noise))["actions"]

            decision = ui.poll()
            if decision is not None:
                break
            if action is None:
                raise RuntimeError("No action chunk available. This should be impossible at t=0.")

            action_t = action[t % query_frequency]
            
            # binarize gripper action.
            if action_t[-1].item() > 0.5:
                action_t = np.concatenate([action_t[:-1], np.ones((1,))])
            else:
                action_t = np.concatenate([action_t[:-1], np.zeros((1,))])
            action_t = np.clip(action_t, -1, 1)
            
            try:
                env.step(action_t)
            except Exception as e:
                print(f"Environment step failed")
                import traceback
                traceback.print_exc()  # This prints the full traceback
                import pdb; pdb.set_trace()

            env_steps = t + 1
            ui.set_running(traj_id, env_steps, completed, successes)
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
        curr_obs = _extract_observation(
                    runtime_config,
                    _env_obs,
            )
        image_list.append(_select_rollout_video_frame(curr_obs, runtime_config))
        request_data = get_pi0_input(curr_obs, runtime_config, instruction)
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


def _extract_observation(robot_config, obs_dict):
    '''
    from https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py
    '''
    image_observations = obs_dict["image"]
    left_image = _find_camera_image(image_observations, robot_config.left_camera_id)
    right_image = _find_camera_image(image_observations, robot_config.right_camera_id)
    wrist_image = _find_camera_image(image_observations, robot_config.wrist_camera_id)

    missing = [
        name
        for name, image in (
            ("left_image", left_image),
            ("right_image", right_image),
            ("wrist_image", wrist_image),
        )
        if image is None
    ]
    allow_missing_cameras = getattr(robot_config, "allow_missing_cameras", False)
    if missing and not allow_missing_cameras:
        raise RuntimeError(
            "Missing DROID camera images: "
            + ", ".join(missing)
            + f". Available image keys: {sorted(image_observations.keys())}. "
            "Check LEFT_CAMERA_ID, RIGHT_CAMERA_ID, and WRIST_CAMERA_ID in examples/scripts/run_real.sh."
        )
    if wrist_image is None and getattr(robot_config, "require_wrist_camera", False):
        raise RuntimeError(
            "Missing DROID wrist camera image required for Wrist-DINO real training. "
            f"Available image keys: {sorted(image_observations.keys())}. "
            "Check WRIST_CAMERA_ID in examples/scripts/run_real_dino.sh."
        )

    left_image_present = left_image is not None
    right_image_present = right_image is not None
    wrist_image_present = wrist_image is not None

    left_image = _to_rgb_image(left_image) if left_image_present else None
    right_image = _to_rgb_image(right_image) if right_image_present else None
    wrist_image = _to_rgb_image(wrist_image) if wrist_image_present else None
    empty_image = _empty_rgb_image_like(left_image, right_image, wrist_image)
    if left_image is None:
        left_image = empty_image.copy()
    if right_image is None:
        right_image = empty_image.copy()
    if wrist_image is None:
        wrist_image = empty_image.copy()

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "left_image_present": left_image_present,
        "right_image_present": right_image_present,
        "wrist_image_present": wrist_image_present,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


def _to_rgb_image(image):
    image = np.asarray(image)[..., :3]
    return image[..., ::-1]


def _empty_rgb_image_like(*images):
    for image in images:
        if image is not None:
            return np.zeros_like(image)
    return np.zeros(EMPTY_IMAGE_SHAPE, dtype=np.uint8)


def _find_camera_image(image_observations, camera_id):
    if not camera_id:
        return None
    candidates = _camera_id_candidates(camera_id)

    matches = []
    for key, image in image_observations.items():
        if any(key == candidate or key.startswith(f"{candidate}_") for candidate in candidates):
            matches.append((key, image))

    if not matches:
        return None

    left_view = [image for key, image in matches if key.endswith("_left") or "left" in key]
    if left_view:
        return left_view[0]

    return matches[0][1]


def _camera_id_candidates(camera_id: str) -> set[str]:
    camera_id = str(camera_id)
    prefixes = ("realsense_", "zedmini_", "zed_mini_", "zed_")
    serial = camera_id
    for prefix in prefixes:
        if serial.startswith(prefix):
            serial = serial.removeprefix(prefix)
            break
    candidates = {camera_id, serial}
    candidates.update(f"{prefix}{serial}" for prefix in prefixes)
    return candidates


def get_pi0_input(obs, robot_config, instruction):
    external_camera = robot_config.camera_to_use
    external_image_key = external_camera + "_image"
    external_present_key = external_camera + "_image_present"
    request_data = {
        "observation/joint_position": obs["joint_position"],
        "observation/gripper_position": obs["gripper_position"],
        "prompt": instruction,
    }
    if (
        getattr(robot_config, "use_exterior_camera", True)
        and obs.get(external_present_key, True)
        and external_image_key in obs
    ):
        request_data["observation/exterior_image_1_left"] = image_tools.resize_with_pad(
            obs[external_image_key], 224, 224
        )
    if (
        getattr(robot_config, "use_wrist_camera", True)
        and obs.get("wrist_image_present", True)
        and "wrist_image" in obs
    ):
        request_data["observation/wrist_image_left"] = image_tools.resize_with_pad(obs["wrist_image"], 224, 224)
    return request_data


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
