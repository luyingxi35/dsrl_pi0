#!/usr/bin/env python3
"""Standalone pi0-only real-world evaluation with wrist-camera observations."""

import argparse
import csv
import dataclasses
import datetime as dt
import logging
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DROID_CONTROL_FREQUENCY = 15
VIDEO_FPS = 15
DEFAULT_WRIST_CAMERA_ID = "17396664"
DEFAULT_EXTERIOR_CAMERA_ID = "241122302552"

RESULT_FIELDS = [
    "episode_id",
    "success",
    "failure_reason",
    "env_steps",
    "duration_s",
    "video_path",
    "timestamp",
    "instruction",
    "use_wrist_camera",
    "use_exterior_camera",
    "wrist_camera_id",
    "exterior_camera_id",
    "policy_host",
    "policy_port",
]


@dataclasses.dataclass(frozen=True)
class PolicyServerConfig:
    host: str
    port: int


@dataclasses.dataclass(frozen=True)
class RobotRuntimeConfig:
    max_timesteps: int
    wrist_camera_id: str | None = None
    exterior_camera_id: str | None = None
    control_frequency_hz: int = DEFAULT_DROID_CONTROL_FREQUENCY

    def validate(self) -> None:
        if not self.wrist_camera_id and not self.exterior_camera_id:
            raise ValueError("At least one of --use_wrist_camera or --use_exterior_camera must be enabled.")
        if self.wrist_camera_id and self.exterior_camera_id and self.exterior_camera_id == self.wrist_camera_id:
            raise ValueError("Hardcoded wrist and exterior camera IDs must be different.")


@dataclasses.dataclass(frozen=True)
class ExecutionConfig:
    """Timing configuration for non-blocking timestamped action scheduling."""

    # Number of actions to execute before re-inferring (replaces query_freq).
    execution_steps: int = 6
    # Time (s) from env.step() call to robot physically responding.
    # The scheduler advances each action's call time by this amount so the
    # robot reaches the target pose at the intended moment.
    robot_action_latency: float = 0.1
    # Minimum lead time (s) required to schedule an action.
    # Actions whose target_time <= curr_time + action_exec_latency are
    # considered stale and skipped (or handled via fallback).
    action_exec_latency: float = 0.01


@dataclasses.dataclass
class RolloutResult:
    episode_id: int
    success: bool
    failure_reason: str
    env_steps: int
    duration_s: float
    video_path: str
    timestamp: str


class PolicyService:
    def __init__(self, config: PolicyServerConfig):
        from openpi_client import websocket_client_policy

        self._client = websocket_client_policy.WebsocketClientPolicy(
            host=config.host,
            port=config.port,
        )

    def preflight(self):
        metadata = self._client.get_server_metadata()
        logging.info("OpenPI policy server metadata: %s", metadata)
        return metadata

    def infer(self, obs):
        return self._client.infer(obs)


class RobotIO:
    def __init__(self, runtime_config: RobotRuntimeConfig):
        from droid.robot_env import RobotEnv

        self._runtime_config = runtime_config
        self._env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")

    @property
    def env(self):
        return self._env

    @property
    def runtime_config(self) -> RobotRuntimeConfig:
        return self._runtime_config

    def preflight(self):
        obs = self._env.get_observation()
        missing = []
        if (
            self._runtime_config.wrist_camera_id
            and _find_camera_image(obs["image"], self._runtime_config.wrist_camera_id) is None
        ):
            missing.append("wrist")
        if (
            self._runtime_config.exterior_camera_id
            and _find_camera_image(obs["image"], self._runtime_config.exterior_camera_id) is None
        ):
            missing.append("exterior")
        if missing:
            raise RuntimeError(
                "DROID camera preflight failed. Missing image feeds for: "
                + ", ".join(missing)
                + f". Available image keys: {sorted(obs['image'].keys())}."
            )
        return obs


class HumanEvalUI:
    PREVIEW_SIZE = (360, 270)
    BUTTON_FONT = ("Arial", 14, "bold")
    BUTTON_WIDTH = 12

    def __init__(self, total_episodes: int):
        import tkinter as tk

        self._tk = tk
        self.root = tk.Tk()
        self.root.title("Pi0 Real Evaluation")
        self.root.geometry("820x610")
        self.root.protocol("WM_DELETE_WINDOW", self.request_quit)

        self.total_episodes = total_episodes
        self.start_requested = False
        self.quit_requested = False
        self.running = False
        self.decision: tuple[bool, str] | None = None
        self.closed = False
        self._preview_photos = {}

        self.status_var = tk.StringVar(value="Waiting to start.")
        self.stats_var = tk.StringVar(value="")

        title = tk.Label(self.root, text="Pi0 Real Evaluation", font=("Arial", 16, "bold"))
        title.pack(pady=(14, 4))

        status = tk.Label(self.root, textvariable=self.status_var, font=("Arial", 11))
        status.pack(pady=4)

        stats = tk.Label(self.root, textvariable=self.stats_var, font=("Arial", 10))
        stats.pack(pady=2)

        preview_container = tk.Frame(self.root)
        preview_container.pack(pady=(10, 8))
        self.preview_labels = {}
        for col, (name, title_text) in enumerate((("wrist", "Wrist"), ("exterior", "Exterior"))):
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

        self.set_idle(episode_id=0, completed=0, successes=0)
        self.update()

    def request_start(self) -> None:
        if not self.running:
            self.start_requested = True

    def request_quit(self) -> None:
        self.quit_requested = True
        if not self.closed:
            self.status_var.set("Quit requested. Finishing current step...")

    def mark_success(self) -> None:
        if self.running and self.decision is None:
            self.decision = (True, "")
            self.status_var.set("Success marked. Stopping rollout...")
            self._set_decision_buttons("disabled")

    def mark_failure(self) -> None:
        if self.running and self.decision is None:
            self.decision = (False, "human_failure")
            self.status_var.set("Failure marked. Stopping rollout...")
            self._set_decision_buttons("disabled")

    def set_idle(self, episode_id: int, completed: int, successes: int) -> None:
        if self.closed:
            return
        self.running = False
        self.start_requested = False
        self.decision = None
        self.status_var.set(f"Episode {episode_id + 1}/{self.total_episodes}: waiting for Start next.")
        self.stats_var.set(_format_stats(completed, successes))
        self.start_button.config(state="normal")
        self._set_decision_buttons("disabled")

    def set_running(self, episode_id: int, completed: int, successes: int) -> None:
        if self.closed:
            return
        self.running = True
        self.start_requested = False
        self.decision = None
        self.status_var.set(f"Episode {episode_id + 1}/{self.total_episodes}: running.")
        self.stats_var.set(_format_stats(completed, successes))
        self.start_button.config(state="disabled")
        self._set_decision_buttons("normal")

    def set_resetting(self, episode_id: int, completed: int, successes: int) -> None:
        if self.closed:
            return
        self.running = False
        self.status_var.set(f"Episode {episode_id + 1}/{self.total_episodes}: resetting robot.")
        self.stats_var.set(_format_stats(completed, successes))
        self.start_button.config(state="disabled")
        self._set_decision_buttons("disabled")
        self.update()

    def wait_for_start(self, episode_id: int, completed: int, successes: int) -> bool:
        self.set_idle(episode_id, completed, successes)
        while not self.start_requested and not self.quit_requested:
            self.update()
            time.sleep(0.05)
        return self.start_requested and not self.quit_requested

    def poll(self) -> tuple[bool, str] | None:
        self.update()
        if self.quit_requested:
            return False, "user_quit"
        return self.decision

    def update_step(self, episode_id: int, step: int, completed: int, successes: int) -> None:
        if self.closed:
            return
        self.status_var.set(f"Episode {episode_id + 1}/{self.total_episodes}: running step {step}.")
        self.stats_var.set(_format_stats(completed, successes))
        self.update()

    def update_camera_previews(self, wrist_image: Any | None = None, exterior_image: Any | None = None) -> None:
        if self.closed:
            return

        try:
            if wrist_image is not None:
                self._set_preview_image("wrist", wrist_image)
            if exterior_image is not None:
                self._set_preview_image("exterior", exterior_image)
            self.update()
        except self._tk.TclError:
            self.closed = True
            self.quit_requested = True

    def _set_preview_image(self, preview_name: str, image: Any) -> None:
        import numpy as np
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

    def update(self) -> None:
        if self.closed:
            return
        try:
            self.root.update()
        except self._tk.TclError:
            self.closed = True
            self.quit_requested = True

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.root.destroy()
        except self._tk.TclError:
            pass
        self.closed = True

    def _set_decision_buttons(self, state: str) -> None:
        self.success_button.config(state=state)
        self.failure_button.config(state=state)


def _format_stats(completed: int, successes: int) -> str:
    if completed == 0:
        return "Completed: 0 | Success rate: n/a"
    return f"Completed: {completed} | Successes: {successes} | Success rate: {successes / completed:.3f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate pi0-only policy on a real DROID robot using selected cameras.")
    parser.add_argument("--instruction", default="put the spoon on the plate", help="Language instruction for pi0.")
    parser.add_argument("--eval_episodes", default=10, type=int, help="Number of real-world episodes to evaluate.")
    parser.add_argument("--max_rollout_steps", default=200, type=int, help="Max robot-control steps per episode.")
    parser.add_argument(
        "--execution_steps",
        default=6,
        type=int,
        help=(
            "Number of (non-stale) actions from each inference chunk to schedule "
            "before re-inferring. Replaces --query_freq. Default: 6."
        ),
    )
    parser.add_argument(
        "--robot_action_latency",
        default=0.1,
        type=float,
        help=(
            "Seconds between env.step() call and the robot physically responding. "
            "Each action's scheduled call time is advanced by this amount so the "
            "robot reaches the target state at the intended moment. Default: 0.1."
        ),
    )
    parser.add_argument(
        "--action_exec_latency",
        default=0.01,
        type=float,
        help=(
            "Minimum lead time (s) required to schedule an action. "
            "Actions whose target_time <= curr_time + action_exec_latency are "
            "considered stale and skipped. Default: 0.01."
        ),
    )
    parser.add_argument("--control_frequency_hz", default=15, type=int, help="Target DROID control frequency.")
    parser.add_argument(
        "--use_wrist_camera",
        default=1,
        type=int,
        choices=(0, 1),
        help="Whether to use the hardcoded Zed Mini wrist camera.",
    )
    parser.add_argument(
        "--use_exterior_camera",
        default=0,
        type=int,
        choices=(0, 1),
        help="Whether to use the hardcoded RealSense exterior camera.",
    )
    parser.add_argument("--policy_host", default="127.0.0.1", help="OpenPI policy server host.")
    parser.add_argument("--policy_port", default=8000, type=int, help="OpenPI policy server port.")
    parser.add_argument("--outputdir", default=None, help="Directory for eval_results.csv and rollout videos.")
    return parser


def resolve_outputdir(outputdir: str | None) -> Path:
    if outputdir:
        path = Path(outputdir)
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("logs") / f"pi0_eval_real_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_runtime(args: argparse.Namespace) -> tuple[PolicyService, RobotIO]:
    policy_service = PolicyService(PolicyServerConfig(host=args.policy_host, port=args.policy_port))
    policy_service.preflight()

    runtime_config = RobotRuntimeConfig(
        max_timesteps=args.max_rollout_steps,
        wrist_camera_id=DEFAULT_WRIST_CAMERA_ID if args.use_wrist_camera else None,
        exterior_camera_id=DEFAULT_EXTERIOR_CAMERA_ID if args.use_exterior_camera else None,
        control_frequency_hz=args.control_frequency_hz,
    )
    runtime_config.validate()

    robot_io = RobotIO(runtime_config)
    robot_io.preflight()
    return policy_service, robot_io


def reset_robot(env: Any, reason: str) -> None:
    logging.info("Resetting DROID environment (%s)...", reason)
    try:
        env.reset()
    except Exception:
        logging.exception("Environment reset failed (%s).", reason)
        raise


def run_rollout(
    args: argparse.Namespace,
    env: Any,
    policy_service: PolicyService,
    robot_io: RobotIO,
    ui: HumanEvalUI,
    episode_id: int,
    completed: int,
    successes: int,
    outputdir: Path,
    exec_config: ExecutionConfig | None = None,
) -> RolloutResult:
    from tqdm import tqdm

    if exec_config is None:
        exec_config = ExecutionConfig()

    runtime_config = robot_io.runtime_config
    dt_step = 1.0 / runtime_config.control_frequency_hz

    # ── Inference concurrency ──────────────────────────────────────────────
    # env.step() uses zerorpc/gevent and MUST stay on the main thread.
    # policy_service.infer() uses openpi websocket (not gevent) and is safe
    # to run in a background thread, keeping the control loop non-blocking.
    #
    # Background thread puts (actions_array, obs_timestamp) here when done.
    # Unbounded; main thread drains it each tick, keeping only the latest.
    inference_queue: queue.Queue = queue.Queue()
    inference_in_progress = threading.Event()

    def _run_inference(obs_snapshot: dict, t_obs: float) -> None:
        try:
            request_data = get_pi0_input(obs_snapshot, args.instruction)
            response = policy_service.infer(request_data)
            inference_queue.put((np.asarray(response["actions"]), t_obs))
        except Exception:
            logging.exception("run_rollout: inference failed in background thread")
        finally:
            inference_in_progress.clear()

    # ── Scheduled action buffer ────────────────────────────────────────────
    # Sorted list of (target_time, action); main thread pops and executes.
    scheduled_actions: list[tuple[float, np.ndarray]] = []

    # ── Episode bookkeeping ────────────────────────────────────────────────
    ui.set_running(episode_id, completed, successes)
    start_time = time.time()
    t_loop_start = start_time
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    image_list: list[np.ndarray] = []
    decision: tuple[bool, str] | None = None
    env_steps = 0

    for t in tqdm(range(runtime_config.max_timesteps), desc=f"pi0 eval episode {episode_id}"):
        decision = ui.poll()
        if decision is not None:
            break

        t_step_end = t_loop_start + (t + 1) * dt_step

        # ── 1. Get observation (main thread, gevent-safe) ──────────────────
        curr_obs = extract_camera_observation(runtime_config, env.get_observation())
        obs_timestamp = time.time()

        # ── 2. Drain inference queue; keep only the most recent result ─────
        latest_result = None
        while not inference_queue.empty():
            try:
                latest_result = inference_queue.get_nowait()
            except queue.Empty:
                break

        if latest_result is not None:
            new_actions, t_obs = latest_result
            action_timestamps = t_obs + np.arange(len(new_actions)) * dt_step
            curr_time = time.time()
            is_new = action_timestamps > (curr_time + exec_config.action_exec_latency)
            if np.any(is_new):
                new_a = new_actions[is_new][: exec_config.execution_steps]
                new_t = action_timestamps[is_new][: exec_config.execution_steps]
                scheduled_actions = [
                    (ts, binarize_and_clip_action(a))
                    for ts, a in zip(new_t, new_a)
                ]
            else:
                logging.warning(
                    "run_rollout: all actions stale at t=%d "
                    "(inference latency exceeded obs horizon)", t
                )

        # ── 3. Trigger inference in background thread if needed ────────────
        # Fires on the regular execution_steps cadence, or immediately when
        # scheduled_actions is exhausted (inference ran longer than expected).
        should_infer = not inference_in_progress.is_set() and (
            t % exec_config.execution_steps == 0
            or not scheduled_actions
        )
        if should_infer:
            inference_in_progress.set()
            threading.Thread(
                target=_run_inference,
                args=(curr_obs, obs_timestamp),
                daemon=True,
                name=f"Infer-t{t}",
            ).start()

        # ── 4. Execute scheduled action (main thread, gevent-safe) ─────────
        # Pop all actions whose adjusted call time
        # (= target_time - robot_action_latency) has already passed.
        # If multiple are due, execute the most recent one.
        curr_time = time.time()
        action_to_exec = None
        while (
            scheduled_actions
            and scheduled_actions[0][0] - exec_config.robot_action_latency <= curr_time
        ):
            _, action_to_exec = scheduled_actions.pop(0)

        if action_to_exec is not None:
            env.step(action_to_exec)

        # ── 5. UI & bookkeeping ────────────────────────────────────────────
        ui.update_camera_previews(
            wrist_image=curr_obs.get("wrist_image"),
            exterior_image=curr_obs.get("exterior_image"),
        )
        image_list.append(_select_video_frame(curr_obs))
        env_steps = t + 1
        ui.update_step(episode_id, env_steps, completed, successes)

        decision = ui.poll()
        if decision is not None:
            break

        # ── 6. Wait until tick deadline ────────────────────────────────────
        sleep_s = t_step_end - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)

    # Allow any in-flight inference to finish before returning.
    inference_in_progress.wait(timeout=5.0)

    if decision is None:
        decision = (False, "timeout")

    success, failure_reason = decision
    duration_s = time.time() - start_time
    video_path = save_rollout_video(outputdir, episode_id, image_list)

    return RolloutResult(
        episode_id=episode_id,
        success=bool(success),
        failure_reason="" if success else failure_reason,
        env_steps=env_steps,
        duration_s=duration_s,
        video_path=video_path,
        timestamp=timestamp,
    )


def extract_camera_observation(robot_config: RobotRuntimeConfig, obs_dict: dict[str, Any]) -> dict[str, np.ndarray]:
    image_observations = obs_dict["image"]
    wrist_image = None
    if robot_config.wrist_camera_id:
        wrist_image = _find_camera_image(image_observations, robot_config.wrist_camera_id)
        if wrist_image is None:
            raise RuntimeError(
                "Missing DROID wrist camera image for "
                f"{robot_config.wrist_camera_id}. Available image keys: {sorted(image_observations.keys())}."
            )
        wrist_image = _to_rgb_image(wrist_image)

    exterior_image = None
    if robot_config.exterior_camera_id:
        exterior_image = _find_camera_image(image_observations, robot_config.exterior_camera_id)
        if exterior_image is None:
            raise RuntimeError(
                "Missing DROID exterior camera image for "
                f"{robot_config.exterior_camera_id}. Available image keys: {sorted(image_observations.keys())}."
            )
        exterior_image = _to_rgb_image(exterior_image)

    robot_state = obs_dict["robot_state"]
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    result = {
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }
    if wrist_image is not None:
        result["wrist_image"] = wrist_image
    if exterior_image is not None:
        result["exterior_image"] = exterior_image
    return result


def _to_rgb_image(image: Any) -> np.ndarray:
    image = np.asarray(image)[..., :3]
    return image[..., ::-1]


def _find_camera_image(image_observations: dict[str, Any], camera_id: str):
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


# Form input instance
def get_pi0_input(obs: dict[str, np.ndarray], instruction: str) -> dict[str, Any]:
    from openpi_client import image_tools

    request_data = {
        "observation/joint_position": obs["joint_position"],
        "observation/gripper_position": obs["gripper_position"],
        "prompt": instruction,
    }
    if "exterior_image" in obs:
        request_data["observation/exterior_image_1_left"] = image_tools.resize_with_pad(
            obs["exterior_image"], 224, 224
        )
    if "wrist_image" in obs:
        request_data["observation/wrist_image_left"] = image_tools.resize_with_pad(obs["wrist_image"], 224, 224)
    return request_data


def get_pi0_wrist_input(obs: dict[str, np.ndarray], instruction: str) -> dict[str, Any]:
    return get_pi0_input(obs, instruction)


def _select_video_frame(obs: dict[str, np.ndarray]) -> np.ndarray:
    if "wrist_image" in obs:
        return obs["wrist_image"]
    if "exterior_image" in obs:
        return obs["exterior_image"]
    raise RuntimeError("No camera image is available for rollout video.")


def binarize_and_clip_action(action):
    if action[-1].item() > 0.5:
        action = np.concatenate([action[:-1], np.ones((1,))])
    else:
        action = np.concatenate([action[:-1], np.zeros((1,))])
    return np.clip(action, -1, 1)


def save_rollout_video(outputdir: Path, episode_id: int, image_list: list[np.ndarray]) -> str:
    if not image_list:
        return ""

    from moviepy.editor import ImageSequenceClip
    from moviepy.video.io.ffmpeg_writer import ffmpeg_write_video

    video_path = outputdir / f"eval_video_{episode_id}.mp4"
    fps = float(VIDEO_FPS)
    video = np.stack(image_list)
    clip = ImageSequenceClip(list(video), fps=fps)
    ffmpeg_write_video(
        clip,
        str(video_path),
        fps,
        codec="libx264",
        audiofile=None,
        logger=None,
    )
    return str(video_path)


def append_result(csv_path: Path, args: argparse.Namespace, result: RolloutResult) -> None:
    row = {
        "episode_id": result.episode_id,
        "success": int(result.success),
        "failure_reason": result.failure_reason,
        "env_steps": result.env_steps,
        "duration_s": f"{result.duration_s:.3f}",
        "video_path": result.video_path,
        "timestamp": result.timestamp,
        "instruction": args.instruction,
        "use_wrist_camera": int(args.use_wrist_camera),
        "use_exterior_camera": int(args.use_exterior_camera),
        "wrist_camera_id": DEFAULT_WRIST_CAMERA_ID if args.use_wrist_camera else "",
        "exterior_camera_id": DEFAULT_EXTERIOR_CAMERA_ID if args.use_exterior_camera else "",
        "policy_host": args.policy_host,
        "policy_port": args.policy_port,
    }
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_evaluation(args: argparse.Namespace) -> None:
    if args.eval_episodes <= 0:
        raise ValueError("--eval_episodes must be positive.")
    if args.execution_steps <= 0:
        raise ValueError("--execution_steps must be positive.")
    if not args.use_wrist_camera and not args.use_exterior_camera:
        raise ValueError("At least one of --use_wrist_camera or --use_exterior_camera must be enabled.")

    exec_config = ExecutionConfig(
        execution_steps=args.execution_steps,
        robot_action_latency=args.robot_action_latency,
        action_exec_latency=args.action_exec_latency,
    )
    logging.info(
        "ExecutionConfig: execution_steps=%d robot_action_latency=%.3fs action_exec_latency=%.3fs",
        exec_config.execution_steps,
        exec_config.robot_action_latency,
        exec_config.action_exec_latency,
    )

    outputdir = resolve_outputdir(args.outputdir)
    csv_path = outputdir / "eval_results.csv"

    logging.info("Writing pi0 evaluation outputs to %s", outputdir)
    policy_service, robot_io = create_runtime(args)
    env = robot_io.env
    ui = HumanEvalUI(args.eval_episodes)

    completed = 0
    successes = 0
    try:
        for episode_id in range(args.eval_episodes):
            if not ui.wait_for_start(episode_id, completed, successes):
                break

            reset_robot(env, reason=f"before episode {episode_id}")
            result = run_rollout(
                args,
                env,
                policy_service,
                robot_io,
                ui,
                episode_id,
                completed,
                successes,
                outputdir,
                exec_config=exec_config,
            )
            completed += 1
            successes += int(result.success)
            append_result(csv_path, args, result)
            logging.info(
                "Episode %s done: success=%s reason=%s steps=%s duration=%.2fs success_rate=%.3f",
                result.episode_id,
                result.success,
                result.failure_reason or "success",
                result.env_steps,
                result.duration_s,
                successes / completed,
            )

            ui.set_resetting(episode_id, completed, successes)
            reset_robot(env, reason=f"after episode {episode_id}")

            if ui.quit_requested:
                break
    finally:
        ui.close()

    logging.info("Pi0 evaluation complete. Results: %s", csv_path)
    print(f"Pi0 evaluation complete. Results: {csv_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    run_evaluation(args)


if __name__ == "__main__":
    main()
