"""Shared robot configurations, utilities, and UI for DROID real-robot scripts.

Consolidates code that was previously duplicated across evaluate_pi0_real.py,
evaluate_policy_real.py, train_real.py, train_real_dino.py, and train_utils_real.py.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np


# ── Camera IDs ────────────────────────────────────────────────────────────────

DEFAULT_WRIST_CAMERA_ID = "17396664"
DEFAULT_EXTERIOR_CAMERA_ID = "241122302552"
VIDEO_FPS = 15


# ── Config dataclasses ────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class PolicyServerConfig:
    host: str
    port: int


@dataclasses.dataclass(frozen=True)
class RobotRuntimeConfig:
    """Runtime config for training scripts (multi-camera, full-featured)."""
    external_camera: str
    left_camera_id: str
    right_camera_id: str
    wrist_camera_id: str
    max_timesteps: int
    control_frequency_hz: int = 10
    allow_missing_cameras: bool = False

    @property
    def camera_to_use(self) -> str:
        return self.external_camera

    def validate(self) -> None:
        camera_ids = {
            "left_camera_id": self.left_camera_id,
            "right_camera_id": self.right_camera_id,
            "wrist_camera_id": self.wrist_camera_id,
        }
        missing = [name for name, value in camera_ids.items() if not value]
        if missing and not (self.allow_missing_cameras and any(camera_ids.values())):
            raise ValueError(
                "DROID camera IDs must be set before running real rollouts. "
                f"Missing: {', '.join(missing)}."
            )
        if self.wrist_camera_id and self.wrist_camera_id in {
            self.left_camera_id, self.right_camera_id
        }:
            raise ValueError(
                f"The wrist camera must be different from the external camera IDs: {camera_ids}"
            )


@dataclasses.dataclass
class RolloutResult:
    episode_id: int
    success: bool
    failure_reason: str
    env_steps: int
    duration_s: float
    video_path: str
    timestamp: str


# ── Policy service ─────────────────────────────────────────────────────────────

class PolicyService:
    """Wrapper around the OpenPI websocket policy client."""

    def __init__(self, config: PolicyServerConfig) -> None:
        from openpi_client import websocket_client_policy as _wcp
        self._config = config
        self._client = _wcp.WebsocketClientPolicy(host=config.host, port=config.port)

    def preflight(self):
        metadata = self._client.get_server_metadata()
        logging.info("OpenPI policy server metadata: %s", metadata)
        return metadata

    def infer(self, obs, noise=None):
        return self._client.infer(obs, noise=noise)

    def get_prefix_rep(self, obs):
        return self._client.get_prefix_rep(obs)


# ── Robot IO ───────────────────────────────────────────────────────────────────

class RobotIO:
    """Thin wrapper around DROID RobotEnv for training scripts."""

    def __init__(self, runtime_config: RobotRuntimeConfig, action_space: str = "joint_position") -> None:
        from droid.robot_env import RobotEnv
        self._runtime_config = runtime_config
        self._env = RobotEnv(action_space=action_space, gripper_action_space="position")

    @property
    def env(self):
        return self._env

    @property
    def runtime_config(self) -> RobotRuntimeConfig:
        return self._runtime_config

    def preflight(self):
        obs = self._env.get_observation()
        image_observations = obs["image"]
        required_cameras = [
            (self._runtime_config.left_camera_id, "left external"),
            (self._runtime_config.right_camera_id, "right external"),
            (self._runtime_config.wrist_camera_id, "wrist"),
        ]
        missing = [
            label
            for cam_id, label in required_cameras
            if cam_id and not _has_camera_image(image_observations, cam_id)
        ]
        if missing:
            raise RuntimeError(
                "DROID camera preflight failed. Missing image feeds for: "
                + ", ".join(missing)
            )
        return obs


# ── Camera utilities ───────────────────────────────────────────────────────────

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


def _find_camera_image(image_observations: dict[str, Any], camera_id: str):
    """Find a camera image by ID, preferring left-view when multiple views exist."""
    if not camera_id:
        return None
    candidates = _camera_id_candidates(camera_id)
    matches = []
    for key, image in image_observations.items():
        if any(key == c or key.startswith(f"{c}_") for c in candidates):
            matches.append((key, image))
    if not matches:
        return None
    left_view = [img for key, img in matches if key.endswith("_left") or "left" in key]
    return left_view[0] if left_view else matches[0][1]


def _has_camera_image(image_observations: dict[str, Any], camera_id: str) -> bool:
    if not camera_id:
        return False
    candidates = _camera_id_candidates(camera_id)
    return any(
        key == c or key.startswith(f"{c}_")
        for key in image_observations
        for c in candidates
    )


def _to_rgb_image(image: Any) -> np.ndarray:
    image = np.asarray(image)[..., :3]
    return image[..., ::-1]


def _empty_rgb_image_like(*images) -> np.ndarray:
    for image in images:
        if image is not None:
            return np.zeros_like(image)
    return np.zeros((224, 224, 3), dtype=np.uint8)


# ── Observation extraction ─────────────────────────────────────────────────────

def extract_observation_train(robot_config: RobotRuntimeConfig, obs_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract and RGB-convert camera images + proprioception for training scripts.

    Returns a dict with left_image, right_image, wrist_image (+ _present flags),
    cartesian_position, joint_position, gripper_position.
    """
    image_observations = obs_dict["image"]
    left_image = _find_camera_image(image_observations, robot_config.left_camera_id)
    right_image = _find_camera_image(image_observations, robot_config.right_camera_id)
    wrist_image = _find_camera_image(image_observations, robot_config.wrist_camera_id)

    missing = [
        name
        for name, img in (("left_image", left_image), ("right_image", right_image), ("wrist_image", wrist_image))
        if img is None
    ]
    allow_missing = getattr(robot_config, "allow_missing_cameras", False)
    if missing and not allow_missing:
        raise RuntimeError(
            "Missing DROID camera images: "
            + ", ".join(missing)
            + f". Available: {sorted(image_observations.keys())}."
        )

    left_image_present = left_image is not None
    right_image_present = right_image is not None
    wrist_image_present = wrist_image is not None

    left_image = _to_rgb_image(left_image) if left_image_present else None
    right_image = _to_rgb_image(right_image) if right_image_present else None
    wrist_image = _to_rgb_image(wrist_image) if wrist_image_present else None

    empty = _empty_rgb_image_like(left_image, right_image, wrist_image)
    if left_image is None:
        left_image = empty.copy()
    if right_image is None:
        right_image = empty.copy()
    if wrist_image is None:
        wrist_image = empty.copy()

    robot_state = obs_dict["robot_state"]
    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "left_image_present": left_image_present,
        "right_image_present": right_image_present,
        "wrist_image_present": wrist_image_present,
        "cartesian_position": np.array(robot_state["cartesian_position"]),
        "joint_position": np.array(robot_state["joint_positions"]),
        "gripper_position": np.array([robot_state["gripper_position"]]),
    }


def extract_observation_eval(
    wrist_camera_id: str | None,
    exterior_camera_id: str | None,
    obs_dict: dict[str, Any],
) -> dict[str, Any]:
    """Extract camera images + proprioception for pi0-eval scripts (simpler interface)."""
    image_observations = obs_dict["image"]

    wrist_image = None
    if wrist_camera_id:
        wrist_image = _find_camera_image(image_observations, wrist_camera_id)
        if wrist_image is None:
            raise RuntimeError(
                f"Missing DROID wrist camera image for {wrist_camera_id}. "
                f"Available: {sorted(image_observations.keys())}."
            )
        wrist_image = _to_rgb_image(wrist_image)

    exterior_image = None
    if exterior_camera_id:
        exterior_image = _find_camera_image(image_observations, exterior_camera_id)
        if exterior_image is None:
            raise RuntimeError(
                f"Missing DROID exterior camera image for {exterior_camera_id}. "
                f"Available: {sorted(image_observations.keys())}."
            )
        exterior_image = _to_rgb_image(exterior_image)

    robot_state = obs_dict["robot_state"]
    result: dict[str, Any] = {
        "joint_position": np.array(robot_state["joint_positions"]),
        "gripper_position": np.array([robot_state["gripper_position"]]),
    }
    if wrist_image is not None:
        result["wrist_image"] = wrist_image
    if exterior_image is not None:
        result["exterior_image"] = exterior_image
    return result


# ── pi0 input construction ─────────────────────────────────────────────────────

def get_pi0_input_eval(obs: dict[str, np.ndarray], instruction: str) -> dict[str, Any]:
    """Build pi0 request dict for pi0-eval scripts (wrist + exterior cameras)."""
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
        request_data["observation/wrist_image_left"] = image_tools.resize_with_pad(
            obs["wrist_image"], 224, 224
        )
    return request_data


def get_pi0_input_train(obs: dict[str, np.ndarray], robot_config: RobotRuntimeConfig, instruction: str) -> dict[str, Any]:
    """Build pi0 request dict for training scripts (left/right/wrist cameras)."""
    from openpi_client import image_tools
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
        request_data["observation/wrist_image_left"] = image_tools.resize_with_pad(
            obs["wrist_image"], 224, 224
        )
    return request_data


# ── Action utilities ───────────────────────────────────────────────────────────

def binarize_and_clip_action(action: np.ndarray) -> np.ndarray:
    """Binarize gripper dimension (last) and clip all dims to [-1, 1]."""
    gripper = np.ones((1,)) if action[-1].item() > 0.5 else np.zeros((1,))
    return np.clip(np.concatenate([action[:-1], gripper]), -1, 1)


# ── Video utilities ────────────────────────────────────────────────────────────

def save_rollout_video(outputdir: Path, episode_id: int, image_list: list[np.ndarray]) -> str:
    if not image_list:
        return ""
    from moviepy.editor import ImageSequenceClip
    from moviepy.video.io.ffmpeg_writer import ffmpeg_write_video

    video_path = outputdir / f"eval_video_{episode_id}.mp4"
    fps = float(VIDEO_FPS)
    video = np.stack(image_list)
    clip = ImageSequenceClip(list(video), fps=fps)
    ffmpeg_write_video(clip, str(video_path), fps, codec="libx264", audiofile=None, logger=None)
    return str(video_path)


def select_video_frame_eval(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Pick one frame for the rollout video (eval scripts)."""
    if "wrist_image" in obs:
        return obs["wrist_image"]
    if "exterior_image" in obs:
        return obs["exterior_image"]
    raise RuntimeError("No camera image available for rollout video.")


def select_video_frame_train(obs: dict[str, Any], robot_config: RobotRuntimeConfig) -> np.ndarray:
    """Pick one frame for the rollout video (training scripts)."""
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


# ── Eval lifecycle utilities ───────────────────────────────────────────────────

def reset_robot(env: Any, reason: str) -> None:
    logging.info("Resetting DROID environment (%s)...", reason)
    try:
        env.reset()
    except Exception:
        logging.exception("Environment reset failed (%s).", reason)
        raise


def resolve_outputdir(outputdir: str | None, prefix: str = "eval_real") -> Path:
    if outputdir:
        path = Path(outputdir)
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("logs") / f"{prefix}_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_result(csv_path: Path, row: dict[str, Any], result_fields: list[str]) -> None:
    """Append one result row to the CSV log."""
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def format_stats(completed: int, successes: int) -> str:
    if completed == 0:
        return "Completed: 0 | Success rate: n/a"
    return f"Completed: {completed} | Successes: {successes} | Success rate: {successes / completed:.3f}"


# ── Human Eval UI ─────────────────────────────────────────────────────────────

class HumanEvalUI:
    """Tkinter GUI for human-supervised robot evaluation and training.

    Args:
        title: Window title string.
        total_episodes: Total number of episodes (shown in status line).
            Pass None for training loops where episode count is unbounded.
        preview_names: Iterable of (key, display_title) pairs for camera previews.
    """

    PREVIEW_SIZE = (360, 270)
    BUTTON_FONT = ("Arial", 14, "bold")
    BUTTON_WIDTH = 12

    def __init__(
        self,
        title: str = "Real Evaluation",
        total_episodes: int | None = None,
        preview_names: tuple[tuple[str, str], ...] = (("wrist", "Wrist"), ("exterior", "Exterior")),
    ) -> None:
        try:
            import tkinter as tk
        except ImportError as exc:
            raise RuntimeError("HumanEvalUI requires tkinter.") from exc

        self._tk = tk
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("820x610")
        self.root.protocol("WM_DELETE_WINDOW", self.request_quit)

        self.total_episodes = total_episodes
        self.start_requested = False
        self.quit_requested = False
        self.running = False
        self.decision: tuple[bool, str] | None = None
        self.closed = False
        self._preview_photos: dict[str, Any] = {}

        self.status_var = tk.StringVar(value="Waiting to start.")
        self.stats_var = tk.StringVar(value="")

        tk.Label(self.root, text=title, font=("Arial", 16, "bold")).pack(pady=(14, 4))
        tk.Label(self.root, textvariable=self.status_var, font=("Arial", 11)).pack(pady=4)
        tk.Label(self.root, textvariable=self.stats_var, font=("Arial", 10)).pack(pady=2)

        preview_container = tk.Frame(self.root)
        preview_container.pack(pady=(10, 8))
        self.preview_labels: dict[str, Any] = {}
        for col, (name, title_text) in enumerate(preview_names):
            col_frame = tk.Frame(preview_container)
            col_frame.grid(row=0, column=col, padx=8)
            tk.Label(col_frame, text=title_text, font=("Arial", 10, "bold")).pack(pady=(0, 4))
            pf = tk.Frame(col_frame, width=self.PREVIEW_SIZE[0], height=self.PREVIEW_SIZE[1], bg="black")
            pf.pack()
            pf.pack_propagate(False)
            lbl = tk.Label(pf, bg="black", bd=0)
            lbl.pack(expand=True)
            self.preview_labels[name] = lbl

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=14)
        self.start_button = tk.Button(
            button_frame, text="Start next", width=self.BUTTON_WIDTH,
            font=self.BUTTON_FONT, command=self.request_start,
        )
        self.start_button.grid(row=0, column=0, padx=6)
        self.success_button = tk.Button(
            button_frame, text="Success", width=self.BUTTON_WIDTH,
            font=self.BUTTON_FONT, command=self.mark_success,
        )
        self.success_button.grid(row=0, column=1, padx=6)
        self.failure_button = tk.Button(
            button_frame, text="Failure", width=self.BUTTON_WIDTH,
            font=self.BUTTON_FONT, command=self.mark_failure,
        )
        self.failure_button.grid(row=0, column=2, padx=6)
        tk.Button(
            self.root, text="Quit", width=self.BUTTON_WIDTH,
            font=self.BUTTON_FONT, command=self.request_quit,
        ).pack(pady=(2, 10))

        self.set_idle(0, 0, 0)
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
        ep_str = f"{episode_id + 1}/{self.total_episodes}" if self.total_episodes else str(episode_id + 1)
        self.status_var.set(f"Episode {ep_str}: waiting for Start next.")
        self.stats_var.set(format_stats(completed, successes))
        self.start_button.config(state="normal")
        self._set_decision_buttons("disabled")

    def set_running(self, episode_id: int, completed: int, successes: int) -> None:
        if self.closed:
            return
        self.running = True
        self.start_requested = False
        self.decision = None
        ep_str = f"{episode_id + 1}/{self.total_episodes}" if self.total_episodes else str(episode_id + 1)
        self.status_var.set(f"Episode {ep_str}: running.")
        self.stats_var.set(format_stats(completed, successes))
        self.start_button.config(state="disabled")
        self._set_decision_buttons("normal")

    def set_resetting(self, episode_id: int, completed: int, successes: int) -> None:
        if self.closed:
            return
        self.running = False
        ep_str = f"{episode_id + 1}/{self.total_episodes}" if self.total_episodes else str(episode_id + 1)
        self.status_var.set(f"Episode {ep_str}: resetting robot.")
        self.stats_var.set(format_stats(completed, successes))
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
        ep_str = f"{episode_id + 1}/{self.total_episodes}" if self.total_episodes else str(episode_id + 1)
        self.status_var.set(f"Episode {ep_str}: running step {step}.")
        self.stats_var.set(format_stats(completed, successes))
        self.update()

    def update_camera_previews(self, **images: np.ndarray | None) -> None:
        """Update preview panels. Pass keyword args matching preview_names keys."""
        if self.closed:
            return
        try:
            for name, image in images.items():
                if image is not None:
                    self._set_preview_image(name, image)
            self.update()
        except self._tk.TclError:
            self.closed = True
            self.quit_requested = True

    def _set_preview_image(self, preview_name: str, image: Any) -> None:
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
