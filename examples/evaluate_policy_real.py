#!/usr/bin/env python3
"""Standalone real-world policy evaluation for DSRL pi0."""

import argparse
import csv
import dataclasses
import datetime as dt
import logging
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PI0_NOISE_HORIZON = 8
VIDEO_FPS = 15
DEFAULT_WRIST_CAMERA_ID = "17396664"
DEFAULT_EXTERIOR_CAMERA_ID = "241122302552"
DEFAULT_EXTERNAL_CAMERA = "right"

RESULT_FIELDS = [
    "episode_id",
    "success",
    "failure_reason",
    "env_steps",
    "duration_s",
    "video_path",
    "timestamp",
    "instruction",
    "restore_path",
    "use_wrist_camera",
    "use_exterior_camera",
    "wrist_camera_id",
    "exterior_camera_id",
    "policy_host",
    "policy_port",
]

TRAIN_ARG_DEFAULTS = {
    "actor_lr": 1e-4,
    "critic_lr": 3e-4,
    "temp_lr": 3e-4,
    "hidden_dims": (1024, 1024, 1024),
    "cnn_features": (32, 32, 32, 32),
    "cnn_strides": (3, 2, 2, 2),
    "cnn_padding": "VALID",
    "latent_dim": 50,
    "discount": 0.99,
    "tau": 0.005,
    "critic_reduction": "min",
    "dropout_rate": 0.0,
    "aug_next": 1,
    "use_bottleneck": 1,
    "encoder_type": "small",
    "encoder_norm": "group",
    "use_spatial_softmax": 1,
    "softmax_temperature": -1,
    "target_entropy": 0.0,
    "num_qs": 2,
    "action_magnitude": 2.0,
    "num_cameras": 3,
}


@dataclasses.dataclass
class RolloutResult:
    episode_id: int
    success: bool
    failure_reason: str
    env_steps: int
    duration_s: float
    video_path: str
    timestamp: str


class HumanEvalUI:
    PREVIEW_SIZE = (360, 270)
    BUTTON_FONT = ("Arial", 14, "bold")
    BUTTON_WIDTH = 12

    def __init__(self, total_episodes: int):
        import tkinter as tk

        self._tk = tk
        self.root = tk.Tk()
        self.root.title("Real Policy Evaluation")
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

        title = tk.Label(self.root, text="Real Policy Evaluation", font=("Arial", 16, "bold"))
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
    parser = argparse.ArgumentParser(description="Evaluate a trained DSRL pi0 policy on a real DROID robot.")
    parser.add_argument("--restore_path", required=True, help="Path to the PixelSAC/DSRL checkpoint to evaluate.")
    parser.add_argument("--instruction", default="put the spoon on the plate", help="Language instruction for pi0.")
    parser.add_argument("--eval_episodes", default=10, type=int, help="Number of real-world episodes to evaluate.")
    parser.add_argument("--max_rollout_steps", default=200, type=int, help="Max robot-control steps per episode.")
    parser.add_argument("--query_freq", default=10, type=int, help="Control steps to execute before querying again.")
    parser.add_argument("--resize_image", default=128, type=int, help="RL observation image resize resolution.")
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
        default=1,
        type=int,
        choices=(0, 1),
        help="Whether to use the hardcoded RealSense exterior camera.",
    )
    parser.add_argument("--policy_host", default="127.0.0.1", help="OpenPI policy server host.")
    parser.add_argument("--policy_port", default=8000, type=int, help="OpenPI policy server port.")
    parser.add_argument("--outputdir", default=None, help="Directory for eval_results.csv and rollout videos.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed used to initialize the agent shell.")
    parser.add_argument("--add_states", default=1, type=int, help="Whether the SAC checkpoint expects state inputs.")

    for name, default in TRAIN_ARG_DEFAULTS.items():
        if name in {"num_cameras"}:
            parser.add_argument(f"--{name}", default=default, type=type(default))
        elif isinstance(default, tuple):
            parser.add_argument(f"--{name}", nargs="+", default=default, type=type(default[0]))
        else:
            parser.add_argument(f"--{name}", default=default, type=type(default))
    return parser


def make_variant(args: argparse.Namespace, attr_dict_cls: type[dict]) -> Any:
    variant = attr_dict_cls(vars(args))
    train_kwargs = {}
    for name, default in TRAIN_ARG_DEFAULTS.items():
        value = getattr(args, name)
        if isinstance(default, tuple):
            value = tuple(value)
        train_kwargs[name] = value
    variant["train_kwargs"] = train_kwargs
    return variant


def resolve_outputdir(outputdir: str | None) -> Path:
    if outputdir:
        path = Path(outputdir)
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("logs") / f"policy_eval_real_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_agent(variant: Any):
    from examples.train_real import DummyEnv
    from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
    from jaxrl2.utils.general_utils import add_batch_dim

    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())

    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **variant.train_kwargs)
    agent.restore_checkpoint(variant.restore_path)
    return agent


def create_runtime(args: argparse.Namespace):
    from examples.train_real import PolicyServerConfig
    from examples.train_real import PolicyService
    from examples.train_real import RobotIO
    from examples.train_real import RobotRuntimeConfig

    policy_service = PolicyService(PolicyServerConfig(host=args.policy_host, port=args.policy_port))
    policy_service.preflight()

    runtime_config = RobotRuntimeConfig(
        external_camera=DEFAULT_EXTERNAL_CAMERA,
        left_camera_id=DEFAULT_EXTERIOR_CAMERA_ID if args.use_exterior_camera else "",
        right_camera_id=DEFAULT_EXTERIOR_CAMERA_ID if args.use_exterior_camera else "",
        wrist_camera_id=DEFAULT_WRIST_CAMERA_ID if args.use_wrist_camera else "",
        max_timesteps=args.max_rollout_steps,
        control_frequency_hz=args.control_frequency_hz,
        allow_missing_cameras=True,
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
    variant: Any,
    agent: Any,
    env: Any,
    policy_service: Any,
    robot_io: Any,
    ui: HumanEvalUI,
    episode_id: int,
    completed: int,
    successes: int,
    outputdir: Path,
) -> RolloutResult:
    import numpy as np
    from tqdm import tqdm

    from examples.train_utils_real import _extract_observation
    from examples.train_utils_real import build_rl_observation
    from examples.train_utils_real import get_pi0_input

    runtime_config = robot_io.runtime_config
    step_time = 1 / runtime_config.control_frequency_hz

    ui.set_running(episode_id, completed, successes)
    start_time = time.time()
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    image_list = []
    actions = None
    decision: tuple[bool, str] | None = None
    env_steps = 0

    for t in tqdm(range(runtime_config.max_timesteps), desc=f"eval episode {episode_id}"):
        decision = ui.poll()
        if decision is not None:
            break

        step_started = time.time()
        _env_obs = env.get_observation()
        curr_obs = _extract_observation(runtime_config, _env_obs)
        exterior_image_key = runtime_config.camera_to_use + "_image"
        ui.update_camera_previews(
            wrist_image=curr_obs["wrist_image"],
            exterior_image=curr_obs[exterior_image_key],
        )
        image_list.append(_select_video_frame(curr_obs, runtime_config))
        request_data = get_pi0_input(curr_obs, runtime_config, args.instruction)

        if t % args.query_freq == 0:
            obs_dict = build_rl_observation(variant, curr_obs, request_data, policy_service)
            if not variant.add_states:
                obs_dict.pop("state", None)
            noise = make_eval_noise(agent, obs_dict)
            response = policy_service.infer(request_data, noise=noise)
            actions = np.asarray(response["actions"])
            if actions.shape[0] < args.query_freq:
                raise RuntimeError(
                    f"Policy server returned {actions.shape[0]} actions, but query_freq={args.query_freq}."
                )

        if actions is None:
            raise RuntimeError("No action chunk available. This should be impossible at t=0.")

        action_t = _binarize_and_clip_action(actions[t % args.query_freq])
        env.step(action_t)
        env_steps = t + 1

        ui.update_step(episode_id, env_steps, completed, successes)
        decision = ui.poll()
        if decision is not None:
            break

        elapsed = time.time() - step_started
        if elapsed < step_time:
            time.sleep(step_time - elapsed)

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


def make_eval_noise(agent: Any, obs_dict: dict[str, Any]):
    import numpy as np
    from jaxrl2.utils.noise_utils import make_full_horizon_noise

    actions_noise = agent.eval_actions(obs_dict)
    actions_noise, noise = make_full_horizon_noise(actions_noise, agent.action_chunk_shape)
    if actions_noise.shape[0] != PI0_NOISE_HORIZON:
        raise RuntimeError(
            f"RL action chunk length {actions_noise.shape[0]} must match pi0 noise horizon {PI0_NOISE_HORIZON}."
        )
    return np.asarray(noise)


def _binarize_and_clip_action(action):
    import numpy as np

    if action[-1].item() > 0.5:
        action = np.concatenate([action[:-1], np.ones((1,))])
    else:
        action = np.concatenate([action[:-1], np.zeros((1,))])
    return np.clip(action, -1, 1)


def _select_video_frame(obs: dict[str, Any], runtime_config: Any):
    exterior_camera = runtime_config.camera_to_use
    exterior_image_key = exterior_camera + "_image"
    exterior_present_key = exterior_camera + "_image_present"
    if obs.get(exterior_present_key, False):
        return obs[exterior_image_key]
    if obs.get("wrist_image_present", False):
        return obs["wrist_image"]
    return obs[exterior_image_key]


def save_rollout_video(outputdir: Path, episode_id: int, image_list: list[Any]) -> str:
    if not image_list:
        return ""

    import numpy as np
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
        "restore_path": args.restore_path,
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
    if args.query_freq <= 0:
        raise ValueError("--query_freq must be positive.")
    if args.query_freq > PI0_NOISE_HORIZON:
        raise ValueError(f"--query_freq must be <= {PI0_NOISE_HORIZON} for the current pi0 real action horizon.")
    if not args.use_wrist_camera and not args.use_exterior_camera:
        raise ValueError("At least one of --use_wrist_camera or --use_exterior_camera must be enabled.")

    from jaxrl2.utils.general_utils import AttrDict

    variant = make_variant(args, AttrDict)
    outputdir = resolve_outputdir(args.outputdir)
    csv_path = outputdir / "eval_results.csv"

    logging.info("Writing evaluation outputs to %s", outputdir)
    agent = create_agent(variant)
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
                variant,
                agent,
                env,
                policy_service,
                robot_io,
                ui,
                episode_id,
                completed,
                successes,
                outputdir,
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

    logging.info("Evaluation complete. Results: %s", csv_path)
    print(f"Evaluation complete. Results: {csv_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    run_evaluation(args)


if __name__ == "__main__":
    main()
