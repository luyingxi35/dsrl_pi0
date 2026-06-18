#!/usr/bin/env python3
"""Robometer reward server — runs under robometer conda env.

Serves Robometer progress estimation via length-prefixed pickle over stdin/stdout.
Mirrors the mani_skill_server.py protocol pattern.

Protocol:
  Client → Server: {"cmd": "infer", "frames": ndarray(T,H,W,3) uint8, "task": str}
               or: {"cmd": "close"}
  Server → Client: {"ok": True, "progress": ndarray(T,), "success_probs": ndarray(T,)}
               or: {"ok": False, "error": str}

Usage (spawned automatically by RobometerRewardClient):
  /opt/yingxi/envs/robometer/bin/python3 robometer_reward_server.py \
      --checkpoint_path /opt/yingxi/checkpoint-400 \
      --base_model_id   /opt/caoyuhang/Pretrained_models/Qwen3-VL-4B-Instruct
"""
import argparse
import os
import struct
import sys
import traceback
from dataclasses import fields

import numpy as np

# ── Add robometer to Python path (runs in robometer env, but code lives in repo) ──
_ROBOMETER_ROOT = "/home/gpu4/yingxi/RoboFPE/robometer"
if _ROBOMETER_ROOT not in sys.path:
    sys.path.insert(0, _ROBOMETER_ROOT)

# Set config path before any robometer imports so load_model_from_hf can find it.
_DEFAULT_CONFIG_PATH = os.path.join(_ROBOMETER_ROOT, "robometer", "configs", "config.yaml")
os.environ.setdefault("ROBOMETER_CONFIG_YAML_PATH", _DEFAULT_CONFIG_PATH)

import pickle  # noqa: E402
import torch   # noqa: E402


# ── Pipe helpers (mirrors mani_skill_server.py protocol) ──────────────────────

def _send(data: dict) -> None:
    raw = pickle.dumps(data, protocol=4)
    sys.stdout.buffer.write(struct.pack(">I", len(raw)))
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()


def _recv() -> dict:
    header = sys.stdin.buffer.read(4)
    if len(header) < 4:
        raise EOFError("Client closed the pipe")
    n = struct.unpack(">I", header)[0]
    return pickle.loads(sys.stdin.buffer.read(n))


# ── Config loading (replicates load_model_from_hf logic, with base_model_id override) ──

def _load_exp_config_with_override(config_path: str, base_model_id: str):
    """Load ExperimentConfig from config.yaml and override base_model_id."""
    import yaml
    from robometer.configs.experiment_configs import ExperimentConfig

    class _SafeLoader(yaml.SafeLoader):
        pass

    # Some checkpoints saved with python-object tags; ignore them safely.
    _SafeLoader.add_constructor(
        "tag:yaml.org,2002:python/object:robometer.models.rewind_transformer.ReWINDTransformerConfig",
        lambda loader, node: loader.construct_mapping(node),
    )

    with open(config_path) as f:
        cfg_dict = yaml.load(f.read(), Loader=_SafeLoader)

    # Override base_model_id before building ExperimentConfig
    model_dict = cfg_dict.get("model", {}) if isinstance(cfg_dict.get("model"), dict) else {}
    loss_dict  = cfg_dict.get("loss",  {}) if isinstance(cfg_dict.get("loss"),  dict) else {}

    model_dict["base_model_id"] = base_model_id
    model_dict["use_unsloth"]   = False   # disable unsloth for inference

    # Resolve OmegaConf-style ${loss.*} interpolations that yaml.safe_load leaves as
    # literal strings.  Without this, the progress head is initialised as continuous
    # (output_size=1) while the checkpoint was trained with discrete 20 bins.
    for _fname, _lkey, _default in [
        ("progress_loss_type",    "progress_loss_type",    "l2"),
        ("progress_discrete_bins", "progress_discrete_bins", 10),
    ]:
        _val = model_dict.get(_fname, "")
        if isinstance(_val, str) and _val.startswith("${"):
            model_dict[_fname] = loss_dict.get(_lkey, _default)

    cfg_dict["model"] = model_dict

    valid_keys = {f.name for f in fields(ExperimentConfig)}
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    return ExperimentConfig(**filtered)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Robometer reward inference server")
    parser.add_argument("--checkpoint_path", required=True,
                        help="Path to fine-tuned Robometer checkpoint (e.g. /opt/yingxi/checkpoint-400)")
    parser.add_argument("--base_model_id", required=True,
                        help="Path to base VLM for processor/tokenizer (e.g. Qwen3-VL-4B-Instruct dir)")
    parser.add_argument("--config_path", default=_DEFAULT_CONFIG_PATH,
                        help="Path to robometer config.yaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[robometer_server] device={device}, checkpoint={args.checkpoint_path}", file=sys.stderr)
    sys.stderr.flush()

    # ── Load config with corrected base_model_id ──────────────────────────────
    exp_config = _load_exp_config_with_override(args.config_path, args.base_model_id)

    # ── Load processor + model weights ───────────────────────────────────────
    from robometer.utils.setup_utils import setup_model_and_processor, setup_batch_collator
    tokenizer, processor, model = setup_model_and_processor(
        exp_config.model,
        hf_model_id=args.checkpoint_path,
    )
    model = model.to(device)
    model.eval()
    print(f"[robometer_server] model loaded on {device}", file=sys.stderr)
    sys.stderr.flush()

    # ── Setup batch collator ──────────────────────────────────────────────────
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

    # ── Determine discrete/continuous mode ────────────────────────────────────
    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config else False
    )
    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )
    print(f"[robometer_server] is_discrete={is_discrete}, num_bins={num_bins}", file=sys.stderr)
    sys.stderr.flush()

    # ── Signal ready to client ────────────────────────────────────────────────
    _send({"ok": True, "ready": True})

    # ── Serve loop ────────────────────────────────────────────────────────────
    from robometer.data.dataset_types import Trajectory, ProgressSample
    from robometer.evals.eval_server import compute_batch_outputs

    while True:
        try:
            msg = _recv()
        except EOFError:
            print("[robometer_server] pipe closed, exiting", file=sys.stderr)
            break

        if msg.get("cmd") == "close":
            print("[robometer_server] received close, exiting", file=sys.stderr)
            break

        frames = np.asarray(msg["frames"], dtype=np.uint8)   # (T, H, W, 3)
        task   = str(msg["task"])
        T      = int(frames.shape[0])

        traj = Trajectory(
            frames=frames,
            frames_shape=tuple(frames.shape),
            task=task,
            id="rollout",
            metadata={"subsequence_length": T},
            video_embeddings=None,
        )
        sample = ProgressSample(trajectory=traj, sample_type="progress")

        try:
            with torch.no_grad():
                batch = batch_collator([sample])
                progress_inputs = batch["progress_inputs"]
                for k, v in progress_inputs.items():
                    if hasattr(v, "to"):
                        progress_inputs[k] = v.to(device)

                results = compute_batch_outputs(
                    model, tokenizer, progress_inputs,
                    sample_type="progress",
                    is_discrete_mode=is_discrete,
                    num_bins=num_bins,
                )

            progress_list = results.get("progress_pred", [[]])
            progress_arr  = np.array(
                progress_list[0] if progress_list else [], dtype=np.float32
            )

            success_data  = (results.get("outputs_success") or {}).get("success_probs", [[]])
            success_arr   = np.array(
                success_data[0] if success_data else [], dtype=np.float32
            )

            _send({"ok": True, "progress": progress_arr, "success_probs": success_arr})

        except Exception as exc:  # noqa: BLE001
            err_msg = f"{exc}\n{traceback.format_exc()}"
            print(f"[robometer_server] inference error: {err_msg}", file=sys.stderr)
            sys.stderr.flush()
            _send({"ok": False, "error": err_msg})


if __name__ == "__main__":
    main()
