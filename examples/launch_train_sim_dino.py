"""CLI entry point for train_sim_dino.py.

Runs ManiSkill PegInsertionVertical via robofac subprocess + pi0_droid + StateSAC + DINOv2.
"""
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse

from examples.train_sim_dino import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DSRL sim training: ManiSkill subprocess + pi0_droid + StateSAC + DINOv2"
    )

    # ── Training configuration ─────────────────────────────────────────────────
    parser.add_argument("--seed",                 default=42,     type=int)
    parser.add_argument("--launch_group_id",      default="")
    parser.add_argument("--eval_episodes",        default=5,      type=int)
    parser.add_argument("--env",                  default="peg_insertion_vertical")
    parser.add_argument("--log_interval",         default=100,    type=int)
    parser.add_argument("--eval_interval",        default=2000,   type=int)
    parser.add_argument("--checkpoint_interval",  default=10000,  type=int)
    parser.add_argument("--batch_size",           default=256,    type=int)
    parser.add_argument("--max_steps",            default=500_000, type=int)
    parser.add_argument("--max_timesteps",        default=600,    type=int)
    parser.add_argument("--add_states",           default=1,      type=int)
    parser.add_argument("--wandb_project",        default="DSRL_pi0_SimDino")
    parser.add_argument("--num_initial_traj_collect", default=5,  type=int)
    parser.add_argument("--algorithm",            default="state_sac")
    parser.add_argument("--prefix",               default="")
    parser.add_argument("--suffix",               default="")
    parser.add_argument("--multi_grad_step",      default=5,      type=int)
    parser.add_argument("--query_freq",           default=8,      type=int)
    parser.add_argument("--rl_noise_horizon",     default=8,      type=int)
    parser.add_argument("--instruction",
                        default="pick up the peg and insert it vertically")
    parser.add_argument("--checkpoint_path",      default="/opt/yingxi/pi0_droid")
    parser.add_argument("--action_scale",         default=0.5,    type=float,
                        help="Scale on DROID max_joint_delta=0.2 rad/step. Default 0.5 => 0.1 rad/step.")
    parser.add_argument("--dino_model",           default="facebook/dinov2-small")
    parser.add_argument("--dino_device",          default="auto")

    # ── Subprocess env ─────────────────────────────────────────────────────────
    parser.add_argument("--robofac_python",
                        default="/opt/yingxi/envs/robofac/bin/python3",
                        help="Path to robofac conda Python (runs ManiSkill server)")
    parser.add_argument("--workspace_bounds_path",
                        default="/home/gpu4/yingxi/dsrl_pi0/workspace_bounds.json",
                        help="Path to workspace_bounds.json from calibrate_workspace.py "
                             "(default: calibrated bounds; set to empty string to disable)")

    # ── Sweep / early-stop ─────────────────────────────────────────────────────
    parser.add_argument("--eval_env_step_interval", default=1000, type=int,
                        help="Eval + checkpoint every N env steps (-1=off)")
    parser.add_argument("--stop_success_rate",      default=0.95, type=float)
    parser.add_argument("--stop_window",            default=2,    type=int)

    # ── StateSACLearner hyperparams (auto-added as CLI flags by parse_training_args)
    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        discount=0.99,
        tau=0.005,
        hidden_dims=(1024, 1024, 1024),
        network_type="transformer",
        transformer_dim=256,
        transformer_depth=3,
        transformer_heads=4,
        transformer_mlp_dim=1024,
        transformer_dropout=0.0,
        critic_reduction="min",
        dropout_rate=0.0,
        target_entropy=0.0,
        num_qs=2,
        action_magnitude=2.0,
    )

    variant, args = parse_training_args(train_args_dict, parser)
    print(variant)
    main(variant)
    sys.exit()
