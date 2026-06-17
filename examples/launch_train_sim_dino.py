"""CLI entry point for train_sim_dino.py.

PegInsertionVertical-v1 (ManiSkill2/SAPIEN) + pi0_droid (local) + StateSAC + DINOv2.
"""
import sys
from pathlib import Path

# Ensure repo root is on sys.path so 'examples' package is importable
# regardless of the CWD from which this script is executed.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse

from examples.train_sim_dino import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DSRL sim training: PegInsertionVertical + pi0_droid + StateSAC + DINOv2"
    )

    # ── Training configuration (NOT passed to StateSACLearner) ─────────────────
    parser.add_argument("--seed",               default=42,         type=int)
    parser.add_argument("--launch_group_id",    default="")
    parser.add_argument("--eval_episodes",      default=5,          type=int,
                        help="Number of rollouts per evaluation.")
    parser.add_argument("--env",                default="peg_insertion_vertical",
                        help="Environment identifier (informational).")
    parser.add_argument("--log_interval",       default=100,        type=int,
                        help="Gradient steps between WandB training logs.")
    parser.add_argument("--eval_interval",      default=2000,       type=int,
                        help="Gradient steps between evaluation runs.")
    parser.add_argument("--checkpoint_interval", default=10000,     type=int,
                        help="Gradient steps between checkpoint saves (-1 = disabled).")
    parser.add_argument("--batch_size",         default=256,        type=int)
    parser.add_argument("--max_steps",          default=500_000,    type=int,
                        help="Total number of gradient steps.")
    parser.add_argument("--max_timesteps",      default=300,        type=int,
                        help="Max control steps per trajectory "
                             "(should match PegInsertionVertical-v1 max_episode_steps=300).")
    parser.add_argument("--add_states",         default=1,          type=int,
                        help="Whether to include state in obs (must be 1 for sim_dino).")
    parser.add_argument("--wandb_project",      default="DSRL_pi0_SimDino")
    parser.add_argument("--num_initial_traj_collect", default=5,    type=int,
                        help="Number of trajectories to collect before starting "
                             "gradient updates.")
    parser.add_argument("--algorithm",          default="state_sac",
                        help="Algorithm identifier (informational).")
    parser.add_argument("--prefix",             default="")
    parser.add_argument("--suffix",             default="")
    parser.add_argument("--multi_grad_step",    default=5,          type=int,
                        help="Gradient steps per trajectory step (UTD ratio).")
    parser.add_argument("--query_freq",         default=8,          type=int,
                        help="Env steps between pi0 inference calls.")
    parser.add_argument("--rl_noise_horizon",   default=8,          type=int,
                        help="Number of noise steps predicted by the RL policy "
                             "(must match pi0_droid action_horizon=8).")
    parser.add_argument("--instruction",
                        default="pick up the peg and insert it vertically",
                        help="Language instruction passed to pi0_droid.")
    parser.add_argument("--checkpoint_path",    default="/opt/yingxi/pi0_droid",
                        help="Path to the pi0_droid checkpoint directory.")
    parser.add_argument("--dino_model",         default="facebook/dinov2-small",
                        help="HuggingFace DINOv2 model name.")
    # ── Sweep / early-stop arguments ────────────────────────────────────────────
    parser.add_argument("--eval_env_step_interval", default=-1, type=int,
                        help="Trigger eval + checkpoint every N env steps. "                             "-1 = disabled (use eval_interval instead).")
    parser.add_argument("--stop_success_rate",      default=0.95, type=float,
                        help="Stop training when the last --stop_window evals all "                             "reach this success rate. Only active when "                             "--eval_env_step_interval > 0.")
    parser.add_argument("--stop_window",            default=2, type=int,
                        help="Number of consecutive evals that must hit "                             "--stop_success_rate before training stops.")

    parser.add_argument("--dino_device",        default="auto",
                        help="Device for DINOv2 inference: auto | cpu | cuda | cuda:0 ...")

    # ── StateSACLearner hyperparameters (added to parser by parse_training_args) ─
    # All entries here become CLI flags AND are passed to StateSACLearner via
    # variant["train_kwargs"].  Do NOT add these again above.
    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        discount=0.99,
        tau=0.005,
        # Network architecture: Transformer (matches run_real_dino.sh defaults)
        hidden_dims=(1024, 1024, 1024),
        network_type="transformer",
        transformer_dim=256,
        transformer_depth=3,
        transformer_heads=4,
        transformer_mlp_dim=1024,
        transformer_dropout=0.0,
        # SAC-specific
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
