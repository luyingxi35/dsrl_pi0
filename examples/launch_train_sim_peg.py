"""CLI entry point for train_sim_peg.py (PegInsertionVertical in LIBERO)."""
import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so 'examples' package is importable
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.train_sim_peg import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DSRL sim training: PegInsertionVertical (LIBERO) + PixelSAC + pi0')

    parser.add_argument('--seed',                default=42,           type=int)
    parser.add_argument('--launch_group_id',     default='')
    parser.add_argument('--eval_episodes',       default=10,           type=int)
    parser.add_argument('--env',                 default='libero_peg',
                        help='libero_peg | libero | aloha_cube')
    parser.add_argument('--log_interval',        default=500,          type=int)
    parser.add_argument('--eval_interval',       default=10000,        type=int)
    parser.add_argument('--checkpoint_interval', default=-1,           type=int)
    parser.add_argument('--batch_size',          default=256,          type=int)
    parser.add_argument('--max_steps',           default=500_000,      type=int)
    parser.add_argument('--add_states',          default=1,            type=int)
    parser.add_argument('--wandb_project',       default='DSRL_pi0_PegLibero')
    parser.add_argument('--start_online_updates', default=500,         type=int)
    parser.add_argument('--algorithm',           default='pixel_sac')
    parser.add_argument('--prefix',              default='')
    parser.add_argument('--suffix',              default='')
    parser.add_argument('--multi_grad_step',     default=20,           type=int)
    parser.add_argument('--resize_image',        default=64,           type=int)
    parser.add_argument('--query_freq',          default=20,           type=int)

    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(128, 128, 128),
        cnn_features=(32, 32, 32, 32),
        cnn_strides=(2, 1, 1, 1),
        cnn_padding='VALID',
        latent_dim=50,
        discount=0.999,
        tau=0.005,
        critic_reduction='mean',
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type='small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy='auto',
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=1,
    )

    variant, args = parse_training_args(train_args_dict, parser)
    print(variant)
    main(variant)
    sys.exit()
