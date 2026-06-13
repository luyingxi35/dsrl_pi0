import argparse
import sys

from examples.train_real_dino import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs on wandb.')
    parser.add_argument('--eval_episodes', default=10, help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--env', default='franka_droid', help='name of environment')
    parser.add_argument('--log_interval', default=10000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=10000, help='checkpoint interval.', type=int)
    parser.add_argument('--batch_size', default=16, help='Mini batch size.', type=int)
    parser.add_argument('--max_steps', default=int(1e6), help='Number of training steps.', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the obervations', type=int)
    parser.add_argument('--wandb_project', default='cql_sim_online', help='wandb project')
    parser.add_argument('--num_initial_traj_collect', default=5, help='number of trajectories to collect before starting online updates', type=int)
    parser.add_argument('--algorithm', default='state_sac', help='type of algorithm')
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=1, help='Number of graident steps to take per environment step, aka UTD', type=int)
    parser.add_argument('--query_freq', default=-1, help='query frequency', type=int)
    parser.add_argument('--instruction', default='pick up the blue peg', help='language instruction for the robot')
    parser.add_argument('--restore_path', default='', help='optional checkpoint path to restore before real-world training')
    parser.add_argument('--resume_from', default='',
        help='Path to a previous outputdir to resume training from. Reuses that '
             'directory, auto-detects the latest agent checkpoint, '
             'training_state.json and replay_buffer.pkl. '
             'Mutually exclusive with --restore_path.')
    parser.add_argument('--policy_host', default='127.0.0.1', help='OpenPI policy server host')
    parser.add_argument('--policy_port', default=8000, help='OpenPI policy server port', type=int)
    parser.add_argument('--external_camera', default='right', choices=['left', 'right'], help='external camera feed to use for pi0 policy inputs')
    parser.add_argument('--use_wrist_camera', default=1, choices=(0, 1), help='whether pi0 policy inputs include wrist camera', type=int)
    parser.add_argument('--use_exterior_camera', default=0, choices=(0, 1), help='whether pi0 policy inputs include the selected exterior camera', type=int)
    parser.add_argument('--left_camera_id', default='', help='DROID left external camera ID')
    parser.add_argument('--right_camera_id', default='', help='DROID right external camera ID')
    parser.add_argument('--wrist_camera_id', required=True, help='DROID wrist camera ID')
    parser.add_argument('--max_rollout_steps', default=600, help='max robot-control steps per trajectory', type=int)
    parser.add_argument('--control_frequency_hz', default=15, help='target DROID control frequency for real rollouts', type=int)
    # ── Action execution (aligned with eval ExecutionConfig) ──────────────────
    parser.add_argument('--action_scale', default=1.0, type=float,
        help='Scale on DROID training max_joint_delta (0.2 rad/step). '
             '1.0 = full speed, 0.5 = half speed (safer default). Default: 1.0.')
    parser.add_argument('--max_joint_speed_rad_s', default=0.3, type=float,
        help='NUC-side per-joint speed cap (rad/s) forwarded to HighFreqController. '
             'Default 0.3 is conservative. Increase (e.g. 1.5) for faster execution. Default: 0.3.')
    parser.add_argument('--robot_action_latency', default=0.20, type=float,
        help='Arm command latency compensation (s): waypoint times are advanced by this '
             'amount so the robot arrives at the intended time. Default: 0.20.')
    parser.add_argument('--controller_frequency', default=200.0, type=float,
        help='HighFreqController loop rate (Hz) on the NUC. Default: 200.')
    parser.add_argument('--dino_model', default='facebook/dinov2-small', help='HuggingFace DINO-v2 model name')
    parser.add_argument('--dino_device', default='auto', help='DINO device: auto, cpu, cuda, cuda:0, etc.')
    parser.add_argument('--rl_noise_horizon', default=8, help='Full pi0 noise horizon predicted by the RL policy.', type=int)

    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(1024, 1024, 1024),
        network_type='transformer',
        transformer_dim=256,
        transformer_depth=3,
        transformer_heads=4,
        transformer_mlp_dim=1024,
        transformer_dropout=0.0,
        discount=0.99,
        tau=0.005,
        critic_reduction='min',
        dropout_rate=0.0,
        target_entropy=0.0,
        num_qs=2,
        action_magnitude=2.0,
    )

    variant, args = parse_training_args(train_args_dict, parser)
    print(variant)
    main(variant)
    sys.exit()
