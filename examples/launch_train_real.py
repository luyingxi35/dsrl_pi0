import argparse
import sys
from examples.train_real import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs on wandb.')
    parser.add_argument('--eval_episodes', default=10,help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--env', default='libero', help='name of environment')
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=-1, help='checkpoint interval.', type=int)
    parser.add_argument('--batch_size', default=16, help='Mini batch size.', type=int)
    parser.add_argument('--max_steps', default=int(1e6), help='Number of training steps.', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the obervations', type=int)
    parser.add_argument('--wandb_project', default='cql_sim_online', help='wandb project')
    parser.add_argument('--num_initial_traj_collect', default=1, help='number of trajectories to collect before starting online updates', type=int)
    parser.add_argument('--algorithm', default='pixel_sac', help='type of algorithm')
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=1, help='Number of graident steps to take per environment step, aka UTD', type=int)
    parser.add_argument('--resize_image', default=-1, help='the size of image if need resizing', type=int)
    parser.add_argument('--query_freq', default=-1, help='query frequency', type=int)
    parser.add_argument('--instruction', default='put the spoon on the plate', help='language instruction for the robot')
    parser.add_argument('--restore_path', default='', help='optional checkpoint path to restore before real-world training')
    parser.add_argument('--policy_host', default='127.0.0.1', help='OpenPI policy server host')
    parser.add_argument('--policy_port', default=8000, help='OpenPI policy server port', type=int)
    parser.add_argument('--external_camera', default='right', choices=['left', 'right'], help='external camera feed to use for policy inputs')
    parser.add_argument('--left_camera_id', required=True, help='DROID left external camera ID')
    parser.add_argument('--right_camera_id', required=True, help='DROID right external camera ID')
    parser.add_argument('--wrist_camera_id', required=True, help='DROID wrist camera ID')
    parser.add_argument('--max_rollout_steps', default=200, help='max robot-control steps per trajectory', type=int)
    parser.add_argument('--control_frequency_hz', default=15, help='target DROID control frequency for real rollouts', type=int)
    
    # The hyperparameters for the real robot experiments
    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr= 3e-4,
        temp_lr=3e-4,
        hidden_dims= (1024, 1024, 1024),
        cnn_features= (32, 32, 32, 32),
        cnn_strides= (3, 2, 2, 2),
        cnn_padding= 'VALID',
        latent_dim= 50,
        discount= 0.99,
        tau= 0.005,
        critic_reduction = 'min',
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type='small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy=0.0,
        num_qs=2,
        action_magnitude=2.0,
        num_cameras=3,
        )

    variant, args = parse_training_args(train_args_dict, parser)
    print(variant)
    main(variant)
    sys.exit()
    
