import numpy as np
from flax.core import freeze

from jaxrl2.agents.state_sac.state_sac_learner import StateSACLearner
from jaxrl2.utils.noise_utils import make_full_horizon_noise


STATE_DIM = 2440
HORIZON = 8
ACTION_DIM = 32


def _make_observations(batch_size=2):
    return {
        "state": np.zeros((batch_size, STATE_DIM, 1), dtype=np.float32),
    }


def _make_agent(batch_size=2):
    observations = _make_observations(batch_size)
    actions = np.zeros((batch_size, HORIZON, ACTION_DIM), dtype=np.float32)
    return StateSACLearner(
        0,
        observations,
        actions,
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(64, 64, 64),
        network_type="transformer",
        transformer_dim=32,
        transformer_depth=1,
        transformer_heads=4,
        transformer_mlp_dim=64,
        transformer_dropout=0.0,
        target_entropy=0.0,
        action_magnitude=2.0,
        num_qs=2,
    )


def test_transformer_actor_outputs_flat_full_horizon_noise():
    agent = _make_agent()
    actions = agent.eval_actions(_make_observations())

    assert actions.shape == (2, HORIZON * ACTION_DIM)
    actions_noise = actions.reshape(2, HORIZON, ACTION_DIM)
    assert actions_noise.shape == (2, HORIZON, ACTION_DIM)
    assert np.max(actions) <= 2.0 + 1e-5
    assert np.min(actions) >= -2.0 - 1e-5


def test_transformer_critic_accepts_sequence_and_flat_actions():
    agent = _make_agent()
    observations = _make_observations()
    sequence_actions = np.zeros((2, HORIZON, ACTION_DIM), dtype=np.float32)
    flat_actions = sequence_actions.reshape(2, HORIZON * ACTION_DIM)

    qs_sequence = agent._critic.apply_fn({"params": agent._critic.params}, observations, sequence_actions)
    qs_flat = agent._critic.apply_fn({"params": agent._critic.params}, observations, flat_actions)

    assert qs_sequence.shape == (2, 2)
    assert qs_flat.shape == (2, 2)


def test_transformer_state_sac_update_smoke():
    agent = _make_agent()
    observations = _make_observations()
    batch = freeze({
        "observations": observations,
        "next_observations": _make_observations(),
        "actions": np.zeros((2, HORIZON, ACTION_DIM), dtype=np.float32),
        "rewards": np.zeros((2,), dtype=np.float32),
        "masks": np.ones((2,), dtype=np.float32),
        "discount": np.full((2,), 0.99, dtype=np.float32),
    })

    info = agent.update(batch)

    assert "actor_loss" in info
    assert "critic_loss" in info
    assert "entropy" in info


def test_full_horizon_noise_formatter_does_not_repeat_tail():
    flat_actions = np.arange(HORIZON * ACTION_DIM, dtype=np.float32)
    actions_noise, pi0_noise = make_full_horizon_noise(flat_actions, (HORIZON, ACTION_DIM))

    assert actions_noise.shape == (HORIZON, ACTION_DIM)
    assert pi0_noise.shape == (1, HORIZON, ACTION_DIM)
    np.testing.assert_array_equal(actions_noise[-1], flat_actions.reshape(HORIZON, ACTION_DIM)[-1])

    batched_flat = flat_actions[np.newaxis]
    actions_noise, pi0_noise = make_full_horizon_noise(batched_flat, (HORIZON, ACTION_DIM))
    assert actions_noise.shape == (HORIZON, ACTION_DIM)
    assert pi0_noise.shape == (1, HORIZON, ACTION_DIM)
