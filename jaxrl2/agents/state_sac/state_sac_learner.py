"""State-only SAC learner for steering pi0 diffusion noise."""
import copy
import functools
import pathlib
from typing import Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import checkpoints
from flax.training import train_state
from typing import Any

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.state_sac.actor_updater import update_actor
from jaxrl2.agents.state_sac.critic_updater import update_critic
from jaxrl2.agents.state_sac.temperature import Temperature
from jaxrl2.agents.state_sac.temperature_updater import update_temperature
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.learned_std_normal_policy import LearnedStdTanhNormalPolicy
from jaxrl2.networks.learned_std_normal_policy import TransformerTanhNormalPolicy
from jaxrl2.networks.values import StateActionEnsemble
from jaxrl2.networks.values import StateActionTransformerEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update


class TrainState(train_state.TrainState):
    batch_stats: Any


@functools.partial(jax.jit, static_argnames=('critic_reduction',))
def _update_jit(
    rng: PRNGKey, actor: TrainState, critic: TrainState,
    target_critic_params: Params, temp: TrainState, batch: TrainState,
    tau: float, target_entropy: float, critic_reduction: str,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:
    key, rng = jax.random.split(rng)
    target_critic = critic.replace(params=target_critic_params)
    new_critic, critic_info = update_critic(
        key,
        actor,
        critic,
        target_critic,
        temp,
        batch,
        critic_reduction=critic_reduction,
    )
    new_target_critic_params = soft_target_update(new_critic.params, target_critic_params, tau)

    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(
        key,
        actor,
        new_critic,
        temp,
        batch,
        critic_reduction=critic_reduction,
    )
    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)

    return rng, new_actor, new_critic, new_target_critic_params, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info,
    }


class StateSACLearner(Agent):
    def __init__(self,
                 seed: int,
                 observations: Union[jnp.ndarray, DatasetDict],
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 critic_reduction: str = 'mean',
                 dropout_rate: Optional[float] = None,
                 init_temperature: float = 1.0,
                 num_qs: int = 2,
                 target_entropy: float = None,
                 action_magnitude: float = 1.0,
                 network_type: str = 'mlp',
                 transformer_dim: int = 256,
                 transformer_depth: int = 3,
                 transformer_heads: int = 4,
                 transformer_mlp_dim: int = 1024,
                 transformer_dropout: float = 0.0):
        self.action_dim = int(np.prod(actions.shape[-2:]))
        self.action_chunk_shape = actions.shape[-2:]
        self.discount = discount
        self.tau = tau
        self.critic_reduction = critic_reduction
        self.network_type = network_type

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0], hidden_dims[0])

        if network_type == 'mlp':
            policy_def = LearnedStdTanhNormalPolicy(
                hidden_dims,
                self.action_dim,
                dropout_rate=dropout_rate,
                low=-action_magnitude,
                high=action_magnitude,
            )
        elif network_type == 'transformer':
            if len(self.action_chunk_shape) != 2:
                raise ValueError(f"Transformer SAC expects 2D action chunks, got {self.action_chunk_shape}.")
            policy_def = TransformerTanhNormalPolicy(
                action_horizon=int(self.action_chunk_shape[0]),
                action_dim_per_step=int(self.action_chunk_shape[1]),
                transformer_dim=transformer_dim,
                transformer_depth=transformer_depth,
                transformer_heads=transformer_heads,
                transformer_mlp_dim=transformer_mlp_dim,
                dropout_rate=transformer_dropout,
                low=-action_magnitude,
                high=action_magnitude,
            )
        else:
            raise ValueError(f"Unsupported network_type: {network_type}")
        actor_params = policy_def.init(actor_key, observations)['params']
        actor = TrainState.create(
            apply_fn=policy_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
            batch_stats=None,
        )

        if network_type == 'mlp':
            critic_def = StateActionEnsemble(hidden_dims, num_qs=num_qs)
        else:
            critic_def = StateActionTransformerEnsemble(
                action_horizon=int(self.action_chunk_shape[0]),
                action_dim_per_step=int(self.action_chunk_shape[1]),
                transformer_dim=transformer_dim,
                transformer_depth=transformer_depth,
                transformer_heads=transformer_heads,
                transformer_mlp_dim=transformer_mlp_dim,
                dropout_rate=transformer_dropout,
                num_qs=num_qs,
            )
        critic_params = critic_def.init(critic_key, observations, actions)['params']
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
            batch_stats=None,
        )
        target_critic_params = copy.deepcopy(critic_params)

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
            batch_stats=None,
        )

        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._temp = temp
        if target_entropy is None or target_entropy == 'auto':
            self.target_entropy = -self.action_dim / 2
        else:
            self.target_entropy = float(target_entropy)
        print(f'target_entropy: {self.target_entropy}')
        print(self.critic_reduction)

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self._rng,
            self._actor,
            self._critic,
            self._target_critic_params,
            self._temp,
            batch,
            self.tau,
            self.target_entropy,
            self.critic_reduction,
        )

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        self._temp = new_temp
        return info

    @property
    def _save_dict(self):
        return {
            'critic': self._critic,
            'target_critic_params': self._target_critic_params,
            'actor': self._actor,
            'temp': self._temp,
        }

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).exists(), f"Checkpoint {dir} does not exist."
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)
        self._actor = output_dict['actor']
        self._critic = output_dict['critic']
        self._target_critic_params = output_dict['target_critic_params']
        self._temp = output_dict['temp']
        print('restored from ', dir)
