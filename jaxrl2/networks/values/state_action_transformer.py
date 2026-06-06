from typing import Callable

import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.constants import default_init
from jaxrl2.networks.mlp import _flatten_dict
from jaxrl2.networks.transformer_blocks import TransformerBlock


class StateActionTransformerValue(nn.Module):
    action_horizon: int
    action_dim_per_step: int
    transformer_dim: int = 256
    transformer_depth: int = 3
    transformer_heads: int = 4
    transformer_mlp_dim: int = 1024
    dropout_rate: float = 0.0
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu

    def _reshape_actions(self, actions: jnp.ndarray, batch_size: int) -> jnp.ndarray:
        expected_flat_dim = self.action_horizon * self.action_dim_per_step
        if actions.shape[-2:] == (self.action_horizon, self.action_dim_per_step):
            return actions
        if actions.shape[-1] == expected_flat_dim:
            return jnp.reshape(actions, (batch_size, self.action_horizon, self.action_dim_per_step))
        raise ValueError(
            "Expected actions with shape "
            f"(batch, {self.action_horizon}, {self.action_dim_per_step}) "
            f"or (batch, {expected_flat_dim}), got {actions.shape}."
        )

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 training: bool = False):
        state = _flatten_dict(observations)
        batch_size = state.shape[0]
        action_tokens_raw = self._reshape_actions(actions, batch_size)

        state_token = nn.Dense(self.transformer_dim, name="state_proj")(state)[:, None, :]
        action_tokens = nn.Dense(self.transformer_dim, name="action_proj")(action_tokens_raw)
        tokens = jnp.concatenate([state_token, action_tokens], axis=1)

        pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.action_horizon + 1, self.transformer_dim),
        )
        tokens = tokens + pos_embedding

        for i in range(self.transformer_depth):
            tokens = TransformerBlock(
                hidden_dim=self.transformer_dim,
                num_heads=self.transformer_heads,
                mlp_dim=self.transformer_mlp_dim,
                dropout_rate=self.dropout_rate,
                activations=self.activations,
                name=f"block_{i}",
            )(tokens, training=training)

        state_token = nn.LayerNorm(name="state_norm")(tokens[:, 0, :])
        q = nn.Dense(1, kernel_init=default_init(1e-2), name="q_head")(state_token)
        return jnp.squeeze(q, -1)


class StateActionTransformerEnsemble(nn.Module):
    action_horizon: int
    action_dim_per_step: int
    transformer_dim: int = 256
    transformer_depth: int = 3
    transformer_heads: int = 4
    transformer_mlp_dim: int = 1024
    dropout_rate: float = 0.0
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    num_qs: int = 2

    @nn.compact
    def __call__(self, states, actions, training: bool = False):
        VmapCritic = nn.vmap(
            StateActionTransformerValue,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        return VmapCritic(
            self.action_horizon,
            self.action_dim_per_step,
            transformer_dim=self.transformer_dim,
            transformer_depth=self.transformer_depth,
            transformer_heads=self.transformer_heads,
            transformer_mlp_dim=self.transformer_mlp_dim,
            dropout_rate=self.dropout_rate,
            activations=self.activations,
        )(states, actions, training)
