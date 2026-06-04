from typing import Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks import MLP
from jaxrl2.networks.constants import default_init
from jaxrl2.networks.mlp import _flatten_dict
from jaxrl2.networks.transformer_blocks import TransformerBlock

class LearnedStdNormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        return distribution

class TanhMultivariateNormalDiag(distrax.Transformed):

    def __init__(self,
                 loc: jnp.ndarray,
                 scale_diag: jnp.ndarray,
                 low: Optional[jnp.ndarray] = None,
                 high: Optional[jnp.ndarray] = None):
        distribution = distrax.MultivariateNormalDiag(loc=loc,
                                                      scale_diag=scale_diag)

        layers = []

        if not (low is None or high is None):

            def rescale_from_tanh(x):
                x = (x + 1) / 2  # (-1, 1) => (0, 1)
                return x * (high - low) + low

            def forward_log_det_jacobian(x):
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            layers.append(
                distrax.Lambda(
                    rescale_from_tanh,
                    forward_log_det_jacobian=forward_log_det_jacobian,
                    event_ndims_in=1,
                    event_ndims_out=1))

        layers.append(distrax.Block(distrax.Tanh(), 1))

        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

class LearnedStdTanhNormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    low: Optional[float] = None
    high: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = TanhMultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds), low=self.low, high=self.high)
        return distribution


class TransformerTanhNormalPolicy(nn.Module):
    action_horizon: int
    action_dim_per_step: int
    transformer_dim: int = 256
    transformer_depth: int = 3
    transformer_heads: int = 4
    transformer_mlp_dim: int = 1024
    dropout_rate: float = 0.0
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    low: Optional[float] = None
    high: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        state = _flatten_dict(observations)
        batch_size = state.shape[0]

        state_token = nn.Dense(self.transformer_dim, name="state_proj")(state)[:, None, :]
        action_queries = self.param(
            "action_queries",
            nn.initializers.normal(stddev=0.02),
            (1, self.action_horizon, self.transformer_dim),
        )
        action_queries = jnp.broadcast_to(
            action_queries,
            (batch_size, self.action_horizon, self.transformer_dim),
        )

        tokens = jnp.concatenate([state_token, action_queries], axis=1)
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
                name=f"block_{i}",
            )(tokens, training=training)

        action_tokens = nn.LayerNorm(name="action_norm")(tokens[:, 1:, :])
        means = nn.Dense(
            self.action_dim_per_step,
            kernel_init=default_init(1e-2),
            name="mean_head",
        )(action_tokens)
        log_stds = nn.Dense(
            self.action_dim_per_step,
            kernel_init=default_init(1e-2),
            name="log_std_head",
        )(action_tokens)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        means = jnp.reshape(means, (batch_size, self.action_horizon * self.action_dim_per_step))
        log_stds = jnp.reshape(log_stds, (batch_size, self.action_horizon * self.action_dim_per_step))
        return TanhMultivariateNormalDiag(
            loc=means,
            scale_diag=jnp.exp(log_stds),
            low=self.low,
            high=self.high,
        )
