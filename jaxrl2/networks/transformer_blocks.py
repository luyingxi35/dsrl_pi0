from typing import Callable

import flax.linen as nn
import jax.numpy as jnp


class TransformerBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.0
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        deterministic = not training

        x = nn.LayerNorm()(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=False,
            deterministic=deterministic,
            kernel_init=nn.initializers.xavier_uniform(),
            force_fp32_for_softmax=True,
        )(x, x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = inputs + x

        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim)(y)
        y = self.activations(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        y = nn.Dense(self.hidden_dim)(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        return x + y
