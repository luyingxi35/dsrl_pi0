import flax.linen as nn
import jax.numpy as jnp


def default_init(scale: float = 2 ** 0.5):
    return nn.initializers.orthogonal(scale)

def xavier_init():
    return nn.initializers.xavier_normal()

def kaiming_init():
    return nn.initializers.kaiming_normal()