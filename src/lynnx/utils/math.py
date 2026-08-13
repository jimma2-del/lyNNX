"""Math functions."""

import jax
import jax.numpy as jnp

def inv_softplus(x: jax.Array) -> jax.Array:
    THRESHOLD = 20
    safe_x = jnp.where(x > THRESHOLD, 1.0, x)
        # extra sanitization due to issue with jnp.where and NaNs for gradients
    return jnp.where(x > THRESHOLD, x, jnp.log(jnp.exp(safe_x) - 1))

def normal_entropy(sigma):
  return (jnp.log(2 * jnp.pi * sigma**2) + 1) / 2