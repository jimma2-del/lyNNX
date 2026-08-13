import numpy as np

import jax.numpy as jnp
from jax.typing import ArrayLike
import jax

from lynnx.utils.batch_utils import batched_index, shape_matches_excluding_batch_axes

class LinearlyInterpolatedTable:
    """Linearly interpolates values using nearby grid points.
    Can also extrapolate values if outside of the grid, though this is usually highly inaccurate.

    Supports batched positions -- getting/setting values at multiple positions at once.
    """

    def __init__(self, min: ArrayLike, max: ArrayLike, step: ArrayLike):
        self.min = np.asarray(min)
        self.max = np.asarray(max)
        self.step = np.asarray(step)

        assert self.min.shape == self.max.shape and self.min.shape == self.step.shape

        self.shape = ()

        for i in range(len(min)):
            self.shape = (*self.shape, 
                int((max[i] - min[i]) // step[i] + 1 + ((max[i] - min[i]) % step[i] != 0))) 
                # add an extra if not perfectly ending on max

    def init(self, init: ArrayLike) -> jax.Array:
        return jnp.full(self.shape, init, dtype=jnp.float32)

    def get(self, data: jax.Array, pos: ArrayLike) -> jax.Array:
        """Get a linearly interpolated value at a position."""
        assert shape_matches_excluding_batch_axes(self.shape, data.shape)
        assert pos.shape[-1] == len(self.shape)

        lower_indices, offsets = self.get_lower_indices_and_offsets(pos)
        corner_indices, weights = self.get_corner_indices_and_weights(lower_indices, offsets)

        # move corner axis to the front
        corner_indices = jnp.moveaxis(corner_indices, -2, 0)
        weights = jnp.moveaxis(weights, -1, 0)

        return jnp.sum(data[batched_index(data, corner_indices)] * weights, axis=0)

    def get_corner_adjustments(
        self, data: jax.Array, pos: ArrayLike, adjust_amount: ArrayLike
    ) -> tuple[jax.Array, jax.Array]:
        """Adjust values at gridpoints to adjust the linearly interpolated value at the position.
        Get the adjustment needed for each gridpoint.

        Returns: 
            corner positions: (*batch_dims, corner_dim, pos_dim) 
            corner adjustment values: (*batch_dims, corner_dim)
        """
        assert shape_matches_excluding_batch_axes(self.shape, data.shape)
        assert pos.shape[-1] == len(self.shape)

        lower_indices, offsets = self.get_lower_indices_and_offsets(pos)
        corner_indices, weights = self.get_corner_indices_and_weights(lower_indices, offsets)

        sum_squared_weights = jnp.sum(weights**2, axis=-1)
        adjust_amounts = weights * adjust_amount[..., None] / sum_squared_weights[..., None]

        return corner_indices, adjust_amounts

    def adjust(self, data: jax.Array, pos: ArrayLike, adjust_amount: ArrayLike) -> jax.Array:
        """Adjust values at gridpoints to adjust the linearly interpolated value at the position."""
        corner_indices, adjust_amounts = self.get_corner_adjustments(data, pos, adjust_amount)

        # move corner axis to the front
        corner_indices = jnp.moveaxis(corner_indices, -2, 0)
        adjust_amounts = jnp.moveaxis(adjust_amounts, -1, 0)

        return data.at[batched_index(data, corner_indices)].add(adjust_amounts)

    def set(self, data: jax.Array, pos: ArrayLike, value: ArrayLike) -> jax.Array:
        """Adjust values at gridpoints to set the linearly interpolated value at the position."""
        cur_value = self.get(data, pos)
        return self.adjust(data, pos, value - cur_value)
    
    def get_lower_indices_and_offsets(self, pos: ArrayLike) -> tuple[jax.Array, jax.Array]:
        """Returns: (*batch_dims, pos_dim), (*batch_dims, pos_dim)."""
        assert pos.shape[-1] == len(self.shape)

        indices = ((pos - self.min) // self.step).astype(int)
        indices = jnp.clip(indices, 0, jnp.array(self.shape) - 2)
            # clip to inside bounds; use extrapolation for those cases

        offsets = (pos - (indices*self.step + self.min)) / self.step
            # offset from lower_index, in units of indices (normalized by step size)

        return indices, offsets

    def get_corner_indices_and_weights(self, lower_indices: ArrayLike, offsets: ArrayLike, chosen_indices=None, weight=None):
        """Args: (*batch_dims, pos_dim), (*batch_dims, pos_dim), (*batch_dims, pos_dim), (*batch_dims).
        Returns: (*batch_dims, corner_dim, pos_dim), (*batch_dims, corner_dim)."""

        assert offsets.shape[-1] == len(self.shape)

        batch_dims = lower_indices.shape[:-1]
        if chosen_indices is None: chosen_indices = jnp.zeros((*batch_dims, 0), dtype=jnp.int32)
        if weight is None: weight = jnp.ones(batch_dims, dtype=jnp.float32)
        
        n_indices_chosen = chosen_indices.shape[-1]
        if n_indices_chosen == len(self.shape):
            return chosen_indices[..., None, :], weight[..., None]

        next_lower_i = lower_indices[..., n_indices_chosen, None]
        next_offset = offsets[..., n_indices_chosen]
        
        indices1, weights1 = self.get_corner_indices_and_weights(lower_indices, offsets, 
            jnp.concatenate((chosen_indices, next_lower_i), axis=-1), weight * (1-next_offset))
        indices2, weights2 = self.get_corner_indices_and_weights(lower_indices, offsets, 
            jnp.concatenate((chosen_indices, next_lower_i + 1), axis=-1), weight * next_offset)

        return jnp.concatenate((indices1, indices2), axis=-2), jnp.concatenate((weights1, weights2), axis=-1)

if __name__ == "__main__":
    #table = LinearlyInterpolatedTable((0,1), (9,11), (2,3))
    table = LinearlyInterpolatedTable((0,1), (9,11), (0.5,0.5))
    table_state = table.init(1)

    print(table_state)

    #table_state = table_state.at[1,1].set(5)

    key = jax.random.key(0)

    for i in range(100):
        key, subkey1, subkey2, = jax.random.split(key, 3)
        coords = jax.random.uniform(subkey1, 2, minval=jnp.array((-10,-10)), maxval=jnp.array((11,20)))
        set_val = jax.random.uniform(subkey2, (), minval=-10, maxval=10)

        table_state = table.set(table_state, coords, set_val)

        if abs(table.get(table_state, coords) - set_val) > 0.001:
            print(set_val, table.get(table_state, coords))

    print(table_state)

    BATCH_DIMS = (10, 3)
    ADJUST_RATE = 1/np.prod(BATCH_DIMS)

    key, subkey1, subkey2, = jax.random.split(key, 3)
    coords = jax.random.uniform(subkey1, (*BATCH_DIMS, 2), minval=jnp.array((2,2)), maxval=jnp.array((8,8)))
    correct_vals = jax.random.uniform(subkey2, BATCH_DIMS, minval=-10, maxval=10)

    for i in range(1000):
        table_state = table.adjust(table_state, coords, (correct_vals - table.get(table_state, coords)) * ADJUST_RATE)
        print(jnp.sum((table.get(table_state, coords) - correct_vals) ** 2))

    print(table.get(table_state, coords) - correct_vals)

    print(table_state)

    #print(table.get(table_state, jnp.array((-1, 2.5))))