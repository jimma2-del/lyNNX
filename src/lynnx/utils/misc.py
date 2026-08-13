"""Miscellaneous utilities."""

from typing import Sequence, TypeVar

import jax
import jax.numpy as jnp

TInput = TypeVar('TInput')

def compacting_mask(items: TInput, mask: jax.Array) -> tuple[TInput, jax.Array]:
    """Takes masked items from array, then compacts the resulting array so that empty items are removed.

    `mask` is assumed to act on the leading axes of `items`, preserving additional trailing axes.
    
    Returns: 
        A flattened array of `mask.size` items. The first `sum(mask)` items will be taken items, 
            with the remaining space being empty padding to ensure a fixed size output.
        An array of indices with the same shape as `mask`. 
            For taken items, the indices will point to the item's position in the new, compacted array.
            For discarded items, the indices will take the sentinel value of `mask.size`.
    """

    flat_mask = jnp.ravel(mask)
    flat_items = jax.tree.map(lambda x: jnp.reshape(x, (-1, *x.shape[mask.ndim:])), items)

    flat_indices = jnp.where(flat_mask, jnp.cumsum(flat_mask) - 1, len(flat_mask))
    compacted = jax.tree.map(lambda x: jnp.zeros_like(x).at[flat_indices].set(x, mode='drop'), flat_items)
    
    indices = jax.tree.map(lambda x: jnp.reshape(x, mask.shape), flat_indices) 

    return compacted, indices

if __name__ == "__main__":
    items = jnp.stack((jnp.arange(0, 4), jnp.arange(4, 8)))
    mask = jnp.ones_like(items)

    print(items)

    comp, comp_is = compacting_mask(items, mask)
    print(comp)
    print(comp_is)

    print(comp[comp_is])