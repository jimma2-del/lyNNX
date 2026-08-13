"""Utilities for handling arrays or PyTrees with possible leading batch axes."""

from typing import Callable, Sequence, TypeVar, Any

import math

import numpy as np

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import chex

def shape_matches_excluding_batch_axes(unbatched_shape: Sequence[int], x_shape: Sequence[int]) -> bool:
    if len(unbatched_shape) == 0: return True
    if len(unbatched_shape) > len(x_shape): return False
    return x_shape[-len(unbatched_shape):] == unbatched_shape

def shape_is_batched(unbatched_shape: Sequence[int], x_shape: Sequence[int]) -> bool:
    return len(x_shape) > len(unbatched_shape) \
        and shape_matches_excluding_batch_axes(unbatched_shape, x_shape)

def is_batched(unbatched_shape: Sequence[int], x: jax.Array) -> bool:
    return shape_is_batched(unbatched_shape, x.shape)

def get_shape_batch_dims(unbatched_shape: Sequence[int], x_shape: Sequence[int]) -> Sequence[int]:
    if len(unbatched_shape) == 0: return x_shape
    assert shape_matches_excluding_batch_axes(unbatched_shape, x_shape)
    return x_shape[:-len(unbatched_shape)]

def get_batch_dims(unbatched_shape: Sequence[int], x: jax.Array) -> Sequence[int]:
    return get_shape_batch_dims(unbatched_shape, x.shape)

TInput = TypeVar('TInput')

def get_tree_batch_dims(unbatched_shapes_dtypes: TInput, x: TInput) -> Sequence[int]:
    chex.assert_trees_all_equal_structs(x, unbatched_shapes_dtypes,
        custom_message="'x' must have the same treedef (structure) as 'unbatched_shapes_dtypes'.")

    assert len(jax.tree.leaves(x)) != 0, "Trees cannot be empty."
    
    assert jax.tree.all(jax.tree.map(
        lambda shape_dtype, x_leaf: shape_matches_excluding_batch_axes(shape_dtype.shape, x_leaf.shape), 
        unbatched_shapes_dtypes, x
    )), "Leaves of 'x' must have the shape defined in the corresponding leaf of 'unbatched_shapes_dtypes'."

    batch_dims_shapes_dtypes = jax.tree.map(
        lambda unbatched_shape_dtype, x_leaf: jax.ShapeDtypeStruct(
            shape = get_batch_dims(unbatched_shape_dtype.shape, x_leaf),
            dtype = unbatched_shape_dtype.dtype
        ),
        unbatched_shapes_dtypes, x
    )

    batch_dims_shapes_dtypes_leaves = jax.tree.leaves(batch_dims_shapes_dtypes)

    assert all([ batch_dims_shape_dtype.shape == batch_dims_shapes_dtypes_leaves[0].shape 
            for batch_dims_shape_dtype in batch_dims_shapes_dtypes_leaves ] ), \
        "Leaf batch dimensions are not the same."

    return batch_dims_shapes_dtypes_leaves[0].shape

def flatten_batched_tree(unbatched_shapes_dtypes: TInput, x: TInput) -> jax.Array:
    """Flattens a batched PyTree into a single array while preserving leading batch axes.
    
    `unbatched_shapes_dtypes`: A PyTree of jax.ShapeDtypeStruct leaves, eg. from jax.eval_shape().
    `x`: A PyTree with the same structure as unbatched_shapes_dtypes and leaves with corresponding shapes, 
        but potentially with extra leading batch axes.
        
    Returns: jax.Array of shape (*batch_dims, total_flattened_size).
    """

    leaves_x = jax.tree.leaves(x)

    if not leaves_x:
        return jnp.array([]) # convention is to return a empty 1D array if empty PyTree

    batch_dims = get_tree_batch_dims(unbatched_shapes_dtypes, x)

    return jnp.concatenate([ leaf.reshape((*batch_dims, -1)) for leaf in leaves_x ], axis=-1)

def unflatten_batched_tree(unbatched_shapes_dtypes: TInput, arr: jax.Array) -> TInput:
    """Unflattens a batched PyTree from a single array while preserving leading batch axes.

    `unbatched_shapes_dtypes`: A PyTree of jax.ShapeDtypeStruct leaves, eg. from jax.eval_shape().
    `arr`: The flattened array to unflatten, potentially with extra leading batch axes.
        
    Returns: PyTree with structure/shape of `unbatched_shapes_dtypes`, but potentially with extra leading batch axes.
    """
    leaves_shape_dtype, treedef = jax.tree.flatten(unbatched_shapes_dtypes)

    sizes = [ math.prod(leaf.shape) for leaf in leaves_shape_dtype ]
    end_is = np.cumsum(sizes)

    assert arr.shape[-1] == end_is[-1], \
        f"Expected flattened (trailing) dimension of {end_is[-1]}, got {arr.shape[-1]}."

    flat_leaves = jnp.split(arr, end_is[:-1], axis=-1)
    leaves = [ leaf.reshape(arr.shape[:-1] + shape_dtype.shape) 
        for leaf, shape_dtype in zip(flat_leaves, leaves_shape_dtype) ]

    return jax.tree.unflatten(treedef, leaves)

def get_tree_flattened_dim(shapes_dtypes: Any) -> int:
    return jax.tree.reduce(
        lambda cum_len, shape_dtype: cum_len + math.prod(shape_dtype.shape), 
        shapes_dtypes, 0
    )

def flatten_batch_axes(unbatched_shape: Sequence[int], x: jax.Array):
    """Flattens leading batch axes into a singular batch axis. Adds a batch axis if none exist."""
    return jnp.reshape(x, (-1, *unbatched_shape))

def flatten_tree_batch_axes(unbatched_shapes_dtypes: TInput, x: TInput) -> TInput:
    """Flattens leading batch axes into a singular batch axis. Adds a batch axis if none exist."""
    return jax.tree.map(lambda x, s_dt: flatten_batch_axes(s_dt.shape, x), x, unbatched_shapes_dtypes)

def get_vmap_axis_size(*args, in_axes: int | None | Sequence[Any] = 0, **kwargs) -> int:
    """Finds what the axis size of the arguments would be if passed into `jax.vmap`."""

    f = jax.vmap(lambda *args, **kwargs: (args, kwargs), in_axes=in_axes, out_axes=0)
    out = jax.eval_shape(f, *args, **kwargs)
    return jax.tree.leaves(out)[0].shape[0]

def split_batched_keys(keys: chex.PRNGKey, num: int | Sequence[int] = 2) -> chex.PRNGKey:
    """Same as `jax.random.split`, but allows extra leading batch dims in `keys`.
        Split operation will be vmapped across batch dims."""
    if isinstance(num, int): num = (num,)
    split_keys = jax.vmap(jax.random.split, in_axes=(0, None))(keys.flatten(), num)
    return split_keys.reshape(num + keys.shape)

def split_key_from_batch(keys: chex.PRNGKey) -> chex.PRNGKey:
    """Splits a single new key from `keys` using an arbitrary element, replacing the used key in `keys`.
    If `keys` has shape (), this function is equivalent to `jax.random.split(keys)`.
    Returns: single key, keys with the same shape as `keys`."""

    if jnp.isscalar(keys): return jax.random.split(keys)

    indices = (0,) * len(keys.shape)
    key1, key2 = jax.random.split(keys[indices])
    
    return key1, keys.at[indices].set(key2)

def dummy_vmap(f: Callable) -> Callable:
    """Mimics the behavior of `jax.vmap`, but does not actually apply vmap transformation; instead, 
        for inputs, simply takes the first element in the batch axis,
        and for outputs, adds a dummy batch axis of length 1."""

    def dummy_vmapped(*args, **kwargs):
        unbatched_args, unbatched_kwargs = jax.tree.map(lambda x: x[0], (args, kwargs))
        return jax.tree.map(lambda x: x[None, ...], f(*unbatched_args, **unbatched_kwargs))

    return dummy_vmapped

def batched_index(arr: jax.Array, indices: jax.Array) -> tuple[jax.Array, ...]:
    """Converts an array containing a batch of indices into a tuple of indices along individual 
        axes of `arr`, ready for use in indexing into `arr`: `arr[batched_index(arr, indices)]`.

    `arr` can have leading batch dimensions if they match with the rightmost batch dimensions of `indices`.
        In this case, `indices` will select the corresponding batch slice from `arr`.

    `arr`: Array to index into, of shape `(arr_batch_dims, *arr_dims)`.

    `indices`: Array of shape `(*batch_dims, len(arr_dims))`; values along the last axis represent
        indices along the corresponding axis of `arr`, while axes on the left are batch axes.

    Returns: Tuple of length `len(arr.shape)`, where each item is an 
        array of indices along the corresponding axis of `arr`.
    """

    indices_batch_dims = indices.shape[:-1]
    n_indices_batch_dims = len(indices_batch_dims)

    n_arr_batch_dims = len(arr.shape) - indices.shape[-1]
    assert n_arr_batch_dims >= 0, (
        "`indices` contains too many indices: "
        f"{len(arr.shape)}-dimensional array cannot be indexed with {indices.shape[-1]} indices."
    )

    arr_batch_dims = arr.shape[:n_arr_batch_dims]
    indices_trailing_batch_dims = indices_batch_dims[n_indices_batch_dims - n_arr_batch_dims:]
    assert arr_batch_dims == indices_batch_dims[n_indices_batch_dims - n_arr_batch_dims:], (
        f"Batch dimensions of `arr` {arr_batch_dims} do not match with " 
        f"rightmost batch dimensions of `indices` {indices_trailing_batch_dims}."
    )

    extra_index_dims = []
    for i, size in enumerate(arr_batch_dims):
        shape = [1] * n_indices_batch_dims
        shape[n_indices_batch_dims - n_arr_batch_dims + i] = size
        extra_index_dims.append(jnp.arange(size).reshape(shape))

    return tuple(extra_index_dims) + tuple(jnp.moveaxis(indices, -1, 0))