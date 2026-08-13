"""Buffers, for use as replay buffers."""

from typing import Any, Generic, TypeVar, Self, Sequence

import math

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import chex
import flax.struct

from lynnx.utils.batch_utils import get_tree_batch_dims, get_vmap_axis_size
from lynnx.utils.misc import compacting_mask

TBufferItem = TypeVar("TBufferItem")

@flax.struct.dataclass
class CircularBuffer(Generic[TBufferItem]):
    """Container which discards earliest inserted items upon reaching maximum capacity.
    
    Sequential order is kept intact, except in the transition from the last element to the first element.
        Multiple circular buffers can be kept in parallel with `batch_dims`.
    """

    item_shapes_dtypes: TBufferItem = flax.struct.field(pytree_node=False)
    length: int = flax.struct.field(pytree_node=False)

    data: TBufferItem # shape (length, *batch_dims, *item_shape_dtype.shape)

    batch_dims: Sequence[int] = flax.struct.field(pytree_node=False, default=())

    start_i: ArrayLike = flax.struct.field(default_factory=lambda: jnp.array(0))
        # inclusive except when filled_len == 0
    filled_len: ArrayLike = flax.struct.field(default_factory=lambda: jnp.array(0))

    @classmethod
    def init(cls, item_shapes_dtypes: TBufferItem, length: int, 
            batch_dims: int | Sequence[int] = ()) -> Self:
        """Initialize an empty CircularBuffer.

        `item_shapes_dtypes`: A PyTree of jax.ShapeDtypeStruct leaves, eg. from jax.eval_shape().
        `length`: Maximum number of items per parallel circular buffer.
        `batch_dims`: Dimensions of circular buffers kept in parallel.
        """

        if isinstance(batch_dims, int):
            batch_dims = (batch_dims,)

        data = jax.tree.map(
            lambda shape_dtype: jnp.empty((length, *batch_dims, *shape_dtype.shape), dtype=shape_dtype.dtype),
            item_shapes_dtypes
        )

        return cls(
            item_shapes_dtypes = item_shapes_dtypes,
            length = length,
            data = data,
            batch_dims = batch_dims,
        )

    def index(self, position: ArrayLike, include_empty: bool = False) -> jax.Array:
        """`position`: 0 means oldest inserted item."""
        position += self.start_i
        if include_empty: position += self.filled_len
        return position % self.length

    def get(self, position: ArrayLike, include_empty: bool = False) -> TBufferItem:
        """Get an item at a position in the circular buffer, 
            with position 0 being the earliest inserted item.
        
        `include_empty`: If True, considers all slots in the buffer as filled with items.
        """
        index = self.index(position, include_empty=include_empty)
        return jax.tree.map(lambda leaf: leaf[index], self.data)

    def set(self, position: ArrayLike, item: TBufferItem, include_empty: bool = False) -> Self:
        """Set an item at a position in the circular buffer,
            with position 0 being the earliest inserted item.

        Returns a copy of the buffer with the item set.
        
        `include_empty`: If True, considers all slots in the buffer as filled with items.
        """
        index = self.index(position, include_empty=include_empty)
        return self.replace(data=jax.tree.map(lambda d, i: d.at[index].set(i), self.data, item))

    def insert(self, items: TBufferItem) -> Self:
        """Insert an array of items into the circular buffer,
            removing the earliest inserted items if the buffer is full.

        `items`: Array of items with shape (seq_len, *batch_dims, *item_shape_dtype.shape).
        """

        if len(get_tree_batch_dims(self.item_shapes_dtypes, items)) < len(self.batch_dims) + 1:
            items = jax.tree.map(lambda x: x[None, ...], items) # single item -> add batch axis

        insert_n = get_vmap_axis_size(items)
        insert_is = jnp.arange(insert_n)

        return self.set(insert_is, items, include_empty=True).insert_update_pointers(insert_n)

    def insert_update_pointers(self, num: ArrayLike) -> Self:
        end_i = self.index(num, include_empty=True)
        filled_len = jnp.minimum(self.filled_len + num, self.length)
        start_i = jnp.where(filled_len == self.length, end_i, self.start_i)

        return self.replace(start_i=start_i, filled_len=filled_len)

    def remove(self, num: int, include_empty: bool = False) -> Self:
        """Remove the specified number of items, starting from the earliest inserted item.
        
        Returns a copy of the buffer with the items marked as removed.

        `include_empty`: If True, considers all slots in the buffer as filled with items.
        """
        if include_empty:
            num = jnp.maximum(0, num - (self.length - self.filled_len))
        else:
            num = jnp.minimum(num, self.filled_len)

        return self.replace(start_i=self.index(num), filled_len=self.filled_len - num)

    def sample(self, key: chex.PRNGKey, seq_len: int | None = None, 
            batch_dims: int | Sequence[int] = ()) -> TBufferItem:
        """Sample a batch of items from the circular buffer with uniform probability.

        `seq_length`: If provided, samples sequences (i, i+1, i+2, ...) instead of single samples.
        
        Returns: An array of samples of shape (*batch_dims, *item_shape_dtype.shape)
            or (*batch_dims, seq_len, *item_shape_dtype.shape) if `seq_len` is specified.
        """
        batch_is_key, seq_is_key = jax.random.split(key)

        if isinstance(batch_dims, int):
            batch_dims = (batch_dims,)

        batch_is = jax.random.randint(batch_is_key, (*batch_dims, len(self.batch_dims)), 
            minval=jnp.zeros(len(self.batch_dims)), maxval=jnp.array(self.batch_dims))
        batch_is = jnp.moveaxis(batch_is, -1, 0)

        maxval = 1 + self.filled_len - (1 if seq_len is None else seq_len)
        seq_is = jax.random.randint(seq_is_key, batch_dims, minval=0, maxval=maxval)

        if seq_len is not None:
            seq_is = (seq_is[..., None] + jnp.arange(seq_len))
            batch_is = batch_is[..., None]

        indices = (self.index(seq_is), *batch_is)
        return jax.tree.map(lambda x: x[indices], self.data)

TOptionalData = TypeVar("TOptionalData")

@chex.dataclass
class CircularBufferWithOptionalData(Generic[TBufferItem, TOptionalData]):
    """Circular buffer which allows items to optionally contain extra data.
    
    This class saves memory as it only allocates space for optional data for items that actually have optional data. 
        Thus, `optional_data_capacity` can be set to be a fraction of the total number of items, 
            depending on the expected fraction of items with optional data.
        The drawback is that if `optional_data_capacity` is exceeded, further items with optional data will have their
            optional data DISCARDED. Do NOT use this class if it is integral that optional data is ALWAYS kept.
    """

    main_buffer: CircularBuffer[tuple[TBufferItem, jax.Array, jax.Array]] 
        # tuple ( main item, has optional data, optional data index )
    optional_data_buffer: CircularBuffer[TOptionalData]

    @classmethod
    def init(cls, 
        main_shapes_dtypes: TBufferItem, 
        optional_data_shapes_dtypes: TBufferItem, 
        main_length: int, 
        optional_data_frac: float | None = None,
        optional_data_capacity: int | None = None,
        batch_dims: int | Sequence[int] = ()
    ) -> Self:
        """Initialize an empty CircularBuffer.

        `main_shapes_dtypes`: A PyTree of jax.ShapeDtypeStruct leaves, eg. from jax.eval_shape().
        `optional_data_shapes_dtypes`: A PyTree of jax.ShapeDtypeStruct leaves, eg. from jax.eval_shape().
        `main_length`: Maximum number of main items per parallel circular buffer.
        `optional_data_frac`: Maximum fraction of main items with optional data.
        `optional_data_capacity`: Total maximum number optional data items stored.
        `batch_dims`: Dimensions of circular buffers kept in parallel.
        """

        assert optional_data_frac is None or optional_data_capacity is None, \
            "Specify either `optional_data_frac` or `optional_data_capacity`, not both."
        assert optional_data_frac is not None or optional_data_capacity is not None, \
            "One of `optional_data_frac` or `optional_data_capacity` must be specified."

        if isinstance(batch_dims, int):
            batch_dims = (batch_dims,)

        if optional_data_capacity is None:
            optional_data_capacity = math.ceil(math.prod(batch_dims) * main_length * optional_data_frac)
        
        if optional_data_capacity == 0: 
            optional_data_capacity = 1

        item_shapes_dtypes = (
            main_shapes_dtypes, 
            jax.ShapeDtypeStruct(shape=(), dtype=jnp.bool),
            jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32),
        )

        main_buffer = CircularBuffer.init(item_shapes_dtypes, length=main_length, batch_dims=batch_dims)
        main_buffer = main_buffer.replace( # ensure has_optional_data is marked False by default
            data=(main_buffer.data[0], jnp.zeros_like(main_buffer.data[1]), main_buffer.data[2]))

        optional_data_buffer = CircularBuffer.init(optional_data_shapes_dtypes, length=optional_data_capacity)

        return cls(main_buffer=main_buffer, optional_data_buffer=optional_data_buffer)

    def get(self, position: ArrayLike) -> tuple[TBufferItem, jax.Array, TOptionalData]:
        """Get an item at a position in the circular buffer and its corresponding optional data,
            with position 0 being the earliest inserted item.
        
        `include_empty`: If True, considers all slots in the buffer as filled with items.

        Returns: 
            an array of samples
            a mask indicating if each item has optional data
            an array of the samples' corresponding optional data
        """
        main_data, has_opt, opt_i = self.main_buffer.get(position)
        return main_data, has_opt, jax.tree.map(lambda x: x[opt_i], self.optional_data_buffer.data)

    def insert(self, items: TBufferItem, has_optional_data_mask: ArrayLike, optional_data: TOptionalData) -> Self:
        """Insert an array of items and their optional data into the circular buffer,
            removing the earliest inserted items if the buffer is full.

        `items`: Array of items of shape (seq_len, *batch_dims, *item_shape_dtype.shape).
        `has_optional_data_mask`: Mask of shape (seq_len, *batch_dims), indicating if each item has optional data.
        `optional_data`: Array of optional data for each item, 
            of shape (seq_len, *batch_dims, *optional_data_shapes_dtypes.shape)
        """
        
        # calculate space left for opt data after opt data of main buffer overriden items is removed
        main_shapes_dtypes = self.main_buffer.item_shapes_dtypes[0]
        if len(get_tree_batch_dims(main_shapes_dtypes, items)) < len(self.main_buffer.batch_dims) + 1: 
            items, has_optional_data_mask, optional_data = jax.tree.map(lambda x: x[None, ...], 
                (items, has_optional_data_mask, optional_data)) # single item -> add batch axis

        removed = self.remove(len(has_optional_data_mask), include_empty=True)
        opt_n_available = self.optional_data_buffer.length - removed.optional_data_buffer.filled_len

        # compact opt data, removing empty items; get index pointers
        compacted_opt_data, opt_is = compacting_mask(optional_data, has_optional_data_mask)

        # discard overflowing opt data, resolve opt_is into absolute indices, insert into main_buffer
        opt_mask_discards_removed = jnp.logical_and(has_optional_data_mask, opt_is < opt_n_available)
        opt_is = self.optional_data_buffer.index(opt_is, include_empty=True)

        main_buffer = self.main_buffer.insert((items, opt_mask_discards_removed, opt_is))

        # mask empty slots in compacted opt data with old opt data to allow insertion into opt data buffer
        if has_optional_data_mask.size > self.optional_data_buffer.length:
            compacted_opt_data_len = self.optional_data_buffer.length
            compacted_opt_data = jax.tree.map(lambda x: x[:compacted_opt_data_len], compacted_opt_data)
        else:
            compacted_opt_data_len = has_optional_data_mask.size

        opt_insert_n = jnp.minimum(jnp.sum(has_optional_data_mask), opt_n_available)
        opt_set_is = jnp.arange(compacted_opt_data_len)
        old_data = self.optional_data_buffer.get(opt_set_is)

        compacted_opt_data = jax.tree.map(lambda new, old, sd: 
                jnp.where((opt_set_is < opt_insert_n)[(...,) + (None,)*len(sd.shape)], new, old),
            compacted_opt_data, old_data, self.optional_data_buffer.item_shapes_dtypes)

        optional_data_buffer = self.optional_data_buffer \
            .set(opt_set_is, compacted_opt_data, include_empty=True) \
            .insert_update_pointers(opt_insert_n)

        return self.replace(main_buffer=main_buffer, optional_data_buffer=optional_data_buffer)

    def remove(self, num: int, include_empty: bool = False) -> Self:
        """Remove the specified number of items and their optional data, 
            starting from the earliest inserted item.
        
        Returns a copy of the buffer with the items marked as removed.

        `include_empty`: If True, considers all slots in the buffer as filled with items.
        """

        _, has_opt, opt_is = self.main_buffer.get(jnp.arange(num), include_empty=include_empty)

        opt_is += jnp.where(opt_is < self.optional_data_buffer.start_i, 
            self.optional_data_buffer.length, 0)
        opt_is = jnp.where(has_opt, opt_is, jnp.iinfo(jnp.int32).min)

        new_opt_start = jnp.maximum(jnp.max(opt_is), self.optional_data_buffer.start_i)

        main_buffer = self.main_buffer.remove(num, include_empty=include_empty)
        opt_buffer = self.optional_data_buffer.remove(new_opt_start - self.optional_data_buffer.start_i)

        return self.replace(main_buffer=main_buffer, optional_data_buffer=opt_buffer)

    def sample(self, key: chex.PRNGKey, seq_len: int | None = None, 
            batch_dims: int | Sequence[int] = ()) -> tuple[TBufferItem, jax.Array, TOptionalData]:
        """Sample a batch of items and their optional data from the circular buffer with uniform probability,

        `seq_length`: If provided, samples sequences (i, i+1, i+2, ...) instead of single samples.
        
        Returned values have additional dimensions (*batch_dims),
            or (*batch_dims, seq_len) if `seq_len` is specified.
        
        Returns: 
            an array of samples
            a mask indicating if each item has optional data
            an array of the samples' corresponding optional data
        """
        main_data, has_opt, opt_i = self.main_buffer.sample(key, seq_len=seq_len, batch_dims=batch_dims)
        return main_data, has_opt, jax.tree.map(lambda x: x[opt_i], self.optional_data_buffer.data)
