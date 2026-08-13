"""Base classes for lyNNX environments."""

from typing import Any, Generic, Sequence
from typing_extensions import TypeVar
from abc import ABC, abstractmethod

import chex

import functools

import jax
from jax.typing import ArrayLike

import jax.numpy as jnp

import numpy as np

from lynnx.utils.math import inv_softplus, normal_entropy
from lynnx.utils.batch_utils import flatten_batched_tree, unflatten_batched_tree, get_tree_flattened_dim

TSpaceElement = TypeVar("TSpaceElement")

class Space(Generic[TSpaceElement]):
    """Defines bounds and structure, such as for observations/actions. 

    Similar to gymnasium.Space, but more flexible as it allows elements to be arbitrary PyTrees,
        rather than requiring custom Tuple/Dict spaces.

    Integer bounds denote discrete values, while float bounds denote continuous values.
    Float bounds can be infinite (np.inf, -np.inf) to indicate an unbounded leaf.

    `low`: Lower bound, inclusive.
    `high`: Upper bound. Inclusive for integer type, exclusive for float type.

    `treedef`: Jax PyTree treedef for an element of the space. Can be used to unflatten elements.
    `shapes_dtypes`: PyTree of `jax.ShapeDtypeStruct`, storing the shape and dtype of each leaf.
    """

    def __init__(self, low: TSpaceElement, high: TSpaceElement) -> None:
        chex.assert_trees_all_equal_structs(low, high,
            custom_message="`low` and `high` must have the same treedef (structure).")

        self.treedef = jax.tree.structure(low)

        # bounds should be static constants -> ensure they are numpy arrays, not jax
        low, high = jax.tree.map(lambda x: np.asarray(x), (low, high))

        self.shapes_dtypes = jax.tree.map(lambda cur_low, cur_high: jax.ShapeDtypeStruct(
            dtype = np.result_type(cur_low, cur_high),
            shape = np.broadcast_shapes(cur_low.shape, cur_high.shape)
        ), low, high)

        self.low = jax.tree.map(
            lambda leaf, shape_dtype: np.broadcast_to(leaf, shape_dtype.shape).astype(shape_dtype.dtype),
            low, self.shapes_dtypes
        )

        self.high = jax.tree.map(
            lambda leaf, shape_dtype: np.broadcast_to(leaf, shape_dtype.shape).astype(shape_dtype.dtype),
            high, self.shapes_dtypes
        )

        # assert low != +inf and high != -inf
        assert jax.tree.all(jax.tree.map(lambda cur_low: np.all(np.logical_not(np.isposinf(cur_low))), self.low)), \
            "`low` values cannot be positive infinity (`+np.inf`)."
        assert jax.tree.all(jax.tree.map(lambda cur_high: np.all(np.logical_not(np.isneginf(cur_high))), self.high)), \
            "`high` values cannot be negative infinity (`-np.inf`)."

        # assert low != high for continuous leafs
        assert jax.tree.all(jax.tree.map(
            lambda cur_low, cur_high: 
                not (np.issubdtype(cur_low.dtype, np.floating) \
                    and np.any(cur_low == cur_high)), 
            self.low, self.high
        )), "`low` cannot equal `high` for continuous leaves, since `high` bound is exclusive while `low` is inclusive."

    def __repr__(self):
        return f"Space(low={repr(self.low)}, high={repr(self.high)})"

    def flatten(self, x: TSpaceElement) -> jax.Array:
        return flatten_batched_tree(self.shapes_dtypes, x)

    def unflatten(self, x: jax.Array) -> TSpaceElement:
        return unflatten_batched_tree(self.shapes_dtypes, x)

    @property
    def flattened_dim(self) -> int:
        return get_tree_flattened_dim(self.shapes_dtypes)

    def contains(self, x: TSpaceElement, batched: bool = False) -> bool:
        """Check if `x` is a valid member of this space. 
        If batched=True, disregards leading batch dimensions"""

        if jax.tree.structure(x) != self.treedef: return False

        def leaf_matches(x_leaf, cur_low, cur_high, shape_dtype):
            shape_compare_start = -len(shape_dtype.shape) if batched else 0
            if x_leaf.shape[shape_compare_start:] != shape_dtype.shape: return False

            if not bool(jnp.all(x_leaf >= cur_low)): return False

            if np.issubdtype(shape_dtype.dtype, np.integer):
                if not jnp.issubdtype(x_leaf.dtype, jnp.integer): return False
                if not bool(jnp.all(x_leaf <= cur_high)): return False
            else:
                if not jnp.issubdtype(x_leaf.dtype, jnp.floating): return False
                if not bool(jnp.all(x_leaf < cur_high)): return False

        return jax.tree.all(leaf_matches, x, self.low, self.high, self.shapes_dtypes)

    def sample(self, key: chex.PRNGKey, batch_dims: Sequence[int] = ()) -> TSpaceElement:
        """Samples a batch of elements of shape `batch_dims` from the space.
        
        Bounded features are sampled according to a uniform distribution.
        
        Continuous (np.floating) features can be unbounded by setting 
        `low` to `-np.inf` and/or `high` to `np.inf`.
            - If one bound is infinite, samples from a standard exponential distribution.
            - If both bounds are infinite, samples from standard normal distribution.
        """

        keys = jax.random.split(key, num=self.treedef.num_leaves)
        keys_tree = jax.tree.unflatten(self.treedef, keys)

        def sample_leaf(cur_low, cur_high, shape_dtype: jax.ShapeDtypeStruct, key: chex.PRNGKey):
            shape = batch_dims + shape_dtype.shape

            if np.issubdtype(shape_dtype.dtype, np.integer):
                return jax.random.randint(key, shape=shape, dtype=shape_dtype.dtype,
                    minval=cur_low, maxval=cur_high + 1)

            else:
                sample = jnp.empty(shape, dtype=shape_dtype.dtype)

                if np.logical_not(np.logical_or(np.isinf(cur_low), np.isinf(cur_high))).any():
                    sample = jax.random.uniform(key, shape=shape, dtype=shape_dtype.dtype,
                        minval=cur_low, maxval=cur_high)

                if np.logical_xor(np.isinf(cur_low), np.isinf(cur_high)).any():
                    exp = jax.random.exponential(key, shape=shape)

                    if np.isinf(cur_low).any():
                        sample = jnp.where(np.isinf(cur_low), cur_high - exp, sample)

                    if np.isinf(cur_high).any():
                        sample = jnp.where(np.isinf(cur_high), cur_low + exp, sample)

                both_unbounded = np.logical_and(np.isinf(cur_low), np.isinf(cur_high))
                if both_unbounded.any():
                    sample = jnp.where(both_unbounded, jax.random.normal(key, shape=shape), sample)

                return sample

        return jax.tree.map(sample_leaf, self.low, self.high, self.shapes_dtypes, keys_tree)

    def squash_continuous_to_bounds(self, x: TSpaceElement) -> TSpaceElement:
        """Squashes unbounded real values (-inf, inf) to the bounds defined by the Space.
        Ignores discrete values."""

        def map_func(leaf, cur_low, cur_high, shape_dtype: jax.ShapeDtypeStruct):
            if np.issubdtype(shape_dtype.dtype, np.integer): return leaf

            # NOTE: extra sanitization due to issue with jnp.where and NaNs for gradients
                # must ensure no NaNs ever get created in either branch; replace infs with dummy values

            squashed = jnp.empty(shape_dtype.shape, dtype=shape_dtype.dtype)

            if np.logical_not(np.logical_or(np.isinf(cur_low), np.isinf(cur_high))).any():
                safe_cur_low = np.where(np.isinf(cur_low), -0.5, cur_low)
                safe_cur_high = np.where(np.isinf(cur_high), 0.5, cur_high)
                squashed = (safe_cur_low+safe_cur_high)/2 + jnp.tanh(leaf)*(safe_cur_high - safe_cur_low)/2
                    # handle both NOT unbounded -> tanh

            if np.logical_xor(np.isinf(cur_low), np.isinf(cur_high)).any():
                softplused = jax.nn.softplus(leaf) # only one side unbounded

                if np.isinf(cur_low).any():
                    squashed = jnp.where(np.isinf(cur_low), cur_high - softplused, squashed)

                if np.isinf(cur_high).any():
                    squashed = jnp.where(np.isinf(cur_high), cur_low + softplused, squashed)

            both_unbounded = np.logical_and(np.isinf(cur_low), np.isinf(cur_high))
            if both_unbounded.any(): # handle both unbounded -> no-op, return original
                squashed = jnp.where(both_unbounded, leaf, squashed)

            return squashed

        return jax.tree.map(map_func, x, self.low, self.high, self.shapes_dtypes)

    def unsquash_continuous_from_bounds(self, x: TSpaceElement) -> TSpaceElement:
        """Undos the `space.squash_continuous_to_bounds(x)` method."""
        
        def map_func(leaf, cur_low, cur_high, shape_dtype: jax.ShapeDtypeStruct):
            if np.issubdtype(shape_dtype.dtype, np.integer): return leaf

            # NOTE: extra sanitization due to issue with jnp.where and NaNs for gradients
                # must ensure no NaNs ever get created in either branch; replace infs with dummy values

            unsquashed = jnp.empty(shape_dtype.shape, dtype=shape_dtype.dtype)

            # both NOT unbounded -> tanh
            if np.logical_not(np.logical_or(np.isinf(cur_low), np.isinf(cur_high))).any():
                safe_cur_low = np.where(np.isinf(cur_low), -0.5, cur_low)
                safe_cur_high = np.where(np.isinf(cur_high), 0.5, cur_high)
                mid = (safe_cur_low+safe_cur_high)/2
                unsquashed = jnp.arctanh(jnp.clip(
                    (leaf - np.where(np.isinf(mid), 0, mid)) / (safe_cur_high-safe_cur_low) * 2,
                    min=-1, max=1
                ))

            # only one side unbounded -> softplus
            if np.isinf(cur_low).any():
                unsquashed = jnp.where(np.isinf(cur_low), 
                    inv_softplus(jnp.maximum(0, cur_high - leaf)), unsquashed)
            if np.isinf(cur_high).any():        
                unsquashed = jnp.where(np.isinf(cur_high), 
                    inv_softplus(jnp.maximum(0, leaf - cur_low)), unsquashed)

            # both unbounded -> no-op, return original
            both_unbounded = np.logical_and(np.isinf(cur_low), np.isinf(cur_high))
            if both_unbounded.any():
                unsquashed = jnp.where(both_unbounded, leaf, unsquashed)

            return unsquashed

        return jax.tree.map(map_func, x, self.low, self.high, self.shapes_dtypes)

    def add_noise_to_continuous(self, key: chex.PRNGKey, x: TSpaceElement, 
            noise_std: ArrayLike, noise_clip: ArrayLike | None = None, bounds_clip: bool = True) -> TSpaceElement:
        """Adds gaussian noise to continuous leaves.

        `bounds_clip`: If True (default), clips final values to bounds.
        """

        keys = jax.random.split(key, num=self.treedef.num_leaves)
        keys_tree = jax.tree.unflatten(self.treedef, keys)

        def map_func(cur, cur_low, cur_high, shape_dtype: jax.ShapeDtypeStruct, key: chex.PRNGKey):
            if np.issubdtype(shape_dtype.dtype, np.integer): return cur

            noise = noise_std * jax.random.normal(key, shape_dtype.shape)
            
            if noise_clip is not None:
                noise = jnp.clip(noise, -noise_clip, noise_clip)

            cur += noise

            if bounds_clip:
                cur = jnp.clip(cur, cur_low, cur_high)

            return cur

        return jax.tree.map(map_func, x, self.low, self.high, self.shapes_dtypes, keys_tree)

    def sample_distribution(self, key: chex.PRNGKey, distribution: TSpaceElement, batch_dims: int | Sequence[int] = (),
            squash_continuous=True, deterministic=False, log_stds=False) -> TSpaceElement:
        """Samples a batch of elements of shape `batch_dims` from the space, according to the given distribution.
        
        The distribution should have the same treedef as the Space. Each leaf should have the same shape,
        but with an extra trailing axis as follows:

            - If discrete (jnp.integer): values will be sampled according to a categorical distribution.
                Trailing axis should have size equal to the maximum number of possible values 
                    for any feature in the Space in that leaf: `max(high - low + 1)`
                Values along trailing axis should be logits. The 0th logit should correspond to the
                    mininum possible value of that feature. Pad with `-jnp.inf` at the end as necessary.

            If continuous (jnp.floating): values will be sampled according to a normal distribution.
                Trailing axis should have size 2. The first element should be the mean, 
                    and the second should be the standard deviation.
                If bounded, values will be squashed to bounds using the softplus (one side bounded) 
                    or tanh (both sides bounded) function.

        `squash_continuous`: If False, does not squash continuous values, leaving them unbounded.
            Useful, eg. for sampling raw outputs, before softplus or tanh.

        `deterministic`: If True, returns a deterministic representative value instead of a random sample.
            Discrete (jnp.integer) leaves: selects the choice with the highest probability (mode).
            Continuous (jnp.floating) leaves: takes the median of the distribution.

        `log_stds`: If True, treats stds as log stds: uses exp(feature[1]) as the standard deviation.
        """
        if isinstance(batch_dims, int): batch_dims = (batch_dims,)

        keys = jax.random.split(key, num=self.treedef.num_leaves)
        keys_tree = jax.tree.unflatten(self.treedef, keys)

        def sample_leaf(cur_dist, cur_low, shape_dtype: jax.ShapeDtypeStruct, key: chex.PRNGKey):
            shape = batch_dims + cur_dist.shape[:-1]

            if np.issubdtype(shape_dtype.dtype, np.integer):
                if deterministic: 
                    indices = jnp.argmax(cur_dist, axis=-1)
                    indices = jnp.tile(jnp.expand_dims(indices, batch_dims), batch_dims + (1,)*len(shape_dtype.shape))
                else: 
                    indices = jax.random.categorical(key, cur_dist, axis=-1, shape=shape)

                return cur_low + indices
            else:
                if deterministic: z = jnp.zeros(shape)
                else: z = jax.random.normal(key, shape=shape, dtype=shape_dtype.dtype)

                std = jnp.exp(cur_dist[..., 1]) if log_stds else cur_dist[..., 1]

                return cur_dist[..., 0] + std*z

        sample = jax.tree.map(sample_leaf, distribution, self.low, self.shapes_dtypes, keys_tree)

        if squash_continuous:
            sample = self.squash_continuous_to_bounds(sample)

        return sample

    def log_probabilities(self, x: TSpaceElement, distribution: TSpaceElement, 
            continuous_squashed=True, log_stds=False) -> TSpaceElement:
        """Computes the individual log probability of sampling each feature of `x` from `distribution`.
        See `space.sample_distribution` for details on the structure of `distribution`.

        Tanh-squashed action probabilities are adjusted by -log(1 - tanh^2(x)), as is commonly done in SAC.
            We compute a more numerically stable version: -2(log2 + x - softplus(2x)).
        Similarly, softplus-squashed actions are adjusted by logaddexp(0, -x), ie. log(1 + e^(-x)).

        `continuous_squashed`: If False, assumes all continuous values are unsquashed and unbounded.
            Useful, eg. for processing raw outputs, before softplus or tanh.

        `log_stds`: If True, treats stds as log stds: uses exp(feature[1]) as the standard deviation.
        """

        def leaf_log_probabilities(unsquashed_leaf, cur_dist, 
                cur_low, cur_high, shape_dtype: jax.ShapeDtypeStruct) -> TSpaceElement:

            if np.issubdtype(shape_dtype.dtype, np.integer):
                all_log_probs = jax.nn.log_softmax(cur_dist, axis=-1)
                dist_is = unsquashed_leaf - cur_low # unsquashed and squashed are the same
                all_log_probs, dist_is = jnp.broadcast_arrays(all_log_probs, dist_is[..., None])
                return jnp.take_along_axis(all_log_probs, dist_is[..., (0, )], axis=-1).squeeze(axis=-1)

            else: 
                std = jnp.exp(cur_dist[..., 1]) if log_stds else cur_dist[..., 1]
                norm_prob = jax.scipy.stats.norm.logpdf(unsquashed_leaf, loc=cur_dist[..., 0], scale=std)

                # NOTE: extra sanitization due to issue with jnp.where and NaNs for gradients
                    # must ensure no NaNs ever get created in either branch; replace infs with dummy values

                ## adjust for squashing transformation for bounded features using jacobian
                adjust = jnp.empty(shape_dtype.shape, dtype=shape_dtype.dtype)

                if np.logical_not(np.logical_or(np.isinf(cur_low), np.isinf(cur_high))).any():
                    adjust = -2 * (jnp.log(2) + unsquashed_leaf - jax.nn.softplus(2*unsquashed_leaf))
                        # neither side unbounded -> tanh; numerically stable form of -log(1 - tanh^2(x))
                
                one_unbounded = np.logical_xor(np.isinf(cur_low), np.isinf(cur_high)) 
                if one_unbounded.any(): # handle one side unbounded -> softplus
                    adjust = jnp.where(one_unbounded, jnp.logaddexp(0, -unsquashed_leaf), adjust) 
                
                both_unbounded = np.logical_and(np.isinf(cur_low), np.isinf(cur_high))
                if both_unbounded.any(): # handle both unbounded -> no transformation
                    adjust = jnp.where(both_unbounded, 0, adjust)

                return norm_prob + adjust

        if continuous_squashed:
            x = self.unsquash_continuous_from_bounds(x)

        return jax.tree.map(leaf_log_probabilities, x, distribution, self.low, self.high, self.shapes_dtypes)

    def log_probability(self, x: TSpaceElement, distribution: TSpaceElement, 
            continuous_squashed=True, log_stds=False) -> ArrayLike:
        """Computes the total log probability of sampling `x` from `distribution`, assuming features are independent.
        See `space.sample_distribution` for details on the structure of `distribution`.

        Tanh-squashed action probabilities are adjusted by -log(1 - tanh^2(x)), as is commonly done in SAC.
            We compute a more numerically stable version: -2(log2 + x - softplus(2x)).
        Similarly, softplus-squashed actions are adjusted by logaddexp(0, -x), ie. log(1 + e^(-x)).

        `continuous_squashed`: If False, assumes all continuous values are unsquashed and unbounded.
            Useful, eg. for processing raw outputs, before softplus or tanh.

        `log_stds`: If True, treats stds as log stds: uses exp(feature[1]) as the standard deviation.
        """

        log_probabilities = self.log_probabilities(x, distribution, continuous_squashed, log_stds)

        log_probabilities = jax.tree.map(
            lambda leaf, s_dt: jnp.sum(leaf, axis=tuple(range(-len(s_dt.shape), 0))),
            log_probabilities, self.shapes_dtypes
        )

        return jax.tree.reduce(lambda tot, cur: tot + cur, log_probabilities)

    def entropies(self, distribution: TSpaceElement, log_stds=False,
            monte_carlo_n_samples: int | None = None, monte_carlo_key: chex.PRNGKey | None = None) -> TSpaceElement:
        """Computes the individual entropy of each feature of `distribution`.
            See `space.sample_distribution` for details on the structure of `distribution`.

        For bounded (one or both sides) continuous features, there is no analytical expression for entropy.
            This function will compute a monte-carlo estimate in these cases, using `monte_carlo_n_samples` samples. 
            The parameters `monte_carlo_n_samples` and `monte_carlo_key` must be provided if there are any such features.

        Tanh-squashed action probabilities are adjusted by -log(1 - tanh^2(x)), as is commonly done in SAC.
            We compute a more numerically stable version: -2(log2 + x - softplus(2x)).
        Similarly, softplus-squashed actions are adjusted by logaddexp(0, -x), ie. log(1 + e^(-x)).

        `log_stds`: If True, treats stds as log stds: uses exp(feature[1]) as the standard deviation.
        """

        def leaf_entropies(monte_carlo_est, cur_dist, cur_low, cur_high, shape_dtype: jax.ShapeDtypeStruct):
            if np.issubdtype(shape_dtype.dtype, np.integer):
                log_probs = jax.nn.log_softmax(cur_dist, axis=-1)
                log_probs = jnp.where(jnp.isneginf(log_probs), 0, log_probs) # handle 0 probability
                return - jnp.exp(log_probs) * log_probs
            else: 
                both_unbounded = np.logical_and(np.isinf(cur_low), np.isinf(cur_high))
                if both_unbounded.any():
                    std = jnp.exp(cur_dist[..., 1]) if log_stds else cur_dist[..., 1]
                    return jnp.where(both_unbounded, normal_entropy(std), monte_carlo_est)

                assert monte_carlo_n_samples is not None, \
                    "`monte_carlo_n_samples` must be provided as there are bounded continuous features."
                return monte_carlo_est

        if monte_carlo_n_samples is None:
            monte_carlo_ests = jax.tree.map(lambda sd: jnp.empty(sd.shape), self.shapes_dtypes)
        else:
            assert monte_carlo_key is not None, \
                "`monte_carlo_key` must be provided for monte carlo estimation."

            # NOTE: unnecessary sample and log_p computations will be optimized out by the compiler
            samples = self.sample_distribution(monte_carlo_key, distribution, batch_dims=monte_carlo_n_samples,
                squash_continuous=False, log_stds=log_stds)

            log_ps = self.log_probabilities(samples, distribution, 
                continuous_squashed=False, log_stds=log_stds)

            monte_carlo_ests = jax.tree.map(lambda log_p: jnp.mean(-log_p, axis=0), log_ps)

        return jax.tree.map(leaf_entropies, monte_carlo_ests, distribution, self.low, self.high, self.shapes_dtypes)

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")
TEnvAction = TypeVar("TEnvAction")
TRenderFrame = TypeVar("TRenderFrame", default=None)

class Environment(ABC, Generic[TEnvState, TEnvObs, TEnvAction, TRenderFrame]):
    """Abstract base class for environments."""

    @abstractmethod
    def reset(self, key: chex.PRNGKey) -> tuple[TEnvState, dict[Any, Any]]:
        """Performs resetting of the environment.
        Returns: state, info"""

    @abstractmethod
    def step(self, key: chex.PRNGKey, state: TEnvState, action: TEnvAction) \
        -> tuple[TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform a step transition in the environment.
        Returns: state, reward, terminated, truncated, info"""

    @abstractmethod
    def get_obs(self, key: chex.PRNGKey, state: TEnvState) -> TEnvObs:
        """Gets an observation from a state."""

    def render(self, state: TEnvState, action: ArrayLike) -> TRenderFrame:
        """Compute a render frame from a state-action pair.
        Intended for human interpretation (visualization, debugging); may not be usable as a policy input.
        Implementations may or may not be jittable.
        Unimplemented by default, returning None."""
        return None

    @property
    @abstractmethod
    def observation_space(self) -> Space[TEnvObs]:
        """Observation space of the environment."""

    @property
    @abstractmethod
    def action_space(self) -> Space[TEnvAction]:
        """Action space of the environment."""

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__