from typing import TypeVar, Generic, Self

import jax
import jax.numpy as jnp
import chex

from lynnx.utils.batch_utils import flatten_tree_batch_axes

TInput = TypeVar('TInput')

@chex.dataclass(frozen=True)
class RunningMeanVar(Generic[TInput]):
    """Class which keeps a running mean and variance, using Welford's formula."""

    mean: TInput
    var: TInput
    count: jax.Array

    @classmethod
    def init(cls, shapes_dtypes: TInput) -> Self:
        """Initialize the class, with no tracked samples yet."""
        return cls(
            mean = jax.tree.map(lambda s_dt: jnp.zeros(s_dt.shape), shapes_dtypes),
            var = jax.tree.map(lambda s_dt: jnp.zeros(s_dt.shape), shapes_dtypes),
            count = jnp.array(0, dtype=jnp.int32)
        )

    def merge(self, other: Self) -> Self:
        """Merges running mean and variance statistics with that of another RunningMeanVar object,
            returning an updated RunningMeanVar object.

        See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """

        new_count = self.count + other.count
        count_ratio = other.count / new_count

        new_mean = jax.tree.map(lambda self_mean, other_mean: self_mean + (other_mean-self_mean)*count_ratio,
            self.mean, other.mean)

        def get_new_var(self_mean: jax.Array, self_var: jax.Array, other_mean: jax.Array, other_var: jax.Array):
            self_M2 = self_var * self.count
            other_M2 = other_var * other.count
            m_2 = self_M2 + other_M2 + jnp.square(other_mean-self_mean)*self.count*count_ratio
            return m_2 / new_count

        new_var = jax.tree.map(get_new_var, self.mean, self.var, other.mean, other.var)

        return RunningMeanVar(mean=new_mean, var=new_var, count=new_count)

    def update(self, x: TInput) -> Self:
        """Updates the running mean and variance with one or more samples, 
            returning an updated RunningMeanVar object."""
        x = flatten_tree_batch_axes(jax.eval_shape(lambda: self.mean), x)

        batch_stats = RunningMeanVar(
            mean = jax.tree.map(lambda x: jnp.mean(x, axis=0), x),
            var = jax.tree.map(lambda x: jnp.var(x, axis=0), x),
            count = len(x)
        )

        return self.merge(batch_stats)

    def normalize(self, x: TInput, epsilon=1e-8) -> TInput:
        """Standardize an input to mean=0, var=1 using the running mean/variance."""
        return jax.tree.map(lambda x, mean, var: (x - mean) / jnp.sqrt(var + epsilon), x, self.mean, self.var)

# if __name__ == "__main__":
#     running_mean_var0 = RunningMeanVar.init(jax.eval_shape(lambda: 1))

#     arr1 = jnp.array([1, 3, -10, 4, 2, 5])
#     arr2 = jnp.array([10, 2, 90, -50, 60, 30])

#     running_mean_var1 = running_mean_var0.update(arr1)
#     running_mean_var2 = running_mean_var1.update(arr2)

#     print(running_mean_var1, jnp.mean(arr1), jnp.var(arr1))
#     comb_arr = jnp.concatenate((arr1, arr2))
#     print(running_mean_var2, jnp.mean(comb_arr), jnp.var(comb_arr))