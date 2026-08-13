"""Useful Flax NNX Modules."""

from typing import Any, Generic, TypeVar, overload, Container, ParamSpec, Callable, Sequence

from functools import update_wrapper

import math

from jax.typing import ArrayLike
import jax
import jax.numpy as jnp

import numpy as np

from flax import nnx

from lynnx.envs.base import Space
from lynnx.utils.running_mean_var import RunningMeanVar
from lynnx.utils.func_utils import optionally_pass

from lynnx.algos.base import AlgoPhase

TNetworkInput = TypeVar("TNetworkInput")

class Identity(nnx.Module):
    """Module which does nothing, directly returning the input, to be used as a placeholder."""
    def __call__(x: TNetworkInput, rngs=None) -> TNetworkInput:
        return x

TStatefulFuncParams = ParamSpec('TStatefulFuncParams')
TStatefulFuncState = ParamSpec('TStatefulFuncState')
TOutput = TypeVar('TOutput')

class StatefulFunc(nnx.Module, Generic[TStatefulFuncState, TStatefulFuncParams, TOutput]):
    """Wrapper that allows a function to be "impure", reading/mutating external state, by converting 
        it into a Flax NNX Module and supplying `*state_args, **state_kwargs` during each call.

    Useful for allowing Flax NNX's object-oriented state tracking to work 
        inside callbacks passed to higher-order functions.

    Examples:
        >>> import jax.numpy as jnp
        >>> from flax import nnx
        >>> from lynnx.utils.nnx_modules import StatefulFunc

        >>> class Counter(nnx.Module):
        ...     def __init__(self):
        ...         self.count = nnx.Variable(jnp.array(0))
        ...     def __call__(self, x):
        ...         self.count.value = self.count.value + x
        ...         return self.count.value

        >>> counter = Counter()
        >>> count_double = StatefulFunc(lambda x, counter: counter(2*x), counter=counter)
        >>> count_double(1)
        Array(2, dtype=int32)
        >>> count_double(1)
        Array(4, dtype=int32)
    """

    def __init__(self, func: Callable[..., TOutput], 
            *state_args: TStatefulFuncState.args, **state_kwargs: TStatefulFuncState.kwargs) -> None:
        self.func = func
        self.state_args = state_args
        self.state_kwargs = state_kwargs

        update_wrapper(self, func) # update function metadata

    def __call__(self, *args: TStatefulFuncParams.args, **kwargs: TStatefulFuncParams.kwargs) -> TOutput:
        return self.func(*self.state_args, *args, **self.state_kwargs, **kwargs)

def stateful_func(*state_args: TStatefulFuncState.args, **state_kwargs: TStatefulFuncState.kwargs) \
        -> Callable[[Callable[..., TOutput]], StatefulFunc[TStatefulFuncState, TStatefulFuncParams, TOutput]]:
    """Decorator factory for :class:`StatefulFunc`."""
    return lambda inner: StatefulFunc(inner, *state_args, **state_kwargs)

class Pipe(nnx.Module):
    """Stores a sequence of callables, and applies them in order when called. 
    Similar to `flax.nnx.Sequential`, but does not apply any extra processing to function outputs.
    """

    def __init__(self, *fns: Callable[..., Any]) -> None:
        self.layers = list(fns)

    def __call__(self, *args, rngs: nnx.Rngs | None = None, **kwargs) -> Any:
        output: Any = None

        for i, f in enumerate(self.layers):
            if not callable(f): raise TypeError(f'Sequence[{i}] is not callable: {f}')

            f = optionally_pass(f, rngs=rngs)
            
            if i == 0: output = f(*args, **kwargs)
            else: output = f(output)

        return output

class MLP(nnx.Module):
    """MLP which applies linear layers, activation functions, and optionally, layer norms."""

    def __init__(self, rngs: nnx.Rngs, layer_dims: Sequence[int],
            do_layer_norm: bool = False, activation_func = nnx.relu) -> None:
        self.layer_dims = layer_dims

        self.do_layer_norm = do_layer_norm
        self.activation_func = activation_func

        self.linear_layers = [ nnx.Linear(in_dim, out_dim, rngs=rngs)
            for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]) ]

        if do_layer_norm:
            self.layer_norms = [ nnx.LayerNorm(dim, rngs=rngs) for dim in layer_dims[1:-1] ]

    def __call__(self, x: jax.Array, rngs: nnx.Rngs = None) -> jax.Array:

        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i](x)

            if i < len(self.linear_layers) - 1: # don't layer_norm/activation_func output layer
                if self.do_layer_norm:
                    x = self.layer_norms[i](x)

                x = self.activation_func(x)

        return x

class RunningMeanVarNorm(nnx.Module, Generic[TNetworkInput]):
    """Normalizes inputs using a running mean and running variance.
    Running mean/var will be updated during the rollout phase only."""

    @overload
    def __init__(self, __running_mean_var: RunningMeanVar[TNetworkInput], 
        clip_threshold: float | None = None, do_update_phases: Container[AlgoPhase] = (AlgoPhase.ROLLOUT,)) -> None: ...

    @overload
    def __init__(self, __shapes_dtypes: TNetworkInput, 
        clip_threshold: float | None = None, do_update_phases: Container[AlgoPhase] = (AlgoPhase.ROLLOUT,)) -> None: ...

    def __init__(self, inp: TNetworkInput | RunningMeanVar[TNetworkInput], 
            clip_threshold: float | None = None, do_update_phases: Container[AlgoPhase] = (AlgoPhase.ROLLOUT,)) -> None:
        self.algo_phase: AlgoPhase = AlgoPhase.OPTIMIZE # hook which will be set by algos

        if not isinstance(inp, RunningMeanVar): inp = RunningMeanVar.init(inp)
        self.running_mean_var = nnx.Variable(inp)

        self.clip_threshold = None if clip_threshold is None else float(clip_threshold)
        self.do_update_phases = do_update_phases

    def __call__(self, x: TNetworkInput, rngs: nnx.Rngs = None) -> TNetworkInput:
        if self.algo_phase in self.do_update_phases: 
            self.running_mean_var.value = self.running_mean_var.value.update(x)

        x = self.running_mean_var.value.normalize(x)

        if self.clip_threshold is not None:
            x = jax.tree.map(lambda x: jnp.clip(x, -self.clip_threshold, self.clip_threshold), x)

        return x

TEnvAction = TypeVar("TEnvAction")

class ActionDistributionHead(nnx.Module, Generic[TEnvAction]):
    """Converts a layer of floats into an action distribution.
    See :func:`lynnx.envs.base.Space.sample_distribution` for details on the structure of the output distribution.
    
    If `do_state_independent_stds` is True, stores tunable parameters representing log stds.
    """

    def __init__(self, action_space: Space[TEnvAction], do_state_independent_stds = True) -> None:
        self.action_space = action_space
        self.do_state_independent_stds = do_state_independent_stds

        self.input_dim = 0
        self.num_continuous = 0
        self._leaves_descrip = []

        for cur_low, cur_high in zip(jax.tree.leaves(action_space.low), jax.tree.leaves(action_space.high)):
            if np.issubdtype(cur_low.dtype, np.integer):
                n_choices = cur_high - cur_low + 1
                n_values_needed = int(np.sum(n_choices))
                self.input_dim += n_values_needed

                self._leaves_descrip.append({ 
                    'discrete': True, 
                    'n_values_needed': n_values_needed,
                    'shape': (*cur_low.shape, int(np.max(n_choices))),
                    'n_choices': n_choices.tolist() # convert to list to mark as static
                })
            else:
                dim = math.prod(cur_low.shape)
                self.num_continuous += dim
                n_values_needed = dim * (2 - do_state_independent_stds)
                self.input_dim += n_values_needed

                self._leaves_descrip.append({ 
                    'discrete': False, 
                    'n_values_needed': n_values_needed,
                    'shape': (*cur_low.shape, 2), # mean, std
                })
        
        if do_state_independent_stds and self.num_continuous > 0:
            self.state_independent_log_stds = nnx.Param(jnp.full(self.num_continuous, jnp.log(0.5)))
                # stds should be initialized small; https://arxiv.org/abs/2006.05990

    def __call__(self, x: jax.Array, rngs=None):
        """Converts a 1D array of flattened values into an unflattened action distribution."""
        leaves = []
        i = 0
        state_indep_log_stds_i = 0

        for leaf_descrip in self._leaves_descrip:
            vals = x[..., i : i+leaf_descrip['n_values_needed']]
            i += leaf_descrip['n_values_needed']

            if leaf_descrip['discrete']:
                leaf = jnp.full(vals.shape[:-1] + leaf_descrip['shape'], -jnp.inf)
                vals_i = 0
                
                for path, cur_n_choices in np.ndenumerate(np.asarray(leaf_descrip['n_choices'])):
                    leaf = leaf.at[(..., *path, slice(cur_n_choices))].set(vals[..., vals_i : vals_i + cur_n_choices])
                    vals_i += cur_n_choices

                leaves.append(leaf)

            else:
                if self.do_state_independent_stds:
                    state_indep_log_stds = self.state_independent_log_stds.value \
                        [state_indep_log_stds_i : state_indep_log_stds_i + leaf_descrip['n_values_needed']]
                    state_indep_log_stds_i += leaf_descrip['n_values_needed']

                    means = vals.reshape(vals.shape[:-1] + leaf_descrip['shape'][:-1])

                    leaves.append(jnp.stack((
                        means, 
                        jnp.broadcast_to(state_indep_log_stds, means.shape)
                    ), axis=-1))
                else:
                    leaves.append(vals.reshape(vals.shape[:-1] + leaf_descrip['shape']))

        return jax.tree.unflatten(self.action_space.treedef, leaves)