"""Implementation of Q-Learning - https://doi.org/10.1007/BF00992698

We take `n_envs` steps in parallel, and update the Q table
    using the sampled transitions completely in parallel.

Along with the basic tabular implementation for discrete observation spaces,
    we offer two options for continuous observations: 
        rounding (default), and linear interpolation using nearby table corners.
    Note that these methods break convergence guarentees.
"""

from typing import TypeVar, Generic, Any, Sequence

import math

import numpy as np

import jax.numpy as jnp
import jax
from jax import flatten_util
from jax.typing import ArrayLike

from chex import dataclass
import chex

from flax import nnx

from lynnx.utils import LinearlyInterpolatedTable
from lynnx.utils.func_utils import try_call
from lynnx.utils.batch_utils import flatten_batched_tree

from lynnx.envs.base import Environment, Space
from lynnx.algos.base import Scheduleable, GreedyQActor, set_algo_phase, AlgoPhase

from lynnx.envs.wrappers import AutoResetWrapper

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")

class TabularQFunc(Generic[TEnvObs], nnx.Module):
    """Get Q values by looking up in a discrete Q table, rounding indices if they fall between gridpoints.
    Implements the `lynnx.algos.base.DiscreteQFunc` protocol."""

    def __init__(self, 
        num_actions: int,
        observation_space: Space[TEnvObs],
        obs_resolution: TEnvObs | None = None,
        q_table_values: jax.Array = None
    ) -> None:
        """`obs_resolution`: PyTree with same treedef/shape as env.observation_space, defaults to 1."""
        self.num_actions = num_actions
        
        self.obs_shapes_dtypes = observation_space.shapes_dtypes

        if obs_resolution is None:
            obs_resolution = jax.tree.map(np.ones_like, observation_space.low)

        self.obs_resolution = jax.tree.map(lambda x: np.asarray(x), obs_resolution)
        chex.assert_trees_all_equal_shapes(self.obs_resolution, observation_space.low)

        self.obs_low_flattened = np.asarray(flatten_util.ravel_pytree(observation_space.low)[0])
        self.obs_high_flattened = np.asarray(flatten_util.ravel_pytree(observation_space.high)[0])
        self.obs_resolution_flattened = np.asarray(flatten_util.ravel_pytree(obs_resolution)[0])

        dim_lens = (self.obs_high_flattened - self.obs_low_flattened) / self.obs_resolution_flattened
        q_table_shape = (num_actions, *(1 + np.round(dim_lens)).astype(int))

        if q_table_values is None:
            q_table_values = jnp.zeros(q_table_shape)

        assert q_table_values.shape == q_table_shape, \
            f"Expected shape {q_table_shape} for `q_table_values`, got {q_table_values.shape}."

        self.q_table_values = nnx.Param(q_table_values)

    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Returns a Q value for every possible action, given `obs`."""
        return jax.vmap(self.get_table_value, in_axes=(None, 0), out_axes=-1)(obs, jnp.arange(self.num_actions))

    def get_table_value(self, obs: TEnvObs, action: ArrayLike) -> ArrayLike:
        """Returns the Q value for the given `obs` and `action`."""
        indices = (action, *self.obs_table_indices(obs))
        return self.q_table_values.value[indices]

    def adjust_table_value(self, obs: TEnvObs, action: ArrayLike, adjust: ArrayLike) -> None:
        """Adjust the Q table's value for the given `obs` and `action` by `adjust`."""
        indices = (action, *self.obs_table_indices(obs))
        self.q_table_values.value = self.q_table_values.value.at[indices].add(adjust)

    def obs_table_indices(self, obs: TEnvObs) -> Sequence[ArrayLike]:
        """Get corresponding indices in the Q table for a particular observation."""
        flattened_obs = flatten_batched_tree(self.obs_shapes_dtypes, obs)

        array_is = jnp.clip(
            jnp.round((flattened_obs - self.obs_low_flattened) / self.obs_resolution_flattened), 
            0, jnp.array(self.q_table_values.value.shape[1:]) - 1
        ).astype(int)

        return tuple(jnp.moveaxis(array_is, -1, 0))

class LinInterpTabularQFunc(Generic[TEnvObs], TabularQFunc[TEnvObs]):
    """Get Q values by linearly interpolating nearby table corners.
    Implements the `lynnx.algos.base.DiscreteQFunc` protocol."""

    def __init__(self, 
        num_actions: int,
        observation_space: Space[TEnvObs],
        obs_resolution: TEnvObs,
        q_table_values: jax.Array = None
    ) -> None:
        """`obs_resolution`: PyTree with same treedef/shape as env.observation_space, defaults to 1."""

        # `LinearlyInterpolatedTable` ensures the space is fully covered by the table corners, 
            # while `TabularQFunc` rounds, potentially rounding down; correct for this
        observation_space = Space(low=observation_space.low,
            high=jax.tree.map(lambda lo, hi, res: np.ceil((hi-lo) / res)*res + lo, 
                observation_space.low, observation_space.high, obs_resolution))

        super().__init__(num_actions, observation_space, obs_resolution, q_table_values)

        self.lin_interp_table = LinearlyInterpolatedTable(
            min=self.obs_low_flattened,
            max=self.obs_high_flattened,
            step=self.obs_resolution_flattened
        )

    def get_table_value(self, obs: TEnvObs, action: ArrayLike) -> ArrayLike:
        """Returns the Q value for the given `obs` and `action`."""
        return self.lin_interp_table.get(self.q_table_values.value[action], self.obs_table_indices(obs))

    def adjust_table_value(self, obs: TEnvObs, action: ArrayLike, adjust: ArrayLike) -> None:
        """Adjust the Q table's value for the given `obs` and `action` by `adjust`."""

        corner_indices, adjust_amounts = self.lin_interp_table.get_corner_adjustments(
            self.q_table_values.value[action], self.obs_table_indices(obs), adjust)

        adjust_indices_tuple = (action[:, None], ) + tuple(jnp.moveaxis(corner_indices, -1, 0))
        self.q_table_values.value = self.q_table_values.value.at[adjust_indices_tuple].add(adjust_amounts)

    def obs_table_indices(self, obs: TEnvObs) -> Sequence[ArrayLike]:
        """Get corresponding indices in the Q table for a particular observation."""
        return flatten_batched_tree(self.obs_shapes_dtypes, obs)

@dataclass(frozen=True)
class Hyperparameters:
    """Hyperparameters for Tabular Q-Learning.

    `n_envs`: Number of environments to run in parallel.
    `discount_rate`: Discount factor gamma for the environment.

    `epsilon`: Chance to take a random action instead of a greedy action during rollouts.
        It is recommended to use a schedule: eg. decay from 1 to ~0.05 over ~10% of training steps.

    `learning_rate`: Learning rate, scaling update sizes.
    """

    n_envs: int = 32
    discount_rate: Scheduleable[float] = 0.95

    epsilon: Scheduleable[float] = 0.05

    learning_rate: Scheduleable[float] = 0.1

@dataclass
class TrainingState(Generic[TEnvState, TEnvObs]):
    """Training state for Tabular Q-Learning."""
    steps: ArrayLike
    env_states: TEnvState

    q_func: TabularQFunc

class TabularQLearning(Generic[TEnvState, TEnvObs]):
    """Main class for Tabular Q-Learning, facilitating initialization and training.
    See the module docstring for more details.
    
    Continuous observations are handled using rounding by default. To enable linear interpolation,
        create an instance of `LinInterpTabularQFunc` and pass it to `init_training_state()`.
    """

    def __init__(self, 
        env: Environment[TEnvState, TEnvObs, ArrayLike],
        hyperparameters: Hyperparameters = Hyperparameters()
    ) -> None:
        """
        IMPORTANT: 
            `env` must already be batched; eg. wrap with `VmapWrapper` BEFORE passing in.
            `env` should not auto-reset.
        """

        assert (
            jnp.isscalar(env.action_space.low) 
            and jnp.issubdtype(env.action_space.shapes_dtypes.dtype, jnp.integer)
            and env.action_space.low == 0
        ), "Action space for Q-Learning must be discrete (jnp integer scalar, min=0)."

        self.env = env
        self.hyperparameters = hyperparameters

    def make_default_q_func(self, init_val: ArrayLike | None = None, rngs=None) -> TabularQFunc[TEnvObs]:
        """Makes the default Q function - a Q table with rounding for continuous observations."""
        q_func = TabularQFunc(int(self.env.action_space.high + 1), self.env.observation_space)

        if init_val is not None:
            q_func.q_table_values.value = jnp.full_like(q_func.q_table_values.value, init_val)

        return q_func

    def make_actor(self, q_func: TabularQFunc[TEnvObs] | None = None, 
            epsilon: ArrayLike = 0, **kwargs) -> GreedyQActor[TEnvObs]:
        """Make an Actor (obs, rngs) -> (action, infos) using `q_func`."""

        if q_func is None: q_func = self.make_default_q_func(**kwargs)
        return GreedyQActor(q_func, int(self.env.action_space.high + 1), epsilon=epsilon)

    def init_training_state(self, rngs: nnx.Rngs, q_func: TabularQFunc[TEnvObs] | None = None) \
            -> TrainingState[TEnvState, TEnvObs]:
        """Initialize a starting training state, ready for training.
        
        Creates a default Q function object.
        """

        if q_func is None: q_func = self.make_default_q_func()
        env_states, _ = jax.jit(self.env.reset)(jax.random.split(rngs.env(), self.hyperparameters.n_envs))

        return TrainingState(
            steps = jnp.array(0, dtype=jnp.int32),
            env_states = env_states,    
            q_func = q_func,
        )

    def train(self, rngs: nnx.Rngs, training_state: TrainingState[TEnvState, TEnvObs], steps: int) \
            -> tuple[TrainingState[TEnvState, TEnvObs], dict[Any, Any]]:
        """Train from the given `training_state`, returning an updated `training_state` and metrics."""

        def train_iteration(training_state: TrainingState[TEnvState, TEnvObs], rngs: nnx.Rngs) \
                -> tuple[TrainingState[TEnvState, TEnvObs], dict[Any, Any]]:
            ## sample transitions from environment ##
            set_algo_phase(training_state.q_func, AlgoPhase.ROLLOUT)

            actor = self.make_actor(training_state.q_func, 
                try_call(self.hyperparameters.epsilon, training_state.steps))

            obs = self.env.get_obs(jax.random.split(rngs.env(), self.hyperparameters.n_envs), training_state.env_states)
            actions, _ = actor(obs, rngs=rngs)

            training_state.env_states, rewards, terminateds, truncated, infos = AutoResetWrapper(self.env).step(
                jax.random.split(rngs.env(), self.hyperparameters.n_envs), training_state.env_states, actions)

            next_obs = self.env.get_obs(jax.random.split(rngs.env(), self.hyperparameters.n_envs), 
                infos[AutoResetWrapper.UNRESET_STATE_INFO_KEY])

            training_state.steps += self.hyperparameters.n_envs

            ## update q functions ##
            set_algo_phase(training_state.q_func, AlgoPhase.OPTIMIZE)

            next_qs = training_state.q_func(next_obs)
            max_next_qs = jnp.max(next_qs, axis=-1)
            # zero out q_val if terminated
            max_next_qs = max_next_qs * jnp.logical_not(terminateds)

            target_qs = rewards \
                + try_call(self.hyperparameters.discount_rate, steps)*max_next_qs
            pred_qs = training_state.q_func.get_table_value(obs, actions)

            adjusts = try_call(self.hyperparameters.learning_rate, steps) * (target_qs - pred_qs)
            training_state.q_func.adjust_table_value(obs, actions, adjusts)

            # loss is only used as a metric
            loss = jnp.mean(jnp.power(target_qs - pred_qs, 2))
            metrics = { 'q_loss': loss, 'steps': training_state.steps }

            return training_state, jax.tree.map(lambda x: jnp.mean(x), metrics)

        # phases must match phases at the end of train_iteration
        set_algo_phase(training_state.q_func, AlgoPhase.OPTIMIZE)

        iterations = math.ceil(steps / self.hyperparameters.n_envs)
        training_state, metrics = nnx.scan(train_iteration)(training_state, rngs.fork(split=iterations))

        # set into eval mode for the user
        set_algo_phase(training_state.q_func, AlgoPhase.EVAL)

        return training_state, metrics