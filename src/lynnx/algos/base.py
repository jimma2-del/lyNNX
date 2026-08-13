"""Base classes and types for RL algorithms."""

from typing import TypeVar, Generic, Protocol, TypeAlias, Sequence, Any, Iterable, Mapping, Callable, ParamSpec
from enum import Enum
from functools import wraps

from abc import ABC, abstractmethod
import dataclasses

import jax.numpy as jnp
import jax

from jax.typing import ArrayLike
import chex

from flax import nnx
import optax

from lynnx.utils.func_utils import optionally_pass, override_signature, try_call
from lynnx.utils.batch_utils import get_tree_batch_dims
from lynnx.envs.base import Space, Environment
#from lynnx.envs.utils import Actor

TScheduleValue = TypeVar('TScheduleValue')

class Schedule(Generic[TScheduleValue], Protocol):
    def __call__(self, steps: int) -> TScheduleValue:
        ...

Scheduleable: TypeAlias = TScheduleValue | Schedule[TScheduleValue]

class AlgoPhase(Enum):
    ROLLOUT = 'rollout'
    OPTIMIZE = 'optimize'
    EVAL = 'eval'

def set_algo_phase(module: nnx.Module, phase: AlgoPhase) -> None:
    if phase == AlgoPhase.OPTIMIZE:
        module.train(algo_phase=phase)
    else:
        module.eval(algo_phase=phase)

TEnvObs = TypeVar("TEnvObs")
TEnvAction = TypeVar("TEnvAction")

class PolicyWithRngs(Generic[TEnvObs, TEnvAction], Protocol):
    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs) -> TEnvAction: ...
class PolicyWithoutRngs(Generic[TEnvObs, TEnvAction], Protocol):
    def __call__(self, obs: TEnvObs) -> TEnvAction: ...
Policy: TypeAlias = PolicyWithoutRngs[TEnvObs, TEnvAction] | PolicyWithRngs[TEnvObs, TEnvAction]

class StochasticPolicyActor(Generic[TEnvObs, TEnvAction], nnx.Module):
    """Actor which chooses actions by sampling from a distribution outputted by the policy."""

    def __init__(self, policy: Policy[TEnvObs, TEnvAction], action_space: Space[TEnvAction],
            deterministic_sampling: bool = False, squash_continuous: bool = True) -> None:
        self.policy = policy
        self.action_space = action_space

        # configurations for `self.__call__()`
        self.deterministic_sampling = deterministic_sampling
        self.squash_continuous = squash_continuous

    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs | None = None,
        deterministic_sampling: bool | None = None, squash_continuous: bool | None = None
    ) -> tuple[TEnvAction, dict[Any, Any]]:
        """Computes an action distribution for the given observation, and samples an action from it.
        Returns the raw action distribution in `info['action_dist']`.

        `squash_continuous`: If False, does not squash continuous values, leaving them unbounded.
            Useful, eg. for sampling raw outputs, before softplus or tanh.

        `deterministic_sampling`: If True, returns a deterministc representative value of the action distribution 
            instead of sampling a random value: either the median or the mode. See `Space.sample_distribution()`.
        """

        if deterministic_sampling is None: deterministic_sampling = self.deterministic_sampling
        if squash_continuous is None: squash_continuous = self.squash_continuous

        action_dist = self.action_distribution(obs, rngs)
        return self.action_space.sample_distribution(rngs.actions(), action_dist, log_stds=True,
            squash_continuous=squash_continuous, deterministic=deterministic_sampling), {'action_dist': action_dist}

    def action_distribution(self, obs, rngs: nnx.Rngs | None = None) -> TEnvAction:
        """Applies the policy on the observation to compute an action distribution.
        See `Space.sample_distribution()` for details on the structure of the returned distribution."""
        return optionally_pass(self.policy, rngs=rngs)(obs)

class DeterministicPolicyActor(Generic[TEnvObs, TEnvAction], nnx.Module):
    """Actor which gets actions directly from the policy, and optionally adds gaussian noise.
    Currently only supports continuous actions."""

    def __init__(self, policy: Policy[TEnvObs, TEnvAction], action_space: Space[TEnvAction],
            noise: ArrayLike = jnp.array(0.0)) -> None:

        assert jax.tree.map(lambda s_dt: jnp.issubdtype(s_dt.dtype, jnp.floating), 
            action_space.shapes_dtypes), "Action space must be continuous (jnp.floating)."

        self.policy = policy
        self.action_space = action_space

        self.noise = nnx.Variable(jnp.array(noise, dtype=jnp.float32))

    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs | None = None,
            noise: ArrayLike | None = None) -> tuple[TEnvAction, dict[Any, Any]]:
        """Computes an action using the policy, optionally adding gaussian noise.
        Returns the raw action without noise in `info['raw_action']`."""

        if noise is None: noise = self.noise.value

        raw_action = optionally_pass(self.policy, rngs=rngs)(obs)
        action = self.action_space.add_noise_to_continuous(rngs.action(), raw_action, noise)

        return action, {'raw_action': raw_action}

class DiscreteQFuncWithRngs(Generic[TEnvObs], Protocol):
    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs) -> jax.Array: ...
class DiscreteQFuncWithoutRngs(Generic[TEnvObs], Protocol):
    def __call__(self, obs: TEnvObs) -> jax.Array: ...
DiscreteQFunc: TypeAlias = DiscreteQFuncWithRngs[TEnvObs] | DiscreteQFuncWithoutRngs[TEnvObs]

class GreedyQActor(Generic[TEnvObs], nnx.Module):
    """Actor which chooses discrete (scalars in the range [0, num_actions-1]) actions
        taking the action with the highest Q value out of the Q values outputted by the Q function.
    
    Supports epsilon-greedy actions, returning a random action instead with probability epsilon.
    """

    def __init__(self, q_func: DiscreteQFunc[TEnvObs], num_actions: int, epsilon: ArrayLike = jnp.array(0.0)) -> None:
        self.q_func = q_func
        self.num_actions = int(num_actions)

        self.epsilon = nnx.Variable(jnp.array(epsilon, dtype=jnp.float32))

    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs | None = None,
            epsilon: ArrayLike | None = None) -> tuple[TEnvAction, dict[Any, Any]]:
        """Returns a random action with probablity `epsilon`, otherwise 
            computes a Q value for every possible action and returns the action with the highest Q value.
        Returns the raw Q values in `info['q_values']`, and whether a random action was used in `info['action_random']`.
        """

        if epsilon is None: epsilon = self.epsilon.value

        q_values = self.q_values(obs, rngs=rngs)
        greedy_action = self.select_greedy_action(q_values)

        random_action = self.random_action(rngs, shape=greedy_action.shape)
        take_random_action = jax.random.bernoulli(rngs.actions(), p=epsilon, shape=greedy_action.shape)

        return jnp.where(take_random_action, random_action, greedy_action), \
            { 'q_values': q_values, 'action_random': take_random_action }

    def q_values(self, obs, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Applies the Q function on the observation to compute a Q value for every possible action."""
        return optionally_pass(self.q_func, rngs=rngs)(obs)

    def select_greedy_action(self, q_values: jax.Array) -> ArrayLike:
        """Returns the action with the highest Q value out of the Q values given."""
        return jnp.argmax(q_values, axis=-1)

    def greedy_action(self, obs: TEnvObs, rngs: nnx.Rngs | None = None) -> ArrayLike:
        """Computes a Q value for every possible action and returns the action with the highest Q value."""
        return self.find_greedy_action(self.q_values(obs, rngs=rngs))

    def random_action(self, rngs: nnx.Rngs, shape: Sequence[int] = ()) -> ArrayLike:
        return jax.random.randint(rngs.actions(), shape=shape, minval=0, maxval=self.num_actions)

class ValueFuncWithRngs(Generic[TEnvObs], Protocol):
    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs) -> jax.Array: ...
class ValueFuncWithoutRngs(Generic[TEnvObs], Protocol):
    def __call__(self, obs: TEnvObs) -> jax.Array: ...
ValueFunc: TypeAlias = ValueFuncWithRngs[TEnvObs] | ValueFuncWithoutRngs[TEnvObs]

P = ParamSpec('P')

class OptimizerFactoryWithGradClip(Generic[P], Protocol):
    def __call__(self, *args: P.args, max_grad_norm: Scheduleable[float] | None = None, 
        **kwargs: P.kwargs) -> optax.GradientTransformationExtraArgs: ...

def with_grad_clip(optimizer_factory: Callable[P, optax.GradientTransformation]) -> OptimizerFactoryWithGradClip[P]:
    """Adds a `max_grad_norm` keyword parameter to the factory, which if set (and is not None),
        adds a `optax.clip_by_global_norm(max_grad_norm)` transform to the front of the returned optimizer."""

    @wraps(optimizer_factory)
    def make_optimizer(*args: P.args, max_grad_norm: Scheduleable[float] | None = None, 
            **kwargs: P.kwargs) -> optax.GradientTransformationExtraArgs:

        return optax.chain(
            *( ( optax.clip_by_global_norm(max_grad_norm), ) 
                if max_grad_norm is not None else () ),
            optimizer_factory(*args, **kwargs),
        )

    return make_optimizer

### UNOFFICIAL Algo spec; not currently enforced, subject to change ###

# TEnvState = TypeVar('TEnvState')
# TRenderState = TypeVar('TRenderState')

# @chex.dataclass
# class TrainingState(Generic[TEnvState]):
#     """Base class for algo training states.
#     Common additional fields include: networks, optimizer, target_networks, replay_buffer, etc."""
#     steps: jax.Array
#     env_states: TEnvState

# @chex.dataclass
# class Hyperparameters:
#     """Base class for algo hyperparmeters."""
#     n_envs: int = 256
#     discount_rate: Scheduleable[float] = 0.99

#     learning_rate: Scheduleable[float] = 2.5e-4
#     max_grad_norm: Scheduleable[float] | None = 10.0
#     optimizer_params: Mapping[str, Scheduleable[float]] = field(
#         default_factory=lambda: { 'weight_decay': 0.0 })

# class Algo(Generic[TEnvState, TEnvObs, TEnvAction, TRenderState]):

#     def __init__(self,
#         env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderState], 
#         hyperparameters: Hyperparameters
#     ) -> None:
#         self.env = env
#         self.hyperparameters = hyperparameters

#     # def init_training_state(rngs, networks, optional params (eg. replay buffer state, prefill steps)) -> TTrainingState

#     def train(rngs, training_state, steps) -> TTrainingState, metrics dict

#     make_actor(networks (or TrainingState?), optional rngs, optional params?)? 
#         can be used as dummy for loading orbax checkpoints if only the actor was saved

#     save/load as a parent method? option to only include actor