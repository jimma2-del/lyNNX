"""Implementation of Advantage Actor-Critic (A2C) - https://openai.com/index/openai-baselines-acktr-a2c/
A2C is a synchronous variant of Asynchronous Advantage Actor-Critic (A3C) - https://arxiv.org/abs/1602.01783

By default, actions are squashed to bounds using the tanh function as is commonly done in SAC.
    Log probabilities are adjusted for entropy calculations by -log(1 - tanh^2(x)).
To use basic action clipping instead, wrap the environment with `ClipActionsToBoundsWrapper(env)`.
"""

from typing import TypeVar, Generic, Any, Sequence, Self, Callable, Mapping

import math

import jax.numpy as jnp
import jax
from jax.typing import ArrayLike

from chex import dataclass
from dataclasses import field

from flax import nnx
import optax

from lynnx.utils.func_utils import try_call, optionally_pass, override_signature
from lynnx.utils.misc import compacting_mask
from lynnx.utils import RunningMeanVar
from lynnx.utils.nnx_modules import MLP, RunningMeanVarNorm, ActionDistributionHead, Pipe

from lynnx.algos.base import Scheduleable, StochasticPolicyActor, set_algo_phase, AlgoPhase, with_grad_clip

from lynnx.envs.base import Environment, Space
from lynnx.envs.wrappers import AutoResetWrapper, SquashContinuousActionsToBoundsWrapper
from lynnx.envs.utils import rollout, Timestep

@dataclass(frozen=True)
class Hyperparameters:
    """Hyperparameters for A2C.
    
    `n_envs`: Number of environments to run in parallel.
    `discount_rate`: Discount factor gamma for the environment.

    `truncated_frac`: Fraction of timesteps expected to be truncated.
        Lowering this increases performance, calling the critic on fewer extra observations.
        However, truncated timesteps exceeding this limit will be treated as terminated.

    `learning_rate`: Learning rate, used for all networks.
    `max_grad_norm`: Maximum gradient global norm, used for gradient clipping.
    `optimizer_params`: Dict of extra parameters for the optimizer.

    `rollout_length`: Number of steps to run per env per update (batch size is `rollout_length * n_envs`).

    `gae_lambda`: Factor controlling bias vs. variance tradeoff for Generalized Advantage Estimation.
        `gae_lambda=1` is equivalent to using classic n-step returns.

    `vf_coef`: Coefficient for the value loss term in the combined loss calculation.
        Controls the learning rate of the value network, as a fraction of the main learning rate.
    `ent_coef`: Coefficient for the entropy bonus term in the combined loss calculation.
        Controls how much higher entropy is favored. At `ent_coef=0`, higher entropy is not favored.

    `ent_weight_continuous`: Entropy for continuous actions is scaled by this coefficient.
        For mixed continuous-discrete action spaces, it may be helpful to reduce the weight of the continuous 
        (differential) entropy, since it tends to have a higher scale than discrete (Shannon's) entropy.

    `normalize_advantages`: If True, standardizes advantages across each rollout batch (mean=0, std=1).
    """

    n_envs: int = 256
    discount_rate: Scheduleable[float] = 0.99
    truncated_frac: float = 1.0 

    learning_rate: Scheduleable[float] = 2.5e-4
    max_grad_norm: Scheduleable[float] | None = 0.5
    optimizer_params: Mapping[str, Scheduleable[float]] = field(
        default_factory=lambda: { 'weight_decay': 0.0 })
    
    rollout_length: int = 5

    gae_lambda: Scheduleable[float] = 0.95

    vf_coef: Scheduleable[float] = 0.5
    ent_coef: Scheduleable[float] = 0.001

    ent_weight_continuous: Scheduleable[float] = 1

    normalize_advantages: bool = False
    
TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")
TEnvAction = TypeVar("TEnvAction")
TTrunkOut = TypeVar("TTrunkOut")

class Networks(nnx.Module, Generic[TEnvObs, TEnvAction, TTrunkOut]):
    """NNX module containing networks for A2C.
    
    Defaults:
        `obs_trunk`: observation standardization + flattening; no learnable parameters
        `policy_head`: hidden layers 128, 128; tanh activation; layer norm enabled
        `value_head`: hidden layers 256, 256; tanh activation; layer norm enabled
    """

    def __init__(self, obs_trunk: Callable[[TEnvObs], TTrunkOut], 
            policy_head: Callable[[TTrunkOut], TEnvAction], value_head: Callable[[TTrunkOut], jax.Array]) -> None:
        self.obs_trunk = obs_trunk
        self.policy_head = policy_head
        self.value_head = value_head

    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs | None = None) -> tuple[TEnvAction, jax.Array]:
        """Returns: action distribution, value."""
        trunk_out = optionally_pass(self.obs_trunk, rngs=rngs)(obs)

        action_dist = optionally_pass(self.policy_head, rngs=rngs)(trunk_out)
        value = optionally_pass(self.value_head, rngs=rngs)(trunk_out)

        return action_dist, value

    @classmethod
    def make_default(cls, rngs: nnx.Rngs, observation_space: Space[TEnvObs], action_space: Space[ArrayLike]) -> Self:
        return cls(
            cls.make_default_obs_trunk(observation_space),
            cls.make_default_policy_head(rngs, observation_space.flattened_dim, action_space),
            cls.make_default_value_head(rngs, observation_space.flattened_dim),
        )

    @staticmethod
    def make_default_obs_trunk(
        observation_space: Space[TEnvObs],
        normalize_observations: bool = True, 
        obs_running_mean_var: RunningMeanVar[TEnvObs] | None = None, 
        obs_clip_threshold: float | None = None
    ) -> Callable[[TEnvObs], TTrunkOut]:
        """Observation standardization + flattening; contains no learnable parameters!"""
        layers = []

        if normalize_observations:
            inp = observation_space.shapes_dtypes if obs_running_mean_var is None else obs_running_mean_var
            layers.append(RunningMeanVarNorm(inp, clip_threshold=obs_clip_threshold))

        layers.append(observation_space.flatten)

        return Pipe(*layers)

    @staticmethod
    def make_default_policy_head(
        rngs: nnx.Rngs, input_dim: int, action_space: Space[ArrayLike], do_state_independent_stds: bool = True,
        hidden_dims: Sequence[int] = (128, 128), do_layer_norm: bool = True, activation_func=nnx.tanh
    ) -> Callable[[TTrunkOut], TEnvAction]:
        head = ActionDistributionHead(action_space, do_state_independent_stds)

        mlp = MLP(
            rngs, (input_dim, *hidden_dims, head.input_dim), 
            do_layer_norm=do_layer_norm, activation_func=activation_func
        )

        return Pipe(mlp, head)

    @staticmethod
    def make_default_value_head(
        rngs: nnx.Rngs, input_dim: int,
        hidden_dims: Sequence[int] = (256, 256), do_layer_norm: bool = True, activation_func=nnx.tanh
    ) -> Callable[[TTrunkOut], jax.Array]:
        return Pipe(
            MLP(
                rngs, (input_dim, *hidden_dims, 1), 
                do_layer_norm=do_layer_norm, activation_func=activation_func
            ),
            lambda x: jnp.squeeze(x, axis=-1)
        )

@dataclass
class TrainingState(Generic[TEnvState, TEnvObs, TEnvAction, TTrunkOut]):
    """Training state for A2C."""
    steps: ArrayLike
    env_states: TEnvState

    networks: Networks[TEnvObs, TEnvAction, TTrunkOut]
    optimizer: nnx.Optimizer

class A2C(Generic[TEnvState, TEnvObs]):
    """Main class for A2C, facilitating initialization and training.
    See the module docstring for more details."""

    def __init__(self, 
        env: Environment[TEnvState, TEnvObs, TEnvAction],
        hyperparameters: Hyperparameters = Hyperparameters()
    ) -> None:
        """
        IMPORTANT: 
            `env` must already be batched; eg. wrap with `VmapWrapper` BEFORE passing in.
            `env` should not auto-reset.
        """
        self.env = env
        self.hyperparameters = hyperparameters

    def make_optax_optimizer(self, base: Callable[..., optax.GradientTransformation] = optax.adamw) \
            -> optax.GradientTransformationExtraArgs:
        """Wraps the `base` optimizer (default AdamW) with a global-norm-based gradient clipping transform,
            and `optax.inject_hyperparams`, which is necessary for this repo's 
            external environment-steps-based parameter scheduling."""
        optimizer_params = self.resolve_optimizer_params(0)

        @optax.inject_hyperparams
        @override_signature(**optimizer_params)
        def make_optimizer(**kwargs):
            return with_grad_clip(base)(**kwargs)

        return make_optimizer(**optimizer_params)

    def resolve_optimizer_params(self, steps: int = 0) -> dict[str, Any]:
        """Get values for each optimizer parameter at a particular number of env steps trained.
        Includes 'learning_rate', 'max_grad_norm', and other extra optimizer parameters."""

        return jax.tree.map(lambda x: try_call(x, steps), {
            'learning_rate': self.hyperparameters.learning_rate,
            'max_grad_norm': self.hyperparameters.max_grad_norm,
            **self.hyperparameters.optimizer_params
        })

    def make_actor(self, 
        networks: Networks[TEnvObs, TEnvAction, TTrunkOut] | None = None, 
        deterministic_sampling: bool = False, squash_continuous: bool = True,
        rngs: nnx.Rngs | None = None
    ) -> StochasticPolicyActor[TEnvObs, TEnvAction]:
        """Make an Actor (obs, rngs) -> (action, infos) using `networks`.

        `rngs` is only necessary if `networks` is not provided.
            It will be used to create default networks.
        """

        if networks is None: 
            networks = Networks.make_default(rngs, self.env.observation_space, self.env.action_space)

        return StochasticPolicyActor(
            Pipe(networks.obs_trunk, networks.policy_head), 
            self.env.action_space,
            deterministic_sampling=deterministic_sampling,
            squash_continuous=squash_continuous
        )

    def init_training_state(self, 
        rngs: nnx.Rngs, 
        networks: Networks[TEnvObs, TEnvAction, TTrunkOut] | None = None,
        optax_optimizer: optax.GradientTransformationExtraArgs | None = None,
    ) -> TrainingState[TEnvState, TEnvObs, TEnvAction, TTrunkOut]:
        """Initialize a starting training state, ready for training.
        
        Creates a default Networks and optax optimizer object if not given.
        """

        if networks is None: 
            networks = Networks.make_default(rngs, self.env.observation_space, self.env.action_space)

        if optax_optimizer is None:
            optax_optimizer = self.make_optax_optimizer()

        optimizer = nnx.Optimizer(networks, optax_optimizer)

        assert hasattr(optimizer.opt_state, 'hyperparams'), \
            "`optax_optimizer` must be initialized using a `optax.inject_hyperparams()`-wrapped function."

        handled_keys = set(optimizer.opt_state.hyperparams)
        missing_keys = set(self.resolve_optimizer_params(0)) - handled_keys
        assert not missing_keys, f"`optax_optimizer` missing hyperparams {missing_keys}; available: {handled_keys}."

        env_states, infos = jax.jit(self.env.reset)(jax.random.split(rngs.env(), self.hyperparameters.n_envs))

        return TrainingState(
            steps = jnp.array(0, dtype=jnp.int32),
            env_states = env_states,

            networks = networks,
            optimizer = optimizer,
        )
    
    def train(self, rngs: nnx.Rngs, training_state: TrainingState[TEnvState, TEnvObs, TEnvAction, TTrunkOut], steps: int) \
            -> tuple[TrainingState[TEnvState, TEnvObs, TEnvAction, TTrunkOut], dict[Any, Any]]:
        """Train from the given `training_state`, returning an updated `training_state` and metrics."""

        total_steps_per_iter = self.hyperparameters.n_envs * self.hyperparameters.rollout_length
        n_truncated = math.ceil(total_steps_per_iter * self.hyperparameters.truncated_frac)

        def train_iteration(training_state: TrainingState[TEnvState, TEnvObs, TEnvAction, TTrunkOut], rngs: nnx.Rngs) \
                -> tuple[TrainingState[TEnvState, TEnvObs, TEnvAction, TTrunkOut], dict[Any, Any]]:
            ## sample transitions from environment ##
            set_algo_phase(training_state.networks, AlgoPhase.ROLLOUT)

            actor = self.make_actor(training_state.networks, squash_continuous=False)

            (unreset_obs, timesteps), training_state.env_states, final_infos = rollout(
                rngs, SquashContinuousActionsToBoundsWrapper(self.env), actor,
                self.hyperparameters.rollout_length, self.hyperparameters.n_envs,
                training_state.env_states,

                take_func = lambda timesteps, rngs: (
                    self.env.get_obs(
                        jax.random.split(rngs.env(), self.hyperparameters.n_envs),
                        timesteps.info[AutoResetWrapper.UNRESET_STATE_INFO_KEY]
                    ), 
                    timesteps.replace(state=None, info=None) # remove unnecessary fields to save memory
                )
            )

            training_state.steps += total_steps_per_iter

            comp_unreset_obs, comp_unreset_is = compacting_mask(
                jax.tree.map(lambda x: x[1:], unreset_obs), timesteps.truncated[:-1])

            # treat truncation as termination if exceeding truncated_frac limit
            override_truncation = jnp.logical_and(timesteps.truncated[:-1], comp_unreset_is >= n_truncated)
            timesteps.terminated = timesteps.terminated.at[:-1].set(
                jnp.logical_or(override_truncation, timesteps.terminated[:-1]))

            # last timesteps should be considered truncated, so bootstrapping is used
            timesteps.truncated = timesteps.truncated.at[-1].set(True)

            # update optimizer schedules using env steps (rather than default grad steps)
            optimizer_params = self.resolve_optimizer_params(training_state.steps)
            for key, new_val in optimizer_params.items():
                training_state.optimizer.opt_state.hyperparams[key].value = new_val

            ## update networks ##
            set_algo_phase(training_state.networks, AlgoPhase.OPTIMIZE)

            discount = try_call(self.hyperparameters.discount_rate, training_state.steps)
            gae_lambda = try_call(self.hyperparameters.gae_lambda, training_state.steps)

            vf_coef = try_call(self.hyperparameters.vf_coef, training_state.steps)
            ent_coef = try_call(self.hyperparameters.ent_coef, training_state.steps)
            ent_weight_continuous = try_call(self.hyperparameters.ent_weight_continuous, training_state.steps)

            def loss_func(networks: Networks[TEnvObs, TEnvAction, TTrunkOut], rngs: nnx.Rngs):
                action_distribution, values = optionally_pass(networks, rngs=rngs)(timesteps.obs)
                const_values = jax.lax.stop_gradient(values)

                if n_truncated == 0:
                    next_values = values[1:]
                else:
                    _, unreset_values = optionally_pass(networks, rngs=rngs)(
                        jax.tree.map(lambda x: x[:n_truncated], comp_unreset_obs))

                    next_values = jnp.where(timesteps.truncated[:-1], 
                        unreset_values[comp_unreset_is], values[1:])

                final_obs = self.env.get_obs(
                    jax.random.split(rngs.env(), self.hyperparameters.n_envs),
                    final_infos[AutoResetWrapper.UNRESET_STATE_INFO_KEY]
                )

                _, final_values = optionally_pass(networks, rngs=rngs)(final_obs)
                next_values = jnp.append(next_values, final_values[None, ...], axis=0)

                next_values = jax.lax.stop_gradient(next_values)

                def gae_iter(next_gae: jax.Array, timestep: Timestep[TEnvState, TEnvObs, TEnvAction],
                    value: jax.Array, next_value: jax.Array):

                    not_terminated = jnp.logical_not(timestep.terminated)
                    not_truncated = jnp.logical_not(timestep.truncated)

                    next_gae = next_gae * not_terminated * not_truncated
                    td_err = -value + timestep.reward + discount*next_value*not_terminated

                    gae = td_err + discount*gae_lambda*next_gae

                    return gae, gae

                _, advantages = nnx.scan(gae_iter, in_axes=(nnx.Carry, 0, 0, 0), reverse=True)(
                    jnp.zeros(self.hyperparameters.n_envs),
                    timesteps, const_values, next_values
                )

                target_values = advantages + const_values
                value_loss = jnp.mean(jnp.power(target_values - values, 2)) # MSE

                if self.hyperparameters.normalize_advantages:
                    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

                log_probabilities = self.env.action_space.log_probability(
                    timesteps.action, action_distribution, continuous_squashed=False, log_stds=True)
                policy_loss = - jnp.mean(log_probabilities * advantages)

                feature_ents = self.env.action_space.entropies(action_distribution, 
                    log_stds=True, monte_carlo_n_samples=1, monte_carlo_key=rngs.actions())
                scaled_feature_ents = jax.tree.map( # reduce continuous entropy weighting
                    lambda leaf, s_dt: 
                        (1 if jnp.issubdtype(s_dt.dtype, jnp.integer) else ent_weight_continuous) * leaf,
                    feature_ents, self.env.action_space.shapes_dtypes
                )
                comb_ents = jax.tree.reduce(lambda tot, cur: tot + cur, # sum entropies
                    jax.tree.map(lambda leaf, s_dt: jnp.sum(leaf, axis=tuple(range(-len(s_dt.shape), 0))),
                        scaled_feature_ents, self.env.action_space.shapes_dtypes))
                mean_entropy = jnp.mean(comb_ents)

                comb_loss = policy_loss + vf_coef*value_loss - ent_coef*mean_entropy
                metrics = { 'policy_loss': policy_loss, 'value_loss': value_loss, 'entropy': mean_entropy }
        
                return comb_loss, metrics

            loss_grad_func = nnx.grad(loss_func, has_aux=True)
            grads, metrics = loss_grad_func(training_state.networks, rngs)
            training_state.optimizer.update(grads) 

            metrics = jax.tree.map(lambda x: jnp.mean(x), metrics)
            metrics['steps'] = training_state.steps

            return training_state, metrics

        # phases must match phases at the end of train_iteration
        set_algo_phase(training_state.networks, AlgoPhase.OPTIMIZE)

        iterations = math.ceil(steps / total_steps_per_iter)
        training_state, metrics = nnx.scan(train_iteration)(training_state, rngs.fork(split=iterations))

        # set into eval mode for the user
        set_algo_phase(training_state.networks, AlgoPhase.EVAL)

        return training_state, metrics