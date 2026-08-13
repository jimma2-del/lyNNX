"""Implementation of Soft Actor-Critic (SAC) - https://arxiv.org/abs/1801.01290

We implement a policy network and twin Q networks. 
We forego the additional value network, and add automatic entropy adjustment - https://arxiv.org/abs/1812.05905

For off-policy algorithms including SAC, we update the shared `obs_trunk` feature extractor
    only during Q function updates with the Q loss; it is kept frozen during policy updates.
"""

from typing import TypeVar, Generic, Any, Sequence, Self, Callable, Mapping, Literal

import math
from functools import reduce

import jax.numpy as jnp
import jax
from jax.typing import ArrayLike

from chex import dataclass
from dataclasses import field

from flax import nnx
import optax

from lynnx.utils.running_mean_var import RunningMeanVar
from lynnx.utils.buffers import CircularBufferWithOptionalData
from lynnx.utils.func_utils import try_call, optionally_pass, override_signature
from lynnx.utils.nnx_modules import MLP, RunningMeanVarNorm, Pipe, ActionDistributionHead

from lynnx.algos.base import Scheduleable, AlgoPhase, set_algo_phase, StochasticPolicyActor, with_grad_clip

from lynnx.envs.base import Environment, Space
from lynnx.envs.wrappers import AutoResetWrapper
from lynnx.envs.utils import rollout, Actor, RandomActor

@dataclass(frozen=True)
class Hyperparameters:
    """Hyperparameters for SAC.

    `n_envs`: Number of environments to run in parallel.
    `discount_rate`: Discount factor gamma for the environment.

    `truncated_frac`: Fraction of timesteps expected to be truncated.
        Lowering this increases performance, calling the critic on fewer extra observations.
        However, truncated timesteps exceeding this limit will be treated as terminated.

    `learning_rate`: Learning rate, used for all networks.
    `max_grad_norm`: Maximum gradient global norm, used for gradient clipping.

    `policy_optimizer_params`: Dict of extra parameters for the policy head optimizer.
    `q_func_optimizer_params`: Dict of extra parameters for the obs trunk & Q heads optimizer.
    `ent_coef_optimizer_params`: Dict of extra parameters for the entropy coefficient (alpha) optimizer,
        for automatic entropy coefficient tuning.

    `batch_size`: Minibatch size for each gradient update.
    `train_freq`: Does approximately 1 gradient step per `train_freq` environment steps.
        If `n_envs > train_freq`, we take 1 step in each env, followed by multiple gradient steps.
        NOTE: Actual number of env/gradient steps taken may be rounded if not evenly divisible.

    `policy_delay`: Updates the policy head once every `policy_delay` Q network updates.
        Actual number of updates done may be rounded if not evenly divisible.
    `polyak_tau`: Coefficient tau for target network soft Polyak averaging updates.
        Target networks are only updated on policy head updates, 
            ie. once every `train_freq * policy_delay` env steps.
        NOTE: Takes precedence over hard updates; `target_update_interval` will be ignored.

    `replay_buffer_size`: Maximum number of samples in the replay buffer.

    `ent_coef`: Entropy regularization coefficient (alpha).
        Leave as None for automatic tuning based on `target_entropy`.
    `target_entropy`: Target entropy for automatic entropy coefficient tuning when `ent_coef=None`.
        If None, this will take the default value of `-action_space.flattened_dim`.
    """

    n_envs: int = 256
    discount_rate: Scheduleable[float] = 0.99
    truncated_frac: float = 1.0

    learning_rate: Scheduleable[float] = 2.5e-4
    max_grad_norm: Scheduleable[float] | None = 10.0

    policy_optimizer_params: Mapping[str, Scheduleable[float]] = field(
        default_factory=lambda: { 'weight_decay': 0.0 })
    q_func_optimizer_params: Mapping[str, Scheduleable[float]] = field(
        default_factory=lambda: { 'weight_decay': 0.0 })
    ent_coef_optimizer_params: Mapping[str, Scheduleable[float]] = field(
        default_factory=lambda: { 'weight_decay': 0.0 })

    batch_size: int = 32
    train_freq: int = 4

    policy_delay: int = 1
    polyak_tau: Scheduleable[float] | None = None

    replay_buffer_size: int = 1_000_000

    ent_coef: Scheduleable[float] | None = None 
    target_entropy: Scheduleable[float] | None = None

class LogEntCoef(nnx.Module):
    """NNX module storing the log entropy coefficient (alpha), 
        allowing it to be tuned by an optimizer."""

    def __init__(self, log_ent_coef: ArrayLike = jnp.array(0.0)):
        self.log_ent_coef = nnx.Param(jnp.array(log_ent_coef, dtype=jnp.float32))

    @property
    def value(self):
        return self.log_ent_coef.value

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")
TEnvAction = TypeVar("TEnvAction")
TTrunkOut = TypeVar("TTrunkOut")

class Networks(nnx.Module, Generic[TEnvObs, TEnvAction, TTrunkOut]):
    """NNX module containing networks for SAC.

    Observations are first processed using a shared `obs_trunk`.
        By default, `obs_trunk` only applies standardization and flattening, with no learnable parameters.

    The trunk output is then fed to the output heads: `policy_head` and each element of `q_heads`.
        Additionally, an action is passed to each element of `q_heads` as a second argument. 
        Actions are given raw; the head is responsible for flattening.
    
    NOTE: Unlike in on-policy algorithms, in off-policy algorithms including SAC, the shared `obs_trunk` 
        is only updated during Q function updates with the Q loss; it is kept frozen during policy updates.

    Defaults:
        `obs_trunk`: observation standardization + flattening; no learnable parameters
        `policy_head`: hidden layers 256, 256; ReLU activation; layer norm enabled
        `q_heads`: hidden layers 256, 256; ReLU activation; layer norm enabled
    """

    def __init__(self, 
        obs_trunk: Callable[[TEnvObs], TTrunkOut], 
        policy_head: Callable[[TTrunkOut], TEnvAction], 
        q_heads: Sequence[Callable[[TTrunkOut, TEnvAction], jax.Array]],
        log_ent_coef: ArrayLike = jnp.array(0.0),
    ) -> None:
        self.obs_trunk = obs_trunk
        self.policy_head = policy_head
        self.q_heads = q_heads

        self.log_ent_coef = LogEntCoef(log_ent_coef)

    @classmethod
    def make_default(cls, rngs: nnx.Rngs, observation_space: Space[TEnvObs], action_space: Space[ArrayLike]) -> Self:
        return cls(
            cls.make_default_obs_trunk(observation_space),
            cls.make_default_policy_head(rngs, observation_space.flattened_dim, action_space),
            [ cls.make_default_q_head(rngs, observation_space.flattened_dim, action_space)
                for i in range(2) ]
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
        rngs: nnx.Rngs, input_dim: int, action_space: Space[ArrayLike],
        hidden_dims: Sequence[int] = (256, 256), do_layer_norm: bool = True, activation_func=nnx.relu
    ) -> Callable[[TTrunkOut], TEnvAction]:
        head = ActionDistributionHead(action_space, do_state_independent_stds=False)

        mlp = MLP(
            rngs, (input_dim, *hidden_dims, head.input_dim), 
            do_layer_norm=do_layer_norm, activation_func=activation_func
        )

        return Pipe(mlp, head)   

    @staticmethod
    def make_default_q_head(
        rngs: nnx.Rngs, input_dim: int, action_space: Space[ArrayLike],
        hidden_dims: Sequence[int] = (256, 256), do_layer_norm: bool = True, activation_func=nnx.relu
    ) -> Callable[[TTrunkOut], jax.Array]:
        return Pipe(
            lambda trunk_out, action: jnp.concatenate((trunk_out, action_space.flatten(action)), axis=-1),
            MLP(
                rngs, (input_dim + action_space.flattened_dim, *hidden_dims, 1), 
                do_layer_norm=do_layer_norm, activation_func=activation_func
            ),
            lambda x: jnp.squeeze(x, axis=-1)
        )


class TargetNetworks(nnx.Module, Generic[TEnvObs, TEnvAction, TTrunkOut]):
    """Module containing the target networks for SAC. 
    Includes `obs_trunk` and `q_heads`; leaves out `policy_head` and `log_ent_coef`."""

    def __init__(self, 
        obs_trunk: Callable[[TEnvObs], TTrunkOut], 
        q_heads: Sequence[Callable[[TTrunkOut, TEnvAction], jax.Array]],
    ) -> None:
        self.obs_trunk = obs_trunk
        self.q_heads = q_heads

    @classmethod
    def from_networks(cls, networks: Networks[TEnvObs, TEnvAction, TTrunkOut]):
        """Creates a `TargetNetworks` instance from a `Networks` instance.

        IMPORTANT: Does not deep copy parameters. Parameters will still reference the original `Networks` instance.
            To make an unlinked deep copy, do `nnx.clone(TargetNetworks.from_networks(networks))`.
        """
        return cls(networks.obs_trunk, networks.q_heads)

@dataclass(frozen=True)
class ReplayTimestep(Generic[TEnvObs, TEnvAction]):
    """Dataclass storing data for a single timestep in the replay buffer.
    The replay buffer samples `ReplayTimestep`s in pairs of (i, i+1)."""
    obs: TEnvObs
    action: TEnvAction
    reward: ArrayLike
    terminated: ArrayLike

@dataclass
class TrainingState(Generic[TEnvState, TEnvObs, TEnvAction, TTrunkOut]):
    """Training state for SAC."""
    steps: ArrayLike
    env_states: TEnvState

    networks: Networks[TEnvObs, TEnvAction, TTrunkOut]
    policy_optimizer: nnx.Optimizer # policy updates only affect the policy head, NOT the shared trunk
    q_func_optimizer: nnx.Optimizer # q func updates affect both the q heads and the shared trunk
    ent_coef_optimizer: nnx.Optimizer

    target_networks: TargetNetworks[TEnvObs, TEnvAction, TTrunkOut]
    replay_buffer: CircularBufferWithOptionalData[ReplayTimestep[TEnvObs, TEnvAction], TEnvObs]

class SAC(Generic[TEnvState, TEnvObs, TEnvAction]):
    """Main class for SAC, facilitating initialization and training.
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

        assert jax.tree.map(lambda s_dt: jnp.issubdtype(s_dt.dtype, jnp.floating), 
            env.action_space.shapes_dtypes), (
                "Action space for SAC must be continuous (jnp.floating). "
                "Support for discrete actions is in development."
            )

        self.env = env
        self.hyperparameters = hyperparameters

        # make replay buffer data shape
        self.replay_timestep_shapes_dtypes = ReplayTimestep(
            obs = self.env.observation_space.shapes_dtypes,
            action = self.env.action_space.shapes_dtypes,
            reward = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
            terminated = jax.ShapeDtypeStruct(shape=(), dtype=jnp.bool),
        )

    def make_optax_optimizer(self, 
        network_name: Literal['policy', 'q_func', 'ent_coef'],
        base: Callable[..., optax.GradientTransformation] = optax.adamw
    ) -> optax.GradientTransformationExtraArgs:
        """Wraps the `base` optimizer (default AdamW) with a global-norm-based gradient clipping transform,
            and `optax.inject_hyperparams`, which is necessary for this repo's 
            external environment-steps-based parameter scheduling."""
        optimizer_params = self.resolve_optimizer_params(network_name, 0)

        @optax.inject_hyperparams
        @override_signature(**optimizer_params)
        def make_optimizer(**kwargs):
            return with_grad_clip(base)(**kwargs)

        return make_optimizer(**optimizer_params)

    def resolve_optimizer_params(self, network_name: Literal['policy', 'q_func', 'ent_coef'], 
            steps: int = 0) -> dict[str, Any]:
        """Get values for each optimizer parameter at a particular number of env steps trained.
        Includes 'learning_rate', 'max_grad_norm', and other extra optimizer parameters.
        
        Specify the optimizer to get parameters for with `network_name`.
        """
        
        additional_params = { 
            'policy': self.hyperparameters.policy_optimizer_params, 
            'q_func': self.hyperparameters.q_func_optimizer_params,
            'ent_coef': self.hyperparameters.ent_coef_optimizer_params
        }

        return jax.tree.map(lambda x: try_call(x, steps), {
            'learning_rate': self.hyperparameters.learning_rate,
            'max_grad_norm': self.hyperparameters.max_grad_norm,
            **additional_params[network_name]
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
        policy_optax_optimizer: optax.GradientTransformationExtraArgs | None = None,
        q_func_optax_optimizer: optax.GradientTransformationExtraArgs | None = None,
        ent_coef_optax_optimizer: optax.GradientTransformationExtraArgs | None = None,
        replay_buffer: CircularBufferWithOptionalData[ReplayTimestep[TEnvObs, TEnvAction], TEnvObs] | None = None,
        prefill_steps: int = 10_000,
    ) -> TrainingState[TEnvState, TEnvObs, TEnvAction, TTrunkOut]:
        """Initialize a starting training state, ready for training.
        
        Creates a default Networks, optax optimizer, and replay buffer object if not given.

        Prefills the replay buffer with `prefill_steps` samples.
        """

        if networks is None: 
            networks = Networks.make_default(rngs, self.env.observation_space, self.env.action_space)

        optimizers = []

        for network_name, network, optax_optimizer in (
            ('policy', networks.policy_head, policy_optax_optimizer), 
            ('q_func', networks, q_func_optax_optimizer),
            ('ent_coef', networks.log_ent_coef, ent_coef_optax_optimizer)
        ):
            if optax_optimizer is None: optax_optimizer = self.make_optax_optimizer(network_name)
            optimizer = nnx.Optimizer(network, optax_optimizer)

            assert hasattr(optimizer.opt_state, 'hyperparams'), \
                "`optax_optimizer` must be initialized using a `optax.inject_hyperparams()`-wrapped function."

            handled_keys = set(optimizer.opt_state.hyperparams)
            missing_keys = set(self.resolve_optimizer_params(network_name, 0)) - handled_keys
            assert not missing_keys, f"`optax_optimizer` missing hyperparams {missing_keys}; available: {handled_keys}."

            optimizers.append(optimizer)

        policy_optimizer, q_func_optimizer, ent_coef_optimizer = optimizers

        target_networks = nnx.clone(TargetNetworks.from_networks(networks))
        set_algo_phase(target_networks, AlgoPhase.EVAL)

        # ensure target network datatypes are floats
        nnx.update(target_networks, optax.incremental_update(
            nnx.state(target_networks), nnx.state(target_networks), 0.5))

        if replay_buffer is None:
            replay_buffer = CircularBufferWithOptionalData.init(
                self.replay_timestep_shapes_dtypes, 
                self.env.observation_space.shapes_dtypes,
                int(self.hyperparameters.replay_buffer_size / self.hyperparameters.n_envs),
                optional_data_frac = self.hyperparameters.truncated_frac,
                batch_dims = self.hyperparameters.n_envs
            )

        # prefill replay buffer
        jitted_rollout = nnx.jit(self.rollout, static_argnames=('iter', 'actor'))
        timesteps, trunc, trunc_obs, env_states = jitted_rollout(rngs,
            RandomActor(self.env.action_space, self.env.observation_space),
            math.ceil(prefill_steps / self.hyperparameters.n_envs),
        )

        replay_buffer = replay_buffer.insert(timesteps, trunc, trunc_obs)

        return TrainingState(
            steps = jnp.array(0, dtype=jnp.int32),
            env_states = env_states,

            networks = networks,
            policy_optimizer = policy_optimizer,
            q_func_optimizer = q_func_optimizer,
            ent_coef_optimizer = ent_coef_optimizer,

            target_networks = target_networks,
            replay_buffer = replay_buffer,
        )

    def rollout(self,
        rngs: nnx.Rngs, 
        actor: Actor[TEnvObs, TEnvAction], 
        iter: int,
        initial_env_states: TEnvState | None = None,
    ) -> tuple[ReplayTimestep[TEnvObs, TEnvAction], jax.Array, TEnvObs, TEnvState]:
        """Collect a rollout of `ReplayTimestep`, truncated, truncated obs.

        Runs `n_envs` environments in parallel for `iter` steps each,
            for a total of `iter * n_envs` transitions.

        Initializes initial environment states if none given.
        """

        (unreset_obs, timesteps), env_states, final_infos = rollout(
            rngs, self.env, actor,
            iter, self.hyperparameters.n_envs,
            initial_env_states,

            take_func = lambda timesteps, rngs: (
                self.env.get_obs(
                    jax.random.split(rngs.env(), self.hyperparameters.n_envs), 
                    timesteps.info[AutoResetWrapper.UNRESET_STATE_INFO_KEY]
                ), 
                timesteps.replace(state=None, info=None) # remove unnecessary fields to save memory
            )
        )

        final_obs = self.env.get_obs(
            jax.random.split(rngs.env(), self.hyperparameters.n_envs), 
            final_infos[AutoResetWrapper.UNRESET_STATE_INFO_KEY]
        )
        
        next_obs = jax.tree.map(lambda middles, finals: 
                jnp.concatenate((middles[1:], finals[None, ...]), axis=0),
            unreset_obs, final_obs)

        replay_timesteps = ReplayTimestep(obs=timesteps.obs, action=timesteps.action, 
            reward=timesteps.reward, terminated=timesteps.terminated)

        return replay_timesteps, timesteps.truncated, next_obs, env_states
    
    def train(self, rngs: nnx.Rngs, training_state: TrainingState[TEnvState, TEnvObs, TEnvAction, TTrunkOut], steps: int) \
            -> tuple[TrainingState[TEnvState, TEnvObs, TEnvAction, TTrunkOut], dict[Any, Any]]:
        """Train from the given `training_state`, returning an updated `training_state` and metrics."""

        steps_per_env_per_iter = math.ceil(self.hyperparameters.train_freq / self.hyperparameters.n_envs)
        total_steps_per_iter = steps_per_env_per_iter * self.hyperparameters.n_envs
        optimize_steps_per_iter = math.ceil(self.hyperparameters.n_envs / self.hyperparameters.train_freq)
        policy_updates_per_iter = math.ceil(optimize_steps_per_iter / self.hyperparameters.policy_delay)

        def train_iteration(training_state: TrainingState[TEnvState, TEnvObs, TEnvAction, TTrunkOut], rngs: nnx.Rngs) \
                -> tuple[TrainingState[TEnvState, TEnvObs, TEnvAction, TTrunkOut], dict[Any, Any]]:
            ## sample transitions from environment ##
            set_algo_phase(training_state.networks, AlgoPhase.ROLLOUT)

            actor = self.make_actor(training_state.networks)
            timesteps, trunc, trunc_obs, training_state.env_states = self.rollout(rngs, 
                actor, steps_per_env_per_iter, training_state.env_states)
            training_state.steps += total_steps_per_iter

            training_state.replay_buffer = training_state.replay_buffer.insert(timesteps, trunc, trunc_obs)

            # update optimizer schedules using env steps (rather than default grad steps)
            for network_name, optimizer in (
                ('policy', training_state.policy_optimizer), 
                ('q_func', training_state.q_func_optimizer),
                ('ent_coef', training_state.ent_coef_optimizer)
            ):
                optimizer_params = self.resolve_optimizer_params(network_name, training_state.steps)
                for key, new_val in optimizer_params.items():
                    optimizer.opt_state.hyperparams[key].value = new_val

            ## update q functions ##
            set_algo_phase(training_state.networks, AlgoPhase.OPTIMIZE)

            manual_ent_coef = try_call(self.hyperparameters.ent_coef, training_state.steps)
            ent_coef = jnp.exp(training_state.networks.log_ent_coef.value) if manual_ent_coef is None else manual_ent_coef

            target_entropy = try_call(self.hyperparameters.target_entropy, training_state.steps)
            if target_entropy is None: target_entropy = -self.env.action_space.flattened_dim

            POLICY_METRICS_KEYS = { 'policy_loss', 'entropy', 'ent_coef_loss' }

            def optimize_step(carry, rngs: nnx.Rngs, i: jax.Array):
                networks, target_networks, policy_optimizer, q_func_optimizer, ent_coef_optimizer = carry

                # sample replay buffer
                samp_timesteps, samp_trunc, samp_trunc_obs = training_state.replay_buffer.sample(
                    rngs.optimize_samples(), seq_len=2, batch_dims=self.hyperparameters.batch_size)

                first_timestep = jax.tree.map(lambda x: x[:, 0], samp_timesteps)
                next_obs = jax.tree.map(lambda main, trunc, sd: 
                        jnp.where(samp_trunc[(slice(None), 0) + (None,)*len(sd.shape)], trunc[:, 0], main[:, 1]), 
                    samp_timesteps.obs, samp_trunc_obs, self.env.observation_space.shapes_dtypes)

                # optimize networks
                next_trunk_out = optionally_pass(networks.obs_trunk, rngs=rngs)(next_obs)
                next_action_dist = optionally_pass(networks.policy_head, rngs=rngs)(next_trunk_out)
                next_action_raw = self.env.action_space.sample_distribution(rngs.actions(), next_action_dist, 
                    squash_continuous=False, log_stds=True)
                next_action = self.env.action_space.squash_continuous_to_bounds(next_action_raw)

                target_next_trunk_out = optionally_pass(target_networks.obs_trunk, rngs=rngs)(next_obs)
                next_q = reduce(jnp.minimum, [ optionally_pass(q_head, rngs=rngs)(target_next_trunk_out, next_action) 
                    for q_head in target_networks.q_heads ])

                # entropy term
                next_ent = - self.env.action_space.log_probability(next_action_raw, next_action_dist, 
                    continuous_squashed=False, log_stds=True)
                next_q += ent_coef * next_ent

                # zero out q value if terminated
                next_q *= jnp.logical_not(first_timestep.terminated)

                target_qs = first_timestep.reward \
                    + try_call(self.hyperparameters.discount_rate, training_state.steps)*next_q

                def q_loss_func(networks: nnx.Module, rngs: nnx.Rngs):
                    trunk_out = optionally_pass(networks.obs_trunk, rngs=rngs)(first_timestep.obs)

                    qs = [ optionally_pass(q_head, rngs=rngs)(trunk_out, first_timestep.action) 
                        for q_head in networks.q_heads ]

                    q_losses = [ jnp.mean(jnp.power(target_qs - q, 2)) for q in qs ] # MSE loss

                    return sum(q_losses), (trunk_out, { f'q{i+1}_loss': q_loss for i, q_loss in enumerate(q_losses) })

                q_loss_grad_func = nnx.grad(q_loss_func, has_aux=True)
                q_grads, (trunk_out, metrics) = q_loss_grad_func(networks, rngs)
                q_func_optimizer.update(q_grads) 

                def policy_and_target_networks_update(rngs, networks, target_networks, policy_optimizer, ent_coef_optimizer):

                    def policy_loss_func(policy_head: nnx.Module, rngs: nnx.Rngs):
                        action_dist = optionally_pass(policy_head, rngs=rngs)(trunk_out)
                        action_raw = self.env.action_space.sample_distribution(rngs.actions(), action_dist, 
                            squash_continuous=False, log_stds=True)
                        action = self.env.action_space.squash_continuous_to_bounds(action_raw)

                        q = reduce(jnp.minimum, [ optionally_pass(q_head, rngs=rngs)(trunk_out, action) 
                            for q_head in networks.q_heads ])

                        log_probs = self.env.action_space.log_probability(action_raw, action_dist, 
                            continuous_squashed=False, log_stds=True)
                        entropy = - jnp.mean(log_probs)

                        policy_loss = - (jnp.mean(q) + ent_coef*entropy)
                        return policy_loss, entropy

                    policy_loss_grad_func = nnx.value_and_grad(policy_loss_func, has_aux=True)
                    (policy_loss, entropy), policy_grads = policy_loss_grad_func(networks.policy_head, rngs)
                    policy_optimizer.update(policy_grads) 

                    metrics = { 'policy_loss': policy_loss, 'entropy': entropy }

                    if manual_ent_coef is None:
                        def ent_coef_loss_func(log_ent_coef):
                            return - (log_ent_coef.value * (target_entropy - entropy))

                        ent_coef_loss_grad_func = nnx.value_and_grad(ent_coef_loss_func)
                        ent_coef_loss, ent_coef_grads = ent_coef_loss_grad_func(networks.log_ent_coef)
                        ent_coef_optimizer.update(ent_coef_grads) 

                        metrics['ent_coef_loss'] = ent_coef_loss

                    tau = try_call(self.hyperparameters.polyak_tau, training_state.steps)
                    nnx.update(target_networks, optax.incremental_update(
                        nnx.state(TargetNetworks.from_networks(networks)), nnx.state(target_networks), tau))

                    return metrics

                empty_policy_metrics = { key: jnp.array(0.0) for key in POLICY_METRICS_KEYS }
                if manual_ent_coef is not None: del empty_policy_metrics['ent_coef_loss']

                metrics |= nnx.cond(i % self.hyperparameters.policy_delay == 0, 
                    policy_and_target_networks_update, lambda *args: empty_policy_metrics,
                    rngs, networks, target_networks, policy_optimizer, ent_coef_optimizer)

                return carry, metrics

            _, metrics = nnx.scan(optimize_step, in_axes=(nnx.Carry, 0, 0))(
                (training_state.networks, training_state.target_networks, 
                    training_state.policy_optimizer, training_state.q_func_optimizer, training_state.ent_coef_optimizer), 
                rngs.fork(split=optimize_steps_per_iter),
                jnp.arange(optimize_steps_per_iter)
            )

            metrics = jax.tree.map(lambda x: jnp.mean(x), metrics)
            metrics['steps'] = training_state.steps

            if manual_ent_coef is None:
                metrics['ent_coef'] = ent_coef

            for key in POLICY_METRICS_KEYS: # account for policy delay's affect on metrics
                if key in metrics:
                    metrics[key] *= optimize_steps_per_iter / policy_updates_per_iter

            return training_state, metrics

        # ensure target network datatypes are all floats
        nnx.update(training_state.target_networks, optax.incremental_update(
            nnx.state(training_state.target_networks), nnx.state(training_state.target_networks), 0.5))

        # phases must match phases at the end of train_iteration
        set_algo_phase(training_state.target_networks, AlgoPhase.EVAL)
        set_algo_phase(training_state.networks, AlgoPhase.OPTIMIZE)

        iterations = math.ceil(steps / total_steps_per_iter)
        training_state, metrics = nnx.scan(train_iteration)(training_state, rngs.fork(split=iterations))

        # set into eval mode for the user
        set_algo_phase(training_state.networks, AlgoPhase.EVAL)

        return training_state, metrics