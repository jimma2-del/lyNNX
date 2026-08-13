"""Implementation of Deep Q-Networks (DQN) - https://arxiv.org/abs/1312.5602

We include the Double DQN add-on - https://arxiv.org/abs/1509.06461
    We also plan to add n-step returns, dueling networks, and NoisyNets.
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

from lynnx.utils import RunningMeanVar
from lynnx.utils.buffers import CircularBufferWithOptionalData
from lynnx.utils.func_utils import try_call, optionally_pass, override_signature
from lynnx.utils.nnx_modules import MLP, RunningMeanVarNorm, Pipe

from lynnx.algos.base import Scheduleable, GreedyQActor, AlgoPhase, set_algo_phase, with_grad_clip

from lynnx.envs.base import Environment, Space
from lynnx.envs.wrappers import AutoResetWrapper
from lynnx.envs.utils import rollout, Actor, RandomActor

@dataclass(frozen=True)
class Hyperparameters:
    """Hyperparameters for DQN.

    `n_envs`: Number of environments to run in parallel.
    `discount_rate`: Discount factor gamma for the environment.

    `truncated_frac`: Fraction of timesteps expected to be truncated.
        Lowering this increases performance, calling the critic on fewer extra observations.
        However, truncated timesteps exceeding this limit will be treated as terminated.

    `epsilon`: Chance to take a random action instead of a greedy action during rollouts.
        It is recommended to use a schedule: eg. decay from 1 to ~0.05 over ~10% of training steps.

    `learning_rate`: Learning rate, used for all networks.
    `max_grad_norm`: Maximum gradient global norm, used for gradient clipping.
    `optimizer_params`: Dict of extra parameters for the optimizer.

    `batch_size`: Minibatch size for each gradient update.
    `train_freq`: Does approximately 1 gradient step per `train_freq` environment steps.
        If `n_envs > train_freq`, we take 1 step in each env, followed by multiple gradient steps.
        NOTE: Actual number of env/gradient steps taken may be rounded if not evenly divisible.

    `target_update_interval`: Hard updates the target network every `target_update_interval` env steps.

    `polyak_tau`: If not None, applies a soft Polyak averaging update to the target network 
        with this as the coefficient tau once every `train_freq` env steps.
        NOTE: Takes precedence over hard updates; `target_update_interval` will be ignored.

    `replay_buffer_size`: Maximum number of samples in the replay buffer.

    `double_dqn`: If True, switches to the Double DQN formula for target calculation.
    """

    n_envs: int = 256
    discount_rate: Scheduleable[float] = 0.99
    truncated_frac: float = 1.0

    epsilon: Scheduleable[float] = 0.05

    learning_rate: Scheduleable[float] = 2.5e-4
    max_grad_norm: Scheduleable[float] | None = 10.0
    optimizer_params: Mapping[str, Scheduleable[float]] = field(
        default_factory=lambda: { 'weight_decay': 0.0 })

    batch_size: int = 32
    train_freq: int = 4

    target_update_interval: int = 10_000

    polyak_tau: Scheduleable[float] | None = None 

    replay_buffer_size: int = 1_000_000

    double_dqn: bool = False

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")
TTrunkOut = TypeVar("TTrunkOut")

class Networks(nnx.Module, Generic[TEnvObs, TTrunkOut]):
    """NNX module containing networks for DQN.

    `qs_head`: Outputs a Q value for EVERY action.
    
    Defaults:
        `obs_trunk`: observation standardization + flattening; no learnable parameters
        `qs_head`: hidden layers 128, 128; ReLU activation; layer norm enabled
    """

    def __init__(self, obs_trunk: Callable[[TEnvObs], TTrunkOut], qs_head: Callable[[TTrunkOut], jax.Array]) -> None:
        self.obs_trunk = obs_trunk
        self.qs_head = qs_head

    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Returns: an array containing a Q value for EVERY action."""
        trunk_out = optionally_pass(self.obs_trunk, rngs=rngs)(obs)
        return optionally_pass(self.qs_head, rngs=rngs)(trunk_out)

    @classmethod
    def make_default(cls, rngs: nnx.Rngs, observation_space: Space[TEnvObs], action_space: Space[ArrayLike]) -> Self:
        return cls(
            cls.make_default_obs_trunk(observation_space),
            cls.make_default_qs_head(rngs, observation_space.flattened_dim, action_space)
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
    def make_default_qs_head(
        rngs: nnx.Rngs, input_dim: int, action_space: Space[ArrayLike],
        hidden_dims: Sequence[int] = (128, 128), do_layer_norm: bool = True, activation_func=nnx.relu
    ) -> Callable[[TTrunkOut], jax.Array]:
        return MLP(
            rngs, (input_dim, *hidden_dims, int(action_space.high + 1)), 
            do_layer_norm=do_layer_norm, activation_func=activation_func
        )

@dataclass(frozen=True)
class ReplayTimestep(Generic[TEnvObs]):
    """Dataclass storing data for a single timestep in the replay buffer.
    The replay buffer samples `ReplayTimestep`s in pairs of (i, i+1)."""
    obs: TEnvObs
    action: ArrayLike
    reward: ArrayLike
    terminated: ArrayLike

@dataclass
class TrainingState(Generic[TEnvState, TEnvObs, TTrunkOut]):
    """Training state for DQN."""
    steps: ArrayLike
    env_states: TEnvState

    networks: Networks[TEnvObs, TTrunkOut]
    optimizer: nnx.Optimizer

    target_networks: Networks[TEnvObs, TTrunkOut]
    replay_buffer: CircularBufferWithOptionalData[ReplayTimestep[TEnvObs], TEnvObs]

class DQN(Generic[TEnvState, TEnvObs]):
    """Main class for DQN, facilitating initialization and training.
    See the module docstring for more details."""

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

        # make replay buffer data shape
        self.replay_timestep_shapes_dtypes = ReplayTimestep(
            obs = self.env.observation_space.shapes_dtypes,
            action = self.env.action_space.shapes_dtypes,
            reward = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
            terminated = jax.ShapeDtypeStruct(shape=(), dtype=jnp.bool),
        )

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
        networks: Networks[TEnvObs, TTrunkOut] | None = None, 
        epsilon: ArrayLike = jnp.array(0.0), 
        rngs: nnx.Rngs | None = None
    ) -> GreedyQActor[TEnvObs]:
        """Make an Actor (obs, rngs) -> (action, infos) using `networks`.

        `rngs` is only necessary if `networks` is not provided.
            It will be used to create default networks.
        """

        if networks is None: 
            networks = Networks.make_default(rngs, self.env.observation_space, self.env.action_space)

        return GreedyQActor(
            Pipe(networks.obs_trunk, networks.qs_head), 
            int(self.env.action_space.high + 1), 
            epsilon=epsilon
        )

    def init_training_state(self,
        rngs: nnx.Rngs,
        networks: Networks[TEnvObs, TTrunkOut] | None = None,
        optax_optimizer: optax.GradientTransformationExtraArgs | None = None,
        replay_buffer: CircularBufferWithOptionalData[ReplayTimestep[TEnvObs], TEnvObs] | None = None,
        prefill_steps: int = 10_000,
    ) -> TrainingState[TEnvState, TEnvObs, TTrunkOut]:
        """Initialize a starting training state, ready for training.

        Creates a default Networks, optax optimizer, and replay buffer object if not given.

        Prefills the replay buffer with `prefill_steps` samples.
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

        target_networks = nnx.clone(networks)
        set_algo_phase(target_networks, AlgoPhase.EVAL)

        if self.hyperparameters.polyak_tau is not None: # convert datatypes to floats
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
            optimizer = optimizer,

            target_networks = target_networks,
            replay_buffer = replay_buffer,
        )

    def rollout(self,
        rngs: nnx.Rngs, 
        actor: Actor[TEnvObs, ArrayLike], 
        iter: int,
        initial_env_states: TEnvState | None = None,
    ) -> tuple[ReplayTimestep[TEnvObs], jax.Array, TEnvObs, TEnvState]:
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
    
    def train(self, rngs: nnx.Rngs, training_state: TrainingState[TEnvState, TEnvObs, TTrunkOut], steps: int) \
            -> tuple[TrainingState[TEnvState, TEnvObs, TTrunkOut], dict[Any, Any]]:
        """Train from the given `training_state`, returning an updated `training_state` and metrics."""

        steps_per_env_per_iter = math.ceil(self.hyperparameters.train_freq / self.hyperparameters.n_envs)
        total_steps_per_iter = steps_per_env_per_iter * self.hyperparameters.n_envs
        optimize_steps_per_iter = math.ceil(self.hyperparameters.n_envs / self.hyperparameters.train_freq)

        def train_iteration(training_state: TrainingState[TEnvState, TEnvObs, TTrunkOut], rngs: nnx.Rngs) \
                -> tuple[TrainingState[TEnvState, TEnvObs, TTrunkOut], dict[Any, Any]]:
            ## sample transitions from environment ##
            set_algo_phase(training_state.networks, AlgoPhase.ROLLOUT)

            actor = self.make_actor(training_state.networks, 
                epsilon=try_call(self.hyperparameters.epsilon, training_state.steps))
            timesteps, trunc, trunc_obs, training_state.env_states = self.rollout(rngs, 
                actor, steps_per_env_per_iter, training_state.env_states)
            training_state.steps += total_steps_per_iter

            training_state.replay_buffer = training_state.replay_buffer.insert(timesteps, trunc, trunc_obs)

            # update optimizer schedules using env steps (rather than default grad steps)
            optimizer_params = self.resolve_optimizer_params(training_state.steps)
            for key, new_val in optimizer_params.items():
                training_state.optimizer.opt_state.hyperparams[key].value = new_val

            ## update q functions ##
            set_algo_phase(training_state.networks, AlgoPhase.OPTIMIZE)

            def optimize_step(carry: tuple[Networks, Networks, nnx.Optimizer], rngs: nnx.Rngs) \
                    -> tuple[tuple[Networks, Networks, nnx.Optimizer], dict[Any, Any]]:
                networks, target_networks, optimizer = carry

                samp_timesteps, samp_trunc, samp_trunc_obs = training_state.replay_buffer.sample(
                    rngs.optimize_samples(), seq_len=2, batch_dims=self.hyperparameters.batch_size)

                first_timestep = jax.tree.map(lambda x: x[:, 0], samp_timesteps)
                next_obs = jax.tree.map(lambda main, trunc, sd: 
                        jnp.where(samp_trunc[(slice(None), 0) + (None,)*len(sd.shape)], trunc[:, 0], main[:, 1]), 
                    samp_timesteps.obs, samp_trunc_obs, self.env.observation_space.shapes_dtypes)

                next_qs = optionally_pass(target_networks, rngs=rngs)(next_obs)

                if self.hyperparameters.double_dqn:
                    policy_net_next_act = jnp.argmax(optionally_pass(networks, rngs=rngs)(next_obs), axis=-1)
                    chosen_next_q = next_qs[jnp.arange(self.hyperparameters.batch_size), policy_net_next_act]
                else:
                    chosen_next_q = jnp.max(next_qs, axis=-1)

                # zero out q_val if terminated
                chosen_next_q *= jnp.logical_not(first_timestep.terminated)

                target_qs = first_timestep.reward \
                    + try_call(self.hyperparameters.discount_rate, training_state.steps)*chosen_next_q

                def loss_func(networks: nnx.Module, rngs: nnx.Rngs):
                    pred_qs_all_actions = optionally_pass(networks, rngs=rngs)(first_timestep.obs)
                        # q-net returns a q-value for every action
                    pred_qs = pred_qs_all_actions[jnp.arange(self.hyperparameters.batch_size), first_timestep.action]
                        # take only the q-value corresponding to the chosen action

                    # simple MSE loss
                    return jnp.mean(jnp.power(target_qs - pred_qs, 2))

                loss_grad_func = nnx.value_and_grad(loss_func)
                loss, grads = loss_grad_func(networks, rngs)
                optimizer.update(grads) 

                # update target network if using polyak averaging
                if self.hyperparameters.polyak_tau is not None:
                    tau = try_call(self.hyperparameters.polyak_tau, training_state.steps)
                    nnx.update(target_networks, optax.incremental_update(
                        nnx.state(networks), nnx.state(target_networks), tau))

                return carry, { 'q_loss': loss }

            _, metrics = nnx.scan(optimize_step)(
                (training_state.networks, training_state.target_networks, training_state.optimizer), 
                rngs.fork(split=optimize_steps_per_iter)
            )

            metrics = jax.tree.map(lambda x: jnp.mean(x), metrics)
            metrics['steps'] = training_state.steps

            # update target if enough steps have passed (not using polyak averaging)
            if self.hyperparameters.polyak_tau is None:
                update_target = training_state.steps % self.hyperparameters.target_update_interval < total_steps_per_iter
                nnx.update(training_state.target_networks, jax.lax.cond(update_target, 
                    lambda opt_state, target_state: opt_state, 
                    lambda opt_state, target_state: target_state,
                    nnx.state(training_state.networks), nnx.state(training_state.target_networks)
                ))

            return training_state, metrics

        if self.hyperparameters.polyak_tau is not None: # convert datatypes to floats
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