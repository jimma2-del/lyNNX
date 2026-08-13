"""Utility functions for environments."""

import math

from typing import Any, Generic, Callable, Protocol, TypeAlias, ParamSpec, Sequence
from typing_extensions import TypeVar

from functools import wraps

import chex

import jax
from jax.typing import ArrayLike

import jax.numpy as jnp

from flax import nnx

from lynnx.utils.batch_utils import get_tree_batch_dims
from lynnx.utils.func_utils import optionally_pass
from lynnx.utils.nnx_modules import Pipe

from lynnx.envs.base import Environment, Space
from lynnx.envs.wrappers import AutoResetWrapper

TEnvState = TypeVar("TEnvState")
TEnvObs = TypeVar("TEnvObs")
TEnvAction = TypeVar("TEnvAction")
TRenderFrame = TypeVar("TRenderFrame", default=None)

class ActorWithRngs(Generic[TEnvObs, TEnvAction], Protocol):
    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs) -> tuple[TEnvAction, dict[Any, Any]]: ...
class ActorWithoutRngs(Generic[TEnvObs, TEnvAction], Protocol):
    def __call__(self, obs: TEnvObs) -> tuple[TEnvAction, dict[Any, Any]]: ...
Actor: TypeAlias = ActorWithoutRngs[TEnvObs, TEnvAction] | ActorWithRngs[TEnvObs, TEnvAction]

P = ParamSpec("P")
R = TypeVar("R")

def with_info(func: Callable[P, R]) -> Callable[P, tuple[R, dict[Any, Any]]]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[R, dict[Any, Any]]:
        return func(*args, **kwargs), {}
    return wrapper

class RandomActor(Generic[TEnvObs, TEnvAction]):
    """Actor which chooses actions by sampling from a uniform distribution."""

    def __init__(self, action_space: Space[TEnvAction], 
            observation_space: Space[TEnvObs] | None = None) -> None:
        """
        `observation_space` is needed to determine batch dims if processing batched observations.
            If not given, `__call__` will always only output a single action, regardless of `obs`.
        """
        self.action_space = action_space
        self.observation_space = observation_space

    def __call__(self, obs: TEnvObs, rngs: nnx.Rngs) -> tuple[TEnvAction, dict[Any, Any]]:
        batch_dims = () if self.observation_space is None \
            else get_tree_batch_dims(self.observation_space.shapes_dtypes, obs)
        return self.action_space.sample(rngs.actions(), batch_dims=batch_dims), {}

@chex.dataclass
class Timestep(Generic[TEnvState, TEnvObs, TEnvAction]):
    state: TEnvState
    obs: TEnvObs
    action: TEnvAction
    action_info: dict[Any, Any]
    reward: ArrayLike
    info: dict[Any, Any]
    terminated: bool
    truncated: bool

TTakeObj = TypeVar('TTakeObj', default=Timestep[TEnvState, TEnvObs, TEnvAction])

class TakeFuncWithRngs(Generic[TEnvState, TEnvObs, TEnvAction, TTakeObj], Protocol):
    def __call__(self, timestep: Timestep[TEnvState, TEnvObs, TEnvAction], rngs: nnx.Rngs) -> TTakeObj: ...
class TakeFuncWithoutRngs(Generic[TEnvState, TEnvObs, TEnvAction, TTakeObj], Protocol):
    def __call__(self, timestep: Timestep[TEnvState, TEnvObs, TEnvAction]) -> TTakeObj: ...
TakeFunc: TypeAlias = TakeFuncWithoutRngs[TEnvState, TEnvObs, TEnvAction, TTakeObj] \
    | TakeFuncWithRngs[TEnvState, TEnvObs, TEnvAction, TTakeObj]

def rollout(rngs: nnx.Rngs,
    env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
    actor: Actor[TEnvObs, TEnvAction],
    iters: int,
    env_batch_dims: int | Sequence[int] = (),
    initial_env_state: TEnvState | None = None,
    initial_env_info: dict[Any, Any] | None = None,
    take_func: TakeFunc[TEnvState, TEnvObs, TEnvAction, TTakeObj] | None = None,
) -> tuple[TTakeObj, TEnvState, dict[Any, Any]]:
    """Runs the environment for `iters` steps, returning collected timesteps or user-defined take objs.

    Automatically resets environments that finish.
        Places the original, unresetted state into `info[UNRESET_STATE_INFO_KEY]` (useful eg. for truncation)
        This will be the same as the returned new_state if not terminated and not truncated.

    If using batched env states, ensure both `env` and `actor` can handle batches BEFORE passing in.
        Eg. wrap environment with `VmapWrapper`, wrap actor with `nnx.split_rngs(splits=n_envs)(nnx.vmap(actor))`.
            Actor should take batched env states, but a SINGLE, unbatched nnx.Rngs.
        Additionally, `env_batch_dims` MUST be specified. `env_batch_dims=()` (default) indicates no batching.

    If `initial_env_state` is given but `initial_env_info` is not given,
        the first info will be a dummy value.

    `take_func`: Optional function to specify which values to take from Timestep.
        A full `Timestep` object is returned by default.
        If the environment is batched, `take_func` should also handle batched timesteps.

    Returns: timesteps, or user defined take objs for each timestep; final states; final infos.
        If the environment is batched, timesteps/take objs will have 
            the following extra leading axes: `shape = (iters, *env_batch_dims, ...)`
    """

    if not isinstance(env, AutoResetWrapper):
        env = AutoResetWrapper(env)

    if initial_env_state is None:
        initial_env_state, initial_env_info = env.reset(jax.random.split(rngs.env(), env_batch_dims))

    if initial_env_info is None:
        _, initial_env_info = env.reset(jax.random.split(jax.random.key(0), env_batch_dims)) # dummy value

    if take_func is None:
        take_func = lambda x: x

    actor = Pipe(actor)

    def batched_env_step(carry: tuple[TEnvState, dict[Any,Any]], rngs: nnx.Rngs) -> tuple[TEnvState, TTakeObj]:
        actor, states, infos = carry

        obs = env.get_obs(jax.random.split(rngs.env(), env_batch_dims), states)
        actions, action_infos = optionally_pass(actor, rngs=rngs)(obs)

        new_states, rewards, terminateds, truncateds, new_infos = env.step(
            jax.random.split(rngs.env(), env_batch_dims), states, actions)

        return (actor, new_states, new_infos), optionally_pass(take_func, rngs=rngs)(Timestep(
            state=states, obs=obs, action=actions, reward=rewards, 
            info=infos, terminated=terminateds, truncated=truncateds, action_info=action_infos)
        )

    (actor, env_states, infos), take_values = nnx.scan(batched_env_step)(
        (actor, initial_env_state, initial_env_info), rngs.fork(split=iters))

    return take_values, env_states, infos

def evaluate_episodes(rngs: nnx.Rngs, 
    env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame], 
    actor: Actor[TEnvObs, TEnvAction], 
    episodes: int, 
    n_envs: int | None = None,
    eps_steps_limit: int = None
) -> tuple[jax.Array, jax.Array]:
    """Runs the actor in `n_envs` environments in parallel until completing `episodes` episodes, 
    returning an array of trajectory returns and and an array of trajectory lengths.

    Do NOT wrap the environment with `AutoResetWrapper` before passing in.

    It is recommended to pass a batched environment (eg. wrapped with `VmapWrapper`) for increased speed.
        Must specify `n_envs` (number of parallel environments) if passed environment is batched.
        If `n_envs` is specified, the environment must be batched BEFORE passing in.
        `actor` should also accept a batched input of `obs`, but only a single `rngs`. 
            Apply `nnx.split_rngs(splits=n_envs)(nnx.vmap(actor))` before passing in if not already batched.

    Runs at LEAST `episodes` episodes. If `episodes` is not a multiple of `n_envs`, will run the next multiple.
    """

    if not isinstance(env, AutoResetWrapper):
        env = AutoResetWrapper(env)

    keys_shape = () if n_envs is None else (n_envs,)
    actor = Pipe(actor)

    eps_per_env = math.ceil(episodes / (n_envs if n_envs is not None else 1))

    def batched_env_step(carry):
        actor, eps_done, rngs, env_state, cur_return, eps_returns, cur_len, eps_lens = carry

        # step env
        obs = env.get_obs(jax.random.split(rngs.env(), keys_shape), env_state)
        action, action_info = optionally_pass(actor, rngs=rngs)(obs)
        env_state, reward, terminated, truncated, info = env.step(
            jax.random.split(rngs.env(), keys_shape), env_state, action)

        if eps_steps_limit is not None:
            truncated = jnp.logical_or(truncated, cur_len >= eps_steps_limit)

        cur_return = cur_return + reward
        cur_len = cur_len + 1
        done = jnp.logical_or(terminated, truncated)

        # add metrics to aggregate array of eps metrics if done
        eps_returns = eps_returns.at[jnp.arange(n_envs), eps_done].set(cur_return)
        eps_lens = eps_lens.at[jnp.arange(n_envs), eps_done].set(cur_len)

        eps_done = jnp.where(done, eps_done + 1, eps_done)

        # reset cur metrics if done
        not_done = jnp.logical_not(done)
        cur_return = cur_return * not_done
        cur_len = cur_len * not_done

        return actor, eps_done, rngs, env_state, cur_return, eps_returns, cur_len, eps_lens

    env_state, info = env.reset(jax.random.split(rngs.env(), keys_shape))

    actor, eps_done, rngs, env_state, cur_return, eps_returns, cur_len, eps_lens = nnx.while_loop(
        lambda input: jnp.any(input[1] < eps_per_env), 
        batched_env_step, 
        (
            actor,
            jnp.zeros(n_envs, dtype=jnp.int32), 
            rngs, 
            env_state, 
            jnp.zeros(n_envs), 
            jnp.empty((n_envs, eps_per_env)), 
            jnp.zeros(n_envs, dtype=jnp.int32), 
            jnp.empty((n_envs, eps_per_env), dtype=jnp.int32)
        )
    )

    return eps_returns.flatten(), eps_lens.flatten()

def rollout_episode(rngs: nnx.Rngs, 
    env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame], 
    actor: Actor[TEnvObs, TEnvAction],
    chunk_size: int = 100,
    take_func: TakeFunc[TEnvState, TEnvObs, TEnvAction, TTakeObj] | None = None,
) -> tuple[TTakeObj, TEnvState, dict[Any, Any]]:
    """Rollout one episode. Intended for visualization, rather than training. Do NOT JIT.
    Performs rollout in chunks of `chunk_size` steps for speed, combining chunks at the end.

    `take_func`: Optional function to specify which values to take from Timestep
        A full `Timestep` object is returned by default.

    Returns: 
        timesteps, or user defined take objs for each timestep;
        final timestep/take obj, after the termination/truncation (may be useful eg. if truncated);
    """

    if take_func is None:
        take_func = lambda x: x

    actor = Pipe(actor)

    @nnx.jit
    @nnx.scan
    def rollout(carry, rngs):
        actor, state, info = carry

        obs = env.get_obs(rngs.env(), state)
        action, action_info = optionally_pass(actor, rngs=rngs)(obs)

        next_state, reward, terminated, truncated, next_info = env.step(rngs.env(), state, action)

        timestep = Timestep(state=state, obs=obs, action=action, reward=reward, info=info,
            terminated=terminated, truncated=truncated, action_info=action_info)

        return (actor, next_state, next_info), (optionally_pass(take_func, rngs=rngs)(timestep), terminated, truncated)

    state, info = jax.jit(env.reset)(rngs.env())

    comb_take_vals = None
    done = False
    
    while not done:
        (actor, state, info), (take_vals, terminated, truncated) = rollout(
            (actor, state, info), rngs.fork(split=chunk_size))

        dones = jnp.logical_or(terminated, truncated)
        done_idx = jnp.argmax(dones)
        done = dones[done_idx]

        if done:
            take_vals = jax.tree.map(lambda x: x[:done_idx+1], take_vals)

            if done_idx == len(dones) - 1:
                (actor, state, info), (final_take_vals, terminated, truncated) = rollout(
                    (actor, state, info), rngs.fork(split=1))
                final_take_val = jax.tree.map(lambda x: x[0], final_take_vals)
            else:
                final_take_val = jax.tree.map(lambda x: x[done_idx + 1], take_vals)

        if comb_take_vals is None:
            comb_take_vals = take_vals
        else: # append to existing
            comb_take_vals = jax.tree.map(lambda comb, new: jnp.concatenate((comb, new), axis=0), 
                comb_take_vals, take_vals)
    
    return comb_take_vals, final_take_val

def stagger_env_states(rngs: nnx.Rngs,
    env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame],
    n_envs: int,
    stagger_step_size: int = 32,
    initial_env_states: TEnvState | None = None
) -> tuple[TTakeObj, TEnvState, dict[Any, Any]]:
    """Staggers environment states by stepping for different numbers of steps.
        Envs will be staggered by `jnp.arange(n_envs) * stagger_step_size` steps respectively.
    
    This is necessary for envs with fixed episode lengths, where all states are reset at the same time.
        Staggering states prevents cyclical nonstationarity due to synced episode phases.
        See https://arxiv.org/abs/2511.21011 for more details. 
            We deviate from this paper as we find that dividing into groups is unnecessary.

    `env` must be able to handle batched states. Eg. wrap with `VmapWrapper` BEFORE passing in.
    """

    if initial_env_states is None:
        initial_env_states, _= env.reset(jax.random.split(rngs.env(), n_envs))

    needed_steps = jnp.arange(n_envs) * stagger_step_size

    def step_envs(carry: tuple[TEnvState, jax.Array], rngs: nnx.Rngs) -> TEnvState:
        states, steps = carry

        actions = env.action_space.sample(rngs.actions(), batch_dims=(n_envs,))
        new_states, _, _, _, _ = env.step(jax.random.split(rngs.env(), n_envs), states, actions)
        steps = steps + 1

        def where_done(cur, new):
            # ignore fields with custom vmapping rules
            if not hasattr(new, 'shape') or new.shape[0] != needed_steps.shape[0]: return new.copy()
                
            return jnp.where((steps > needed_steps)[(slice(None),) + (None,)*(cur.ndim - 1)], cur, new)

        next_states = jax.tree.map(where_done, states, new_states)

        return next_states, steps

    carry = initial_env_states, jnp.array(0, dtype=jnp.int32)
    states, _ = nnx.scan(step_envs, out_axes=nnx.Carry)(carry, rngs.fork(split=(n_envs-1)*stagger_step_size))

    return states

def visualize_pygame(rngs: nnx.Rngs, 
    env: Environment[TEnvState, TEnvObs, TEnvAction, TRenderFrame], 
    actor: Actor[TEnvObs, TEnvAction],
    window_size: tuple[int, int] | None = None, 
    fps: int = 10,
    render_func: Callable[[TEnvState, TEnvAction], ArrayLike] | None = None,
    verbose: bool = False,
    episode_steps_limit: int | None = None,
) -> tuple[Timestep, bool]:
    """Visualizes episodes, rendering in pygame, until closed.
    `render_func` should output an array of pixels (y, x, r, g, b). Defaults to `env.render` if not given.
    Does not JIT wrap the environment or actor. Try JIT wrapping before passing in if slow,
        though this may incur penalties due to the slow transfer of data between devices.
    """
    import pygame
    import numpy as np

    if render_func is None:
        render_func = env.render

    state, info = env.reset(rngs.env())

    if window_size is None: # infer window_size from render_func
        dummy_img = render_func(state, env.action_space.sample(jax.random.key(0)))
        window_size = dummy_img.shape[-2::-1] # first two dims, swapped

    steps = 0
    eps_return = 0
    terminated = False
    truncated = False

    pygame.init()
    clock = pygame.time.Clock()
    pygame.display.set_caption("Environment Visualization")
    screen = pygame.display.set_mode(window_size)

    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        if terminated or truncated:
            print(f"{'Terminated' if terminated else 'Truncated'} at steps={steps}, return={eps_return}\n")

            state, info = env.reset(rngs.env())
            steps = 0
            eps_return = 0
            terminated = False
            truncated = False
        
        else:
            obs = env.get_obs(rngs.env(), state)
            action, action_info = optionally_pass(actor, rngs=rngs)(obs)

            state, reward, terminated, truncated, info = env.step(rngs.env(), state, action)

            steps += 1
        
            if reward != 0 or verbose:
                eps_return += reward
                print(f"steps={steps}, reward={reward}, return={eps_return}", end="")

                if verbose:
                    print(f" obs={obs}, action={action}", end="")
                
                print()

        image_array = np.asarray(render_func(state, action))
        pygame_surface = pygame.surfarray.make_surface(image_array.swapaxes(0,1))
        screen.blit(pygame_surface, (0,0))
        pygame.display.flip()

        clock.tick(fps)

        truncated = truncated or (episode_steps_limit is not None and steps >= episode_steps_limit)

    pygame.quit()