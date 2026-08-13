"""Utility for converting a gymnax `Environment`/`Space` into an `Environment`/Space` following the lyNNX API.

NOTE: These "wrappers" are unrelated to the Wrapper type in wrappers.py, 
    which take in a lyNNX Environment, rather than an environment from an external library.
"""

from typing import Any, Generic
from typing_extensions import TypeVar
from abc import ABC, abstractmethod

import numpy as np

import jax
from jax.typing import ArrayLike
import chex

import jax.numpy as jnp

from gymnax.environments.environment import Environment as GymnaxEnv
import gymnax.environments.spaces as GymnaxSpaces

from .gymnax_vis import render_env

from lynnx.envs.base import Environment, Space

def space_from_gymnax_space(gymnax_space: GymnaxSpaces.Space) -> Space:
    if isinstance(gymnax_space, GymnaxSpaces.Discrete):
        return Space(low=np.array(0, dtype=np.int32), high=np.array(gymnax_space.n - 1, dtype=np.int32))

    if isinstance(gymnax_space, GymnaxSpaces.Box):
        return Space(
            low=np.asarray(jnp.broadcast_to(gymnax_space.low, gymnax_space.shape).astype(jnp.float32)), 
            high=np.asarray(jnp.broadcast_to(gymnax_space.high, gymnax_space.shape).astype(dtype=jnp.float32))
        )

    if isinstance(gymnax_space, GymnaxSpaces.Dict):
        sub_spaces = { key: space_from_gymnax_space(space) for key, space in gymnax_space.spaces.items() }
        low = { key: space.low for key, space in sub_spaces.items() }
        high = { key: space.high for key, space in sub_spaces.items() }
        return Space(low=low, high=high)

    if isinstance(gymnax_space, GymnaxSpaces.Tuple):
        sub_spaces = [ space_from_gymnax_space(space) for space in gymnax_space.spaces ]
        low = [ space.low for space in sub_spaces ]
        high = [ space.high for space in sub_spaces ]
        return Space(low=low, high=high)

    raise ValueError("Unknown Gymnax Space type.")

TEnvState = TypeVar("TEnvState")
TEnvParams = TypeVar("TEnvParams")

class GymnaxWrapper(Environment[TEnvState, ArrayLike, ArrayLike, np.ndarray], Generic[TEnvState, TEnvParams]):
    """Wrapper for Gymnax environments."""

    def __init__(self, gymnax_env: GymnaxEnv[TEnvState, TEnvParams], gymnax_params: TEnvParams | None = None):
        self.gymnax_env = gymnax_env
        self.gymnax_params = gymnax_params if gymnax_params is not None else gymnax_env.default_params

        self._observation_space = space_from_gymnax_space(self.gymnax_env.observation_space(self.gymnax_params))
        self._action_space = space_from_gymnax_space(self.gymnax_env.action_space(self.gymnax_params))

        # info shapes dtypes needed to make dummy info for self.reset(), since gymnax does not provide info on resets
        def get_step_info():
            _, dummy_state = gymnax_env.reset_env(jax.random.key(0), self.gymnax_params)
            _, _, _, _, info = self.step(jax.random.key(0), 
                dummy_state, 
                self.action_space.sample(jax.random.key(0))
            )

            return info

        self.info_shapes_dtypes = jax.eval_shape(get_step_info)

    def reset(self, key: chex.PRNGKey) -> tuple[TEnvState, dict[Any, Any]]:
        obs, state = self.gymnax_env.reset_env(key, self.gymnax_params)
        dummy_info = jax.tree.map(lambda s_dt: jnp.empty(s_dt.shape, dtype=s_dt.dtype), self.info_shapes_dtypes)
        return state, dummy_info

    def step(self, key: chex.PRNGKey, state: TEnvState, action: ArrayLike) \
        -> tuple[TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        obs, state, reward, terminated, info = self.gymnax_env.step_env(key, state, action, self.gymnax_params)
        return state, reward, terminated, False, info

    def get_obs(self, key: chex.PRNGKey, state: TEnvState) -> ArrayLike:
        return self.gymnax_env.get_obs(state=state, params=self.gymnax_params, key=key)

    def render(self, state: TEnvState, action: ArrayLike) -> np.ndarray:
        return render_env(self.gymnax_env, state, self.gymnax_params)

    @property
    def observation_space(self) -> Space[ArrayLike]:
        return self._observation_space

    @property
    def action_space(self) -> Space[ArrayLike]:
        return self._action_space

if __name__ == "__main__":
    space = space_from_gymnax_space(GymnaxSpaces.Dict({
        "foo": GymnaxSpaces.Box(-10, 12, (), jnp.float32),
        "foo2": GymnaxSpaces.Box(-10, 12, (2,3), jnp.float32),
        "bar": GymnaxSpaces.Discrete(5),
        "foobar": GymnaxSpaces.Tuple((
            GymnaxSpaces.Box(-10, 12, (), jnp.float32),
            GymnaxSpaces.Discrete(5)
        ))
    }))

    print("low", space.low)
    print("high", space.high)

    key = jax.random.key(0)
    print("sample", space.sample(key))