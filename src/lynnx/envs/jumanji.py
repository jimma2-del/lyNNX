"""Utility for converting a Jumanji `Environment`/`Spec` into an `Environment`/`Space` following the lyNNX API.

NOTE: These "wrappers" are unrelated to the Wrapper type in wrappers.py, 
    which take in a lyNNX Environment, rather than an environment from an external library.
"""

from typing import Any, Generic, Callable
from typing_extensions import TypeVar
from abc import ABC, abstractmethod

import numpy as np

import jax
from jax.typing import ArrayLike
import chex

import jax.numpy as jnp

from jumanji.env import Environment as JumanjiEnv
from jumanji import specs

from lynnx.envs.base import Environment, Space

def space_from_jumanji_spec(spec: specs.Spec) -> Space:
    if isinstance(spec, specs.BoundedArray):
        return Space(
            low=np.asarray(jnp.broadcast_to(spec.minimum, spec.shape).astype(spec.dtype)), 
            high=np.asarray(jnp.broadcast_to(spec.maximum, spec.shape).astype(spec.dtype))
        )

    elif isinstance(spec, specs.Array):
        if jnp.issubdtype(spec.dtype, jnp.integer):
            iinfo = jnp.iinfo(spec.dtype)
            min_val = np.asarray(iinfo.min)
            max_val = np.asarray(iinfo.max)
        else:
            min_val = -np.inf
            max_val = +np.inf

        return Space(
            low=np.full(spec.shape, min_val, dtype=spec.dtype), 
            high=np.full(spec.shape, max_val, dtype=spec.dtype)
        )

    else: # nested spec
        sub_spaces = { f"{key}": space_from_jumanji_spec(value)
            for key, value in vars(spec).items() if isinstance(value, specs.Spec) }

        low = spec._constructor(**{ key: space.low for key, space in sub_spaces.items() })
        high = spec._constructor(**{ key: space.high for key, space in sub_spaces.items() })

        return Space(low=low, high=high)

TEnvState = TypeVar("TEnvState")
TActionSpec = TypeVar("TActionSpec", bound=specs.Array)
TEnvObs = TypeVar("TEnvObs")

class JumanjiWrapper(Environment[TEnvState, TEnvObs, jax.Array, Any], Generic[TEnvState, TActionSpec, TEnvObs]):
    """Wrapper for Jumanji environments."""

    def __init__(self, 
        jumanji_env: JumanjiEnv[TEnvState, TActionSpec, TEnvObs], 
        get_obs_func: Callable[[TEnvState], TEnvObs] | None = None
    ) -> None:
        """
        `get_obs_func`: Jumanji environments do not have a standard public-facing method to get an observation from a state.
            However, many environments implement such a method internally, and we can guess the method name
            (eg. `_state_to_observation`, `_observation_from_state`). However, if this fails, the user should provide their own
            function to get an observation from a state.
        """

        self.jumanji_env = jumanji_env

        if get_obs_func is None:
            OBS_METHOD_POSSIBLE_NAMES = ('_state_to_observation', '_observation_from_state')

            for name in OBS_METHOD_POSSIBLE_NAMES:
                method = getattr(jumanji_env, name, None)

                if callable(method):
                    get_obs_func = method
                    break

        assert get_obs_func is not None, "No get_obs function provided, and cannot locate this function in environment."

        self._get_obs = get_obs_func

        self._observation_space = space_from_jumanji_spec(jumanji_env.observation_spec)
        self._action_space = space_from_jumanji_spec(jumanji_env.action_spec)

    def reset(self, key: chex.PRNGKey) -> tuple[TEnvState, dict[Any, Any]]:
        state, timestep = self.jumanji_env.reset(key)
        info = timestep.extras
        return state, info

    def step(self, key: chex.PRNGKey, state: TEnvState, action: jax.Array) \
        -> tuple[TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        state, timestep = self.jumanji_env.step(state, action)

        terminated = timestep.discount == 0
        truncated = jnp.logical_and(timestep.last(), jnp.logical_not(terminated))

        return state, timestep.reward, terminated, truncated, timestep.extras

    def get_obs(self, key: chex.PRNGKey, state: TEnvState) -> TEnvObs:
        return self._get_obs(state)

    def render(self, state: TEnvState, action: ArrayLike) -> Any:
        return self.jumanji_env.render(state)

    @property
    def observation_space(self) -> Space[TEnvObs]:
        return self._observation_space

    @property
    def action_space(self) -> Space[jax.Array]:
        return self._action_space

if __name__ == "__main__":
    ...