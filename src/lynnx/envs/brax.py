"""Utility for converting a Brax `PipelineEnv` into an `Environment`/`Space` following the lyNNX API.

NOTE: These "wrappers" are unrelated to the Wrapper type in wrappers.py, 
    which take in a lyNNX Environment, rather than an environment from an external library.
"""

from typing import Any

import numpy as np

import jax
from jax.typing import ArrayLike
import chex

import jax.numpy as jnp

from brax.envs.base import PipelineEnv, State as BraxState
from brax.io import image

from lynnx.envs.base import Environment, Space

class BraxWrapper(Environment[BraxState, jax.Array, jax.Array, np.ndarray]):
    """Wrapper for Brax environments."""

    @staticmethod
    def combine_metrics_info(metrics: dict[Any, Any], info: dict[Any, Any]) -> dict[Any, Any]:
        comb = info.copy()
       
        for key, val in metrics.items():
            comb['metrics/' + str(key)] = val

        return comb

    def __init__(self, brax_env: PipelineEnv, render_settings: dict[Any, Any] = {}) -> None:
        """
        `brax_env`: Brax environment to convert.
            NOTE: This environment may not work properly if `brax_env` is already batched.
                Try passing `batch_size=None` to `brax.envs.create()`, and vmapping manually with `VmapWrapper()`.
            NOTE: Duplicate resets may be performed when using algorithms/utils if `brax_env` already auto-resets.
                To avoid this, pass `auto_reset=False` to `brax.envs.create()`.
        """

        self.brax_env = brax_env
        self.render_settings = render_settings

        obs = np.inf * np.ones(brax_env.observation_size, dtype=np.float32)
        self._observation_space = Space(-obs, obs)

        action = np.asarray(brax_env.sys.actuator.ctrl_range)
        self._action_space = Space(action[:, 0], action[:, 1])

    def reset(self, key: chex.PRNGKey) -> tuple[BraxState, dict[Any, Any]]:
        state = self.brax_env.reset(key)
        return state, self.combine_metrics_info(state.metrics, state.info)

    def step(self, key: chex.PRNGKey, state: BraxState, action: jax.Array) \
        -> tuple[BraxState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        state = self.brax_env.step(state, action)

        truncated = False
        if 'truncation' in state.info:
            truncated = state.info['truncation']

        terminated = jnp.logical_and(state.done, jnp.logical_not(truncated))

        return state, state.reward, terminated, truncated, self.combine_metrics_info(state.metrics, state.info)

    def get_obs(self, key: chex.PRNGKey, state: BraxState) -> jax.Array:
        return state.obs

    def render(self, state: BraxState, action: ArrayLike) -> np.ndarray:
        return image.render_array(self.brax_env.sys, state.pipeline_state, **self.render_settings)

    @property
    def observation_space(self) -> Space[jax.Array]:
        return self._observation_space

    @property
    def action_space(self) -> Space[jax.Array]:
        return self._action_space