"""GPU-accelerated FlappyBird environment, with JAX image observation rendering."""

import numpy as np

from chex import dataclass
from jax.typing import ArrayLike
from jax import Array
from typing import Any
import chex

from lynnx.envs.base import Environment, Space

import jax.numpy as jnp
import jax
import dataclasses

@dataclass(frozen=True)
class State:
    """State for the FlappyBird environment.

    Coordinates (y, x) are relative to the center of the screen.,
        with positive y being the downward direction.
    """

    bird_pos_y: ArrayLike
    bird_vel_y: ArrayLike

    pipe1_pos_x: ArrayLike
    pipe1_pos_y: ArrayLike

    pipe2_pos_x: ArrayLike
    pipe2_pos_y: ArrayLike

@dataclass(frozen=True)
class Settings:
    """Settings for the FlappyBird environment.

    Positive y refers to the downward direction.
    """

    window_size: tuple[int, int] = (800, 600) # height, width

    bird_size: int = 40

    pipe_width: int = 120
    pipe_gap_height: int = 250
    pipe_min_height: int = 50

    #dist_between_pipe_centers: int = 400

    bird_vel_x: int = 150 # per second
    bird_accel_y: int = 1500
    bird_flap_vel_y: int = -600

    bird_pos_x = 150

@dataclass(frozen=True)
class Rewards:
    """Reward values for the FlappyBird environment."""
    pass_pipe: int = 1
    death: int = -1

@dataclass(frozen=True)
class RenderSettings:
    """Settings for the FlappyBird environment's image observation rendering."""
    background_color = jnp.array((139, 212, 245), dtype=jnp.uint8)
    pipe_color = jnp.array((114, 197, 100), dtype=jnp.uint8)
    bird_color = jnp.array((253, 218, 66), dtype=jnp.uint8)

class FlappyBirdEnv(Environment[State, Array, ArrayLike, Array]):
    """GPU-accelerated FlappyBird environment, with JAX image observation rendering.

    A recreation of the popular Flappy Bird mobile game:
        The agent controls a "bird", which on its own, falls down rapidly to the ground.
        Every step, the agent chooses whether or not to "flap" and move upwards.
        Vertical "pipes" appear on the screen, moving from the right side to the left side.
            Pipes appear in sets of upper and lower portions, with a gap in the middle.
        The agent fails if it hits the pipes or the upper and lower bounds of the screen.
            The agent must maneuver the bird through the gaps in the pipes.
    
    Coordinates (y, x) are relative to the center of the screen.,
        with positive y being the downward direction.

    State-based observations (default) consist of an array of 3 floats: 
        The bird's vertical velocity, 
        Vertical distance from the bird to the center of the next pipe gap.
        Horizontal distance from the bird to the center of the next pipe gap.

    Image-based observations can also be enabled, for a (800, 600, 3) array observation.
        Color channels will be normalized to [0, 1].
        NOTE: Image-based observations is experimental.

    Actions: a single integer [0, 1] indicating whether to flap.

    Rewards: 
        a reward (default 1) for passing a pipe
        a reward (default -1) for colliding with a pipe or the upper and lower bounds of the screen
    """

    def __init__(self, dt=0.1, settings=Settings(), rewards=Rewards(), render_settings=RenderSettings(),
            use_image_obs: bool = False) -> None:
        self.DT = dt
        self.settings = settings
        self.rewards = rewards
        self.render_settings = render_settings

        self.use_image_obs = use_image_obs

    def reset(self, key: chex.PRNGKey) -> tuple[State, dict[Any, Any]]:
        pipe1_key, pipe2_key = jax.random.split(key, 2)

        return State(
            bird_pos_y = self.settings.window_size[0] / 2,
            bird_vel_y = jnp.array(0, dtype=jnp.float32),

            pipe1_pos_x = self.settings.bird_pos_x - self.settings.pipe_width/2 - self.settings.bird_size,
            pipe1_pos_y = self._gen_pipe_y(pipe1_key),

            pipe2_pos_x = self.settings.window_size[1]/2 + self.settings.bird_pos_x  - self.settings.bird_size,
            pipe2_pos_y = self._gen_pipe_y(pipe2_key)
        ), {}

    def step(self, key: chex.PRNGKey, state: State, action: ArrayLike) \
        -> tuple[State, Array, Array, Array, dict[Any, Any]]:

        # update velocities & positions
        new_bird_vel_y = jnp.where(action != 0, 
            self.settings.bird_flap_vel_y,
            state.bird_vel_y + self.settings.bird_accel_y*self.DT
        )

        prev_pipe1_pos_x = state.pipe1_pos_x

        state = dataclasses.replace(state,
            bird_pos_y = state.bird_pos_y + new_bird_vel_y*self.DT,
            bird_vel_y = new_bird_vel_y,

            pipe1_pos_x = state.pipe1_pos_x - self.settings.bird_vel_x*self.DT,
            pipe2_pos_x = state.pipe2_pos_x - self.settings.bird_vel_x*self.DT,
        )

        # remove pipe and spawn in new pipe if left pipe (1) left screen
        pipe1_left_screen = state.pipe1_pos_x + self.settings.pipe_width/2 < 0

        state = dataclasses.replace(state,
            pipe1_pos_x = jnp.where(pipe1_left_screen, state.pipe2_pos_x, state.pipe1_pos_x),
            pipe1_pos_y = jnp.where(pipe1_left_screen, state.pipe2_pos_y, state.pipe1_pos_y),

            pipe2_pos_x = jnp.where(pipe1_left_screen, self.settings.window_size[1] + self.settings.pipe_width/2, state.pipe2_pos_x),
            pipe2_pos_y = jnp.where(pipe1_left_screen, self._gen_pipe_y(key), state.pipe2_pos_y),
        )

        reward = 0
        terminated = False

        # check for collisions
        hit_bot = state.bird_pos_y + self.settings.bird_size/2 > self.settings.window_size[0]
        hit_top = state.bird_pos_y - self.settings.bird_size/2 < 0
        
        x_in_pipe = jnp.logical_and(
            self.settings.bird_pos_x + self.settings.bird_size/2 >= state.pipe1_pos_x - self.settings.pipe_width/2,
            self.settings.bird_pos_x - self.settings.bird_size/2 <= state.pipe1_pos_x + self.settings.pipe_width/2
        )

        y_in_pipe_gap = jnp.logical_and(
            state.bird_pos_y + self.settings.bird_size/2 <= state.pipe1_pos_y + self.settings.pipe_gap_height/2,
            state.bird_pos_y - self.settings.bird_size/2 >= state.pipe1_pos_y - self.settings.pipe_gap_height/2
        )

        hit_pipe = jnp.logical_and(x_in_pipe, jnp.logical_not(y_in_pipe_gap))

        hit = jnp.logical_or(jnp.logical_or(hit_bot, hit_top), hit_pipe)

        reward += jnp.where(hit, self.rewards.death, 0)
        terminated = hit

        # check if passed pipe
        passed_pipe = jnp.logical_and(prev_pipe1_pos_x >= self.settings.bird_pos_x, state.pipe1_pos_x < self.settings.bird_pos_x)
        reward += jnp.where(passed_pipe, self.rewards.pass_pipe, 0)

        return state, reward, terminated, False, {}
    
    def _gen_pipe_y(self, key):
        height_to_center = self.settings.pipe_min_height + self.settings.pipe_gap_height/2

        return jax.random.uniform(key, shape=(), 
            minval=height_to_center, 
            maxval=self.settings.window_size[0] - height_to_center + 1
        )

    def get_obs(self, key: chex.PRNGKey, state: State) -> Array:
        if self.use_image_obs:
            return self.render(state, 0)
        
        return self.get_state_obs(state)

    def get_state_obs(self, state: State) -> Array:
        use_pipe2 = state.pipe1_pos_x + self.settings.pipe_width/2 + self.settings.bird_size < self.settings.bird_pos_x

        pipe_pos_x = jnp.where(use_pipe2, state.pipe2_pos_x, state.pipe1_pos_x)
        pipe_pos_y = jnp.where(use_pipe2, state.pipe2_pos_y, state.pipe1_pos_y)

        pipe_dx = pipe_pos_x - self.settings.bird_pos_x
        pipe_dy = pipe_pos_y - state.bird_pos_y

        return jnp.array((state.bird_vel_y, pipe_dy, pipe_dx))

    def render(self, state: State, action: ArrayLike) -> Array:
        image = jnp.full((*self.settings.window_size, 3), self.render_settings.background_color, dtype=jnp.uint8)

        # pipes
        pipes_pos = ((state.pipe1_pos_y, state.pipe1_pos_x), (state.pipe2_pos_y, state.pipe2_pos_x))

        for pipe_y, pipe_x in pipes_pos:
            y_vals, x_vals = jnp.mgrid[0:self.settings.window_size[0], 0:self.settings.window_size[1]]

            x_in_pipe = jnp.logical_and(
                x_vals <= pipe_x + self.settings.pipe_width/2,
                x_vals >= pipe_x - self.settings.pipe_width/2
            )

            y_in_pipe_gap = jnp.logical_and(
                y_vals <= pipe_y + self.settings.pipe_gap_height/2,
                y_vals >= pipe_y - self.settings.pipe_gap_height/2
            )

            pipe_mask = jnp.logical_and(x_in_pipe, jnp.logical_not(y_in_pipe_gap))
            
            image = jnp.where(pipe_mask[:, :, None], self.render_settings.pipe_color[None, None, :], image)

        # bird
        rect = jnp.full((self.settings.bird_size, self.settings.bird_size, 3), 
            self.render_settings.bird_color, dtype=jnp.uint8)

        bird_top = jnp.rint(state.bird_pos_y - self.settings.bird_size/2).astype(int)
        bird_left = jnp.rint(self.settings.bird_pos_x - self.settings.bird_size/2).astype(int)

        image = jax.lax.dynamic_update_slice(image, rect, (bird_top, bird_left, 0))

        return image

    @property
    def observation_space(self) -> Space[Array]:
        """Observation space of the environment.
        
        IMPORTANT: for state-based observations, high and low bounds are only currently accurate 
            for the default settings.
        """
        if self.use_image_obs:
            shape = (*self.settings.window_size, 3)

            Space(
                low=np.full(shape, 0.0, dtype=np.float32), 
                high=np.full(shape, 1.0, dtype=np.float32), 
            )

        return Space(
            low=np.array((-600.0, -625.0, -100), dtype=np.float32), 
            high=np.array((1500.0,  625.0, 255), dtype=np.float32)
        )

    @property
    def action_space(self) -> Space[ArrayLike]:
        """Action space of the environment."""
        return Space(low=np.array(0, dtype=np.int32), high=np.array(1, dtype=np.int32))