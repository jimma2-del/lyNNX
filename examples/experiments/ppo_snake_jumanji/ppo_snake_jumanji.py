"""Jumanji's `Snake-v1`, trained with PPO + reward shaping + CNN/action-masking networks + RMSprop.
Adapted from the "Further Customization" section of the tutorial notebook (examples/tutorial/tutorial.ipynb).
"""
import time
import os
import json

import jax
import jax.numpy as jnp

from flax import nnx
from optax import schedules
import optax
import chex
import orbax.checkpoint as ocp

from lynnx.algos import ppo

from lynnx.envs.base import Environment, Space
from lynnx.envs.wrappers import Wrapper, EpisodeStepCountWrapper, VmapWrapper
from lynnx.utils.nnx_modules import ActionDistributionHead
from lynnx.envs.utils import evaluate_episodes, rollout_episode

from lynnx.envs.jumanji import JumanjiWrapper
import jumanji
from jumanji.environments.routing.snake import Observation as SnakeObs, State as SnakeState

DIR = os.path.dirname(os.path.abspath(__file__))

## ENVIRONMENTS

# Make env
jumanji_env = jumanji.make('Snake-v1')
env = JumanjiWrapper(jumanji_env)

# Custom wrapper
class CustomSnakeWrapper(Wrapper[SnakeState, SnakeObs, jax.Array, jax.Array]):
    def __init__(self,
        env: Environment[SnakeState, SnakeObs, jax.Array, jax.Array],
        step_penalty: float = -0.01,
        death_penalty: float = -1.0,
    ) -> None:
        super().__init__(env)

        self.step_penalty = step_penalty
        self.death_penalty = death_penalty

    # Override the step method to add custom functionality
    def step(self, key: chex.PRNGKey, state: SnakeState, action: jax.Array) \
            -> tuple[SnakeState, jax.Array, jax.Array, jax.Array, dict]:
        state, reward, terminated, truncated, info = super().step(key, state, action)

        # apply custom penalties
        reward += self.step_penalty
        reward += terminated * self.death_penalty

        return state, reward, terminated, truncated, info

    # Other methods will keep default behavior

train_env = CustomSnakeWrapper(env,
    step_penalty = -0.01,
    death_penalty = -1.0,
)

eval_env = env # keep original env for evaluation

# Truncation wrappers
MAX_EPS_LEN = 3000
train_env = EpisodeStepCountWrapper(train_env, max_eps_len=MAX_EPS_LEN, terminate=False)
eval_env = EpisodeStepCountWrapper(eval_env, max_eps_len=MAX_EPS_LEN, terminate=False)


## HYPERPARMETERS

STEPS = 10_000_000 # total training steps

hyperparameters = ppo.Hyperparameters(
    n_envs = 256,

    learning_rate = schedules.linear_schedule(3e-4, 1e-4, STEPS),
    max_grad_norm = 0.5,
    optimizer_params = { 'decay': 0.9 },

    discount_rate = 0.99,
    gae_lambda = 0.95,

    rollout_length = 32,
    n_minibatches = 32,
    n_epochs = 8,

    clip_epsilon = 0.2,

    vf_coef = 0.5,
    ent_coef = 0.01,

    normalize_advantages = True,

    recompute_advantages = True,
    target_kl = 0.02,

    truncated_frac = 1/1000,
)

rngs = nnx.Rngs(0, env=1, actions=2, params=3, optimize_samples=4)


## CUSTOM OPTIMIZER: RMSprop

# define optimizer factory function, wrapping with optax.inject_hyperparams
@optax.inject_hyperparams
def make_custom_optimizer(learning_rate, max_grad_norm, decay=0.9):
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.rmsprop(learning_rate, decay)
    )

# get starting values (steps=0) for all optimizer parameters
algo = ppo.PPO(VmapWrapper(train_env), hyperparameters)
optimizer_params = algo.resolve_optimizer_params(steps=0)
    # convenience function to get a dictionary of values for all optimizer parameters at a particular step
    # includes learning_rate and max_grad_norm, taking them from the root hyperparameters dataclass

# make the optimizer, initialized with the starting values
custom_optimizer = make_custom_optimizer(**optimizer_params)


## CUSTOM NETWORKS: shared CNN and invalid action masking

class SnakeCNN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.conv1 = nnx.Conv(5, 32, kernel_size=(3, 3), rngs=rngs) # (12, 12, 5) -> (12, 12, 32)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), strides=2, rngs=rngs) # (12, 12, 32) -> (6, 6, 64)
        self.linear = nnx.Linear(6 * 6 * 64, 256, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(256, rngs=rngs)

    def __call__(self, x: SnakeObs, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, jax.Array]:
        features = x.grid # 'grid' contains the actual observation features

        features = nnx.relu(self.conv1(features))
        features = nnx.relu(self.conv2(features))
        features = features.reshape(*features.shape[:-3], -1)  # flatten
        features = nnx.relu(self.layer_norm(self.linear(features)))

        return features, x.action_mask # pass the action mask down as well for use in the policy head

obs_trunk = SnakeCNN(rngs)

class SnakePolicyHead(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.action_distribution_head = ActionDistributionHead(env.action_space)
            # this class unflattens actions into a distribution,
                # and contains learnable log std params for continuous actions
            # not really necessary for a simple discrete action space; included for illustrative purposes

        self.linear = nnx.Linear(256, self.action_distribution_head.input_dim, rngs=rngs)

    def __call__(self, x: tuple[jax.Array, jax.Array], rngs: nnx.Rngs | None = None) -> jax.Array:
        features, action_mask = x

        features = self.linear(features)
        action_dist = self.action_distribution_head(features) # unflatten into an action distribution
            # does nothing here since unflattening is not necessary; included for illustrative purposes

        # apply invalid action masking, setting invalid action logits to -jnp.inf
        action_dist = jnp.where(action_mask, action_dist, -jnp.inf)

        # in Jumanji Snake-v1, there can be no valid actions for certain states,
            # which leads to NaNs as ALL logits are masked to -jnp.inf
        no_valid_actions = jnp.all(jnp.logical_not(action_mask), axis=-1, keepdims=True)
        action_dist = jnp.where(no_valid_actions, 0, action_dist)
            # lets replace the distribution with a uniform distribution in these cases

        return action_dist

policy_head = SnakePolicyHead(rngs)

value_head = nnx.Sequential(
    lambda x: x[0], # discard x[1] -> the action mask
    ppo.Networks.make_default_value_head(rngs, 256, hidden_dims=(256,))
)

custom_networks = ppo.Networks(obs_trunk, policy_head, value_head)


## TRAINING

EVAL_EPS = 256
EVAL_INTERVAL = 1_000_000
N_LOGS_PER_EVAL = 3

METRICS_PATH = os.path.join(DIR, 'metrics.jsonl')
EVALS_PATH = os.path.join(DIR, 'evals.jsonl')

def append_jsonl(path: str, data: dict):
    # transpose data
    data = jax.device_get(data)
    transposed = [ { key: val[i].item() for key, val in data.items() } 
        for i in range(len(data['steps'])) ]

    # convert jsonl
    lines = [ json.dumps(item) + '\n' for item in transposed ]

    with open(path, 'a') as f:
        f.writelines(lines)

algo = ppo.PPO(VmapWrapper(train_env), hyperparameters)
train = nnx.jit(algo.train, static_argnames=('steps',), donate_argnames=('training_state'))

@nnx.jit
def evaluate(rngs, actor):
    return evaluate_episodes(
        rngs, VmapWrapper(eval_env), actor,
        EVAL_EPS, EVAL_EPS
    )

training_state = algo.init_training_state(rngs, networks=custom_networks, optax_optimizer=custom_optimizer)

while training_state.steps < STEPS:
    print()

    start_time = time.perf_counter()
    training_state, metrics = train(rngs, training_state, EVAL_INTERVAL)
    elasped_time = time.perf_counter() - start_time

    # NOTE: Elapsed time will be significantly higher during the first two iterations due to JIT compile time.
        # Steps/sec will greatly increase for the remaining iterations.

    # Print metrics
    avg_metrics = jax.tree.map(lambda x: list(map(jnp.mean, jnp.array_split(x, N_LOGS_PER_EVAL))), metrics)
    steps = avg_metrics.pop('steps')
    for i in range(N_LOGS_PER_EVAL):
        print(f"Step {steps[i]:.0f}: " + " ".join([ f"{key}={val[i]:.5g}" for key, val in avg_metrics.items() ]))

    print()

    append_jsonl(METRICS_PATH, { 'steps': steps, **avg_metrics }) # Save metrics

    sps = EVAL_INTERVAL / elasped_time
    print(f"COMPLETED steps={training_state.steps}; sps={sps:,.1f}")

    # Evaluate
    actor = algo.make_actor(training_state.networks, deterministic_sampling=True)
        # deterministic_sampling=True -> use action distribution means; no randomness
    returns, lengths = evaluate(rngs, actor)

    eval_metrics = {
        'return_mean': jnp.mean(returns), 'return_std': jnp.std(returns, ddof=1),
        'length_mean': jnp.mean(lengths), 'length_std': jnp.std(lengths, ddof=1)
    }

    print(f"Episode Return: mean={eval_metrics['return_mean']} std={eval_metrics['return_std']}")
    print(f"Episode Length: mean={eval_metrics['length_mean']} std={eval_metrics['length_std']}")

    eval_log_metrics = { k: (v,) for k, v in { 'steps': training_state.steps, **eval_metrics }.items() }
    append_jsonl(EVALS_PATH, eval_log_metrics) # Save eval


## SAVE TRAINING STATE

# Make temporary directory to store checkpoints
os.makedirs(os.path.join(DIR, '_tmp'), exist_ok=True)

SAVE_PATH = os.path.join(DIR, '_tmp', f'training_state_{training_state.steps}_steps')

state = nnx.state(training_state)
checkpointer_save = ocp.StandardCheckpointer()
checkpointer_save.save(SAVE_PATH, state)

## VISUALIZATION

# Rollout trained actor
actor = algo.make_actor(training_state.networks, deterministic_sampling=True)

rngs = nnx.Rngs(0, env=100, actions=200)
timesteps, final_timestep = rollout_episode(rngs, eval_env, actor)

eps_steps = len(timesteps.reward)
eps_return = sum(timesteps.reward)
print(f"{'Truncated' if timesteps.truncated[-1] else 'Terminated'} at steps={eps_steps}, return={eps_return}.")

# Make gif animation
VISUALIZE_FPS = 10
VISUALIZE_FRAME_SKIP = 1
    # skip frames to speed up rendering time

states = timesteps.state.state # need to unwrap state due to EpisodeStepCountWrapper

# Transpose the PyTree of state arrays into a list of individual states
states = jax.device_get(states)
states = [ jax.tree.map(lambda x: x[i], states) for i in range(0, eps_steps, VISUALIZE_FRAME_SKIP) ]

print("Animating... -- this can take a few minutes")
delay_ms = 1/VISUALIZE_FPS * VISUALIZE_FRAME_SKIP
jumanji_env.animate(states, delay_ms, os.path.join(DIR, f'visualization_{training_state.steps}_steps.gif'))