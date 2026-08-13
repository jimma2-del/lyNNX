"""`gymnax` library's `MountainCarContinuous-v0`, trained with TD3 with heavy exploration noise."""

import time
import os
import json

import jax
import jax.numpy as jnp

from flax import nnx
from optax import schedules
import orbax.checkpoint as ocp

from lynnx.algos import td3

from lynnx.envs.wrappers import VmapWrapper
from lynnx.envs.utils import evaluate_episodes, rollout_episode

from lynnx.envs.gymnax import GymnaxWrapper
import gymnax
from gymnax.visualize import Visualizer

DIR = os.path.dirname(os.path.abspath(__file__))

## ENVIRONMENTS

# Make env
gymnax_env, gymnax_env_params = gymnax.make("MountainCarContinuous-v0")
env = GymnaxWrapper(gymnax_env, gymnax_env_params)

train_env = env
eval_env = env

## HYPERPARMETERS

STEPS = 10_000_000 # total training steps

hyperparameters = td3.Hyperparameters(
    learning_rate = 2.5e-4,
    n_envs = 256,

    policy_delay = 2,
    exploration_noise = schedules.linear_schedule(1, 0.1, 0.9*STEPS),
    target_noise = 0.2,
    target_noise_clip = 0.5,

    train_freq = 32,
    batch_size = 256,

    replay_buffer_size = 100_000,
    truncated_frac = 0.0,

    polyak_tau = 0.005,
)

rngs = nnx.Rngs(0, env=1, actions=2, params=3, optimize_samples=4)


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

algo = td3.TD3(VmapWrapper(train_env), hyperparameters)
train = nnx.jit(algo.train, static_argnames=('steps',), donate_argnames=('training_state'))

@nnx.jit
def evaluate(rngs, actor):
    return evaluate_episodes(
        rngs, VmapWrapper(eval_env), actor,
        EVAL_EPS, EVAL_EPS
    )

training_state = algo.init_training_state(rngs)

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
    actor = algo.make_actor(training_state.networks, noise=0)
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
actor = algo.make_actor(training_state.networks, noise=0)

rngs = nnx.Rngs(0, env=10, actions=20)
timesteps, final_timestep = rollout_episode(rngs, eval_env, actor)

eps_steps = len(timesteps.reward)
eps_return = sum(timesteps.reward)
print(f"{'Truncated' if timesteps.truncated[-1] else 'Terminated'} at steps={eps_steps}, return={eps_return}.")

# Make gif animation

# Transpose the PyTree of state arrays into a list of individual states
states = jax.device_get(timesteps.state)
states = [ jax.tree.map(lambda x: x[i], states) for i in range(0, eps_steps) ]

cum_rewards = jnp.cumsum(timesteps.reward)

vis = Visualizer(gymnax_env, gymnax_env_params, states, cum_rewards)
vis.animate(save_fname=os.path.join(DIR, f'visualization_{training_state.steps}_steps.gif'))