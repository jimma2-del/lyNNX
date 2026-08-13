"""Mujoco Playground's `G1JoystickFlatTerrain`, trained with PPO with asymmetric actor-critic."""

import time
import os
import json

import mediapy

import jax
import jax.numpy as jnp

from flax import nnx
from optax import schedules
import orbax.checkpoint as ocp

from lynnx.algos import ppo

from lynnx.envs.wrappers import VmapWrapper, EpisodeStepCountWrapper, PrecomputedResetsPoolWrapper
from lynnx.envs.utils import evaluate_episodes, rollout_episode

from lynnx.utils.nnx_modules import MLP, RunningMeanVarNorm, ActionDistributionHead, Pipe
from lynnx.utils.batch_utils import flatten_batched_tree, get_tree_flattened_dim

from lynnx.envs.mujoco_playground import MuJoCoPlaygroundWrapper
from mujoco_playground import registry

DIR = os.path.dirname(os.path.abspath(__file__))

rngs = nnx.Rngs(0, env=1, actions=2, params=3, optimize_samples=4)

## ENVIRONMENTS

# Make env
ENV_NAME = "G1JoystickFlatTerrain"
MAX_STEPS = 1000

N_ENVS = 2048
CAMERA = 'track'

config = registry.get_default_config(ENV_NAME)

# backend: Mujoco Warp or MJX
config.impl = 'warp' # 'jax'

# config needed for 'warp' backend
config.naconmax = 8 * N_ENVS # maximum num of contacts allowed in ALL parallel envs combined
    # should be adjusted depending on n_envs
config.njmax = 120 # maximum num PER env; default value of 29*2 + 8*4 is not enough

# Simulation frame rates
# config.ctrl_dt = 0.02
# config.sim_dt = 0.002

mjx_env = registry.load(ENV_NAME, config)
env = MuJoCoPlaygroundWrapper(mjx_env, {'camera': CAMERA})

# Cache reset states to avoid recomputing resets (expensive)
RESETS_POOL_SIZE = N_ENVS*4 - 1 # NOTE: cannot be the same as config.naconmax due to a bug
resets_pool_states_infos = jax.vmap(env.reset)(jax.random.split(rngs.env(), RESETS_POOL_SIZE))
precomputed_resets = lambda env: PrecomputedResetsPoolWrapper(env, resets_pool_states_infos)

# truncate episodes after length
steps_limited = lambda env: EpisodeStepCountWrapper(env, max_eps_len=MAX_STEPS)

train_env = steps_limited(precomputed_resets(VmapWrapper(env))) # Apply vmap BEFORE precomputed resets wrapper
eval_env = train_env

vis_env = steps_limited(precomputed_resets(env))


## HYPERPARMETERS

STEPS = 200_000_000 # total training steps

hyperparameters = ppo.Hyperparameters(
    discount_rate = 0.97,

    learning_rate = schedules.cosine_decay_schedule(2e-4, STEPS),
    n_envs = N_ENVS,
    gae_lambda = 0.95,

    rollout_length = 32,
    n_minibatches = 32, 
    n_epochs = 8,

    clip_epsilon = 0.2,

    vf_coef = 0.5, 
    ent_coef = 0.005,

    normalize_advantages = True,

    recompute_advantages = True,
    target_kl = 0.02,

    truncated_frac = 1.1 / MAX_STEPS
)


## CUSTOM NETWORKS: Asymmetric Actor-Critic
    # policy net only gets sensor data (partial observation); critic gets full "privileged" state

obs_trunk = RunningMeanVarNorm(env.observation_space.shapes_dtypes) # no flatten

policy_obs_sdt = env.observation_space.shapes_dtypes['state']
policy_head = ActionDistributionHead(env.action_space)
policy_head = Pipe(
    lambda x: x['state'], # take partial observation (sensor data)
    lambda x: flatten_batched_tree(policy_obs_sdt, x),
    MLP(rngs, (get_tree_flattened_dim(policy_obs_sdt), 512, 256, 128, policy_head.input_dim), 
        activation_func=nnx.swish), 
    policy_head
)

value_func_obs_sdt = env.observation_space.shapes_dtypes['privileged_state']
value_head = Pipe(
    lambda x: x['privileged_state'], # take full "privileged" state
    lambda x: flatten_batched_tree(value_func_obs_sdt, x),
    MLP(rngs, (get_tree_flattened_dim(value_func_obs_sdt), 512, 256, 128, 1), 
        activation_func=nnx.swish),
    lambda x: jnp.squeeze(x, axis=-1)
)

networks = ppo.Networks(obs_trunk=obs_trunk, policy_head=policy_head, value_head=value_head)


## TRAINING

EVAL_EPS = 256
EVAL_INTERVAL = 10_000_000
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

algo = ppo.PPO(train_env, hyperparameters)
train = nnx.jit(algo.train, static_argnames=('steps',), donate_argnames=('training_state'))

@nnx.jit
def evaluate(rngs, actor):
    return evaluate_episodes(
        rngs, eval_env, actor,
        EVAL_EPS, EVAL_EPS
    )

training_state = algo.init_training_state(rngs, networks=networks)

# If episodes have a fixed length, it may be helpful to stagger states to avoid synced episode phases
    # NOT necessary for G1JoystickFlatTerrain because episodes are reset early if robot falls over

# jitted_stagger = nnx.jit(stagger_env_states, 
#     static_argnames=('env', 'n_envs', 'stagger_step_size'), donate_argnames=('initial_env_states'))
# training_state.env_states = jitted_stagger(rngs, AutoResetWrapper(train_env), N_ENVS, 
#     stagger_step_size=1, initial_env_states=training_state.env_states)

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

# IMPORTANT for loading: env states are set to None when saving
    # env states cannot be saved for Mujoco Playground due to zero size arrays (orbax limitation)
training_state.env_states = None

# Make temporary directory to store checkpoints
os.makedirs(os.path.join(DIR, '_tmp'), exist_ok=True)

SAVE_PATH = os.path.join(DIR, '_tmp', f'training_state_{training_state.steps}_steps')

state = nnx.state(training_state)
checkpointer_save = ocp.StandardCheckpointer()
checkpointer_save.save(SAVE_PATH, state)

## VISUALIZATION

# Rollout trained actor
actor = algo.make_actor(training_state.networks, deterministic_sampling=True)

_impl_removed = lambda env_state: env_state.replace(state=env_state.state.tree_replace({ 'data._impl': None }))

rngs = nnx.Rngs(0, env=10, actions=20)
timesteps, final_timestep = rollout_episode(rngs, vis_env, actor,
    take_func = lambda ts: ts.replace(state=_impl_removed(ts.state)))
        # remove unnecessary warp `_impl` property, which takes up a lot of memory

eps_steps = len(timesteps.reward)
eps_return = sum(timesteps.reward)
print(f"{'Truncated' if timesteps.truncated[-1] else 'Terminated'} at steps={eps_steps}, return={eps_return}.")

# Make video animation
states = timesteps.state.state # unwrap EpisodeStepCountWrapper state

# Transpose the PyTree of state arrays into a list of individual states
states = jax.device_get(states)
states = [ jax.tree.map(lambda x: x[i], states) for i in range(0, eps_steps) ]

frames = mjx_env.render(states, camera=CAMERA)
mediapy.write_video(os.path.join(DIR, f'visualization_{training_state.steps}_steps.mp4'), 
    frames, fps=1.0/mjx_env.dt)