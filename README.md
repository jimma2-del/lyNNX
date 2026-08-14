![Banner Demo Animations](https://raw.githubusercontent.com/jimma2-del/lyNNX/master/docs/assets/banner-anims.gif)
<small>*Agents **trained** using lyNNX. Environments from: [Gymnax](https://github.com/RobertTLange/gymnax), [Jumanji](https://github.com/instadeepai/jumanji), [Mujoco Playground](https://github.com/google-deepmind/mujoco_playground).*</small>

# lyNNX: Deep RL in Flax NNX
<a href="https://www.python.org/doc/versions/">
    <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python Versions">
</a>
<a href="https://opensource.org/license/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-purple.svg" alt="License" />
</a>
<a href="https://colab.research.google.com/github/jimma2-del/lyNNX/blob/master/examples/tutorial/tutorial.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"/>
</a>

<small>*lyNNX is still experimental. Expect breaking changes!*</small>

lyNNX is a repository of deep reinforcement learning (deep RL) algorithms and environments, written completely in JAX + Flax NNX for end-to-end GPU-accelerated training. 

**Train any environment, instantly.** Algorithms are designed to support, whenever possible, any observation space and any action space &mdash; whether discrete, continuous, or a mix of both. Wrappers allow plug-and-play use with popular existing JAX RL environment libraries: [Gymnax](https://github.com/RobertTLange/gymnax), [Jumanji](https://github.com/instadeepai/jumanji), [Brax](https://github.com/google/brax), [Mujoco Playground](https://github.com/google-deepmind/mujoco_playground), etc.

**Full customizability, without editing source code.** Pass your own hyperparameters, network architectures/user-defined classes, optimizers, etc. Customize environments with the many built-in wrappers or create your own. Turn any (non-structural) hyperparameter into a schedule based on training progress.

**Rollout, evaluate, or visualize your agent in a single function call.** Utility functions are included to simplify common use cases, allowing you to focus on what matters:
```py
timesteps, final_states, final_infos = rollout(rngs, VmapWrapper(env), actor, iters=32, env_batch_dims=256)
returns, lengths = evaluate_episodes(rngs, env, actor, episodes=256, n_envs=256, eps_steps_limit=1000)
visualize_pygame(rngs, env, actor)
```

**Modern, Pythonic API with Flax NNX.** Use JAX transformations directly on native Python objects, keeping the traditional mutability and reference semantics. This simplifies state management, and brings the interface more in line with more conventional PyTorch paradigms. [Why Flax NNX?](https://flax.readthedocs.io/en/latest/why.html)

**JIT-compile the entire training loop.** Code is written 100% in JAX for massive parallelization on the GPU. All memory buffers are kept completely on device, eliminating host-device transfer bottlenecks. Run experiments at 100k+ steps per second, allowing for rapid iteration and troubleshooting.

## Features Implemented

### Algorithms include: 
- [Tabular Q-Learning](https://doi.org/10.1007/BF00992698), w/ rounding or linear interpolation for continuous observations
- [Deep Q-Learning (DQN)](https://arxiv.org/abs/1312.5602), w/ [Double DQN](https://arxiv.org/abs/1509.06461) add-on
- [Advantage Actor-Critic (A2C)](https://openai.com/index/openai-baselines-acktr-a2c/)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347), w/ action clipping or tanh-based squashing
- [Twin Delayed Deep Deterministic Policy Gradient (TD3)](https://arxiv.org/abs/1802.09477)
- [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290), w/ [automatic entropy adjustment](https://arxiv.org/abs/1812.05905)

### Environments include: 
- **Gridworld:** place walls and endpoints, set tile rewards; visualize Q table as arrows.
- **Flappy Bird:** includes a fully JIT-compilable renderer written in JAX, allowing for end-to-end GPU-accelerated training with image observations.
- **More to come!** WIP 2D Soccer environment (image observations, can be MARL).

## Getting Started

It is highly recommended to check out the [tutorial notebook](https://github.com/jimma2-del/lyNNX/tree/master/examples/tutorial/tutorial.ipynb) for a more complete guide. <a href="https://colab.research.google.com/github/jimma2-del/lyNNX/blob/master/examples/tutorial/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>

Additionally, check out [examples/experiements](https://github.com/jimma2-del/lyNNX/tree/master/examples/experiments) for sample scripts.

The rest of this section will act as an abridged summary.

### Installation
Prerequisites: Python 3.12 or later.

Install JAX. Installation details vary between accelerator devices; see <https://docs.jax.dev/en/latest/installation.html>. For CUDA 12: 
```bash
pip install -U jax[cuda12]
```

Install lyNNX from PyPI:
```bash
pip install lynnx
```

To use lyNNX with external environment libraries, install them using:
```bash
pip install lynnx[gymnax,jumanji,playground,viz]
```

### Basic Usage
Run a sample experiment script (see [examples/experiements](https://github.com/jimma2-del/lyNNX/tree/master/examples/experiments)):
```bash
python examples/experiments/dqn_cartpole_gymnax/dqn_cartpole_gymnax.py
```

Or, write your own script:
```py
from flax import nnx

from lynnx.algos import dqn
from lynnx.envs.wrappers import VmapWrapper
from lynnx.envs.utils import evaluate_episodes

from lynnx.envs.gymnax import GymnaxWrapper
import gymnax

gymnax_env, gymnax_env_params = gymnax.make("CartPole-v1")
env = GymnaxWrapper(gymnax_env, gymnax_env_params)

algo = dqn.DQN(VmapWrapper(env), hyperparameters=dqn.Hyperparameters())
train = nnx.jit(algo.train, static_argnames=('steps',), donate_argnames=('training_state'))

rngs = nnx.Rngs(0, env=1, actions=2, params=3, optimize_samples=4)
training_state = algo.init_training_state(rngs, prefill_steps=5000)
training_state, metrics = train(rngs, training_state, steps=100_000)

actor = algo.make_actor(training_state.networks, epsilon=0)
evaluate = nnx.jit(evaluate_episodes, static_argnames=('env', 'episodes', 'n_envs'))

returns, lengths = evaluate(rngs, VmapWrapper(env), actor, episodes=256, n_envs=256)
print(f"Episode Return: mean={jnp.mean(returns)} std={jnp.std(returns, ddof=1)}")
print(f"Episode Length: mean={jnp.mean(lengths)} std={jnp.std(lengths, ddof=1)}")
```

## Roadmap
### Short-term plans include:
- Add better logging utilities and support for Weights & Biases.
- Properly benchmark performance/speed and compare with other libraries.
- Refine the environment API to be more flexible. Specifically, we should support states that persist throughout the entire training process, allowing for curriculum learning and reducing compile times for reset states caching.
- Support multi-device training with the Anakin Podracer architecture.
- Add more DQN add-ons: n-step returns, dueling networks, NoisyNets
- Add support for discrete action spaces in SAC.
- Integrate existing 2D Soccer environment (image observations, can be MARL).

### Possible mid-to-long term features include:
- Prioritized experience replay (PER). This affects all off-policy algorihms: DQN, TD3, and SAC.
- QR-DQN, completing all of the Rainbow DQN enchancements.
- Support for CPU-based environments using the Sebulba Podracer architecture.
- More algorithms! Model-based algorithms: Dreamer-v3, MuZero.