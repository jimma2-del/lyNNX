# Experiments
This directory contains complete sample scripts for training various RL algorithms on various lyNNX and external environments.

For example, run
```bash
python dqn_cartpole_gymnax/dqn_cartpole_gymnax.py
```

Experiments generate the following files:
- `evals.jsonl` &mdash; stores evaluation metrics
- `metrics.jsonl` &mdash; stores training metrics (eg. losses)
- `visualization_{STEPS}_steps`, file type varies &mdash; visualization of the final, trained agent
- `_tmp/training_state_{STEPS}_steps/` &mdash; folder containing a saved checkpoint after training

Generate a plot of the training run's episode return over time with 
```bash
python plot.py <EXPERIMENT_DIR_NAME> <EVAL_EPISODES>
```