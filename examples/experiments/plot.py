import json, sys, os
import numpy as np
import matplotlib.pyplot as plt

EXPERIMENT_NAME = 'td3_mountaincarcontinuous_gymnax'
if len(sys.argv) > 1: EXPERIMENT_NAME = sys.argv[1]

EVAL_EPS = 256
if len(sys.argv) > 2: EVAL_EPS = sys.argv[2]

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), EXPERIMENT_NAME)

with open(os.path.join(DIR, 'evals.jsonl'), 'r') as f:
    lines = [ json.loads(line) for line in f if line.strip() ]

data = { key: np.asarray([ line[key] for line in lines ]) for key in lines[0].keys() }

plt.plot(data['steps'], data['return_mean'], label="mean")
plt.fill_between(data['steps'], data['return_mean'] - data['return_std'], data['return_mean'] + data['return_std'], 
    alpha=0.3, label=f"±1 std across {EVAL_EPS} eps")

plt.xlabel("Step")
plt.ylabel("Undiscounted Episode Return")
plt.title("Return Curve — Single Training Run")
plt.legend()

plt.savefig(os.path.join(DIR, 'return_curve.png'))

