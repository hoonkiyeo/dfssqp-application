# DF-SSQP Application

Application of derivative-free stochastic SQP to a constrained single-index
learning problem. A unit-norm direction compresses industry information; a neural
model is trained at each fixed direction to evaluate its validation loss.

The implementation is being rebuilt in small, readable parts. The folder structure
and settings are ready; data preparation, training and optimization are not yet
implemented in this repository.

## Where to look

| Location | Purpose |
| --- | --- |
| `config.py` | Data periods, algorithm parameters, training settings and seeds |
| `data/` | Data preparation; local inputs in `raw/`, prepared arrays in `processed/` |
| `models/` | Single-index model, training loss evaluation and fixed-direction baselines |
| `algorithms/` | Optimization methods, with each complete iteration visible in its file |
| `experiments/` | Run experiments, evaluate saved directions and plot results |
| `tests/` | Checks for implemented calculations and reproducibility |
| `docs/` | Formulation, user-managed notes and [references](docs/references.md) |
| `runs/` | Local experiment settings, results and checkpoints |

## Settings

Edit `config.py`. It starts from the agreed baseline with `iota1 = 1.0`.
For a different step coefficient, change that value in the same file.
The runner will save a copy of the settings with each experiment's results.

## Environment

Use Python 3.11 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

No experiment command is available yet. Data, generated outputs and local
environments are excluded from Git; data preparation code and documents are tracked.
