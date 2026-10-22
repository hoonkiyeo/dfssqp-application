# DF-SSQP Application

Application of derivative-free stochastic SQP to a constrained single-index
learning problem. A unit-norm direction compresses industry information; a neural
model is trained at each fixed direction to evaluate its validation loss.

Data preparation and the fixed-direction training objective are implemented and
verified against the original calculations. Outer optimization will be added next.

## Where to look

| Location | Purpose |
| --- | --- |
| `config.py` | Data periods, algorithm parameters, training settings and seeds |
| `data/` | Data preparation; source inputs in `raw/`, prepared arrays in `processed/` |
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

## Prepare the data

The repository includes the two frozen source ZIPs under `data/raw/` and the
prepared monthly arrays under `data/processed/`. The source ZIP hashes in `config.py`
pin the data vintage. To rebuild the monthly arrays and run the data checks:

```bash
python -m data.prepare
python -m pytest tests/test_prepare.py -q
```

`data/prepare.py` shows the calculation in order:

1. Read daily industry returns and convert percentages to decimals. The separate
   factor file checks dates only; its market returns are not used.
2. Fit inverse-volatility benchmark weights using training days only.
3. Form daily benchmark returns, then sum squared daily returns within each month.
4. Take logs and pair the current month with the next month, within each split.

| Array | Meaning | Training | Validation |
| --- | --- | --- | --- |
| `z` | Current-month log realized variances of the 48 industries | 410 × 48 | 136 × 48 |
| `u` | Current-month benchmark log realized variance | 410 | 136 |
| `y` | Next-month benchmark log realized variance | 410 | 136 |

The command saves `training.npz`, `validation.npz` and `metadata.json` under
`data/processed/`. Each array file also contains month labels and industry names;
the metadata records source hashes and benchmark weights. These arrays have not
been standardized. The reserved test targets are not constructed.

## Model and loss evaluation

`models/single_index.py` defines beta*u + F_theta(index), with a 1-8-1 Softplus link.
`models/objective.py` shows the full fit: compute the index, standardize from training
only, initialize a fresh model, take 1,000 Adam steps, and compute validation MSE in
original response units. Each seed fixes separate initialization and minibatch streams.

Run one fit at the configured initial direction and first evaluation seed:

```bash
python -m models.objective
python -m pytest tests -q
```

For repeated fits, load the prepared `z`, `u`, `y` arrays once into dictionaries and
call `evaluate(w, seed, training, validation)`. It returns a scalar loss. Set
`details=True` to also inspect predictions, initialization, minibatches, fitted
parameters and scaling statistics. Parameters are in standardized coordinates.
Set `torch.set_num_threads(config.torch_threads)` once in the calling script.

No outer optimization experiment command is available yet. Raw ZIPs and prepared
monthly data are tracked in Git. Experiment outputs under `runs/` and local
environments remain excluded.
