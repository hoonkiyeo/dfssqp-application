# DF-SSQP Application

Application of derivative-free stochastic SQP to a constrained single-index
learning problem. A unit-norm direction compresses industry information; a neural
model is trained at each fixed direction to evaluate its validation loss.

Data preparation, the training objective and the AI/ZI DF-SSQP iteration are
verified against the original calculations. The experiment runner saves progress
and supports background execution and resumption.

## Where to look

| Location | Purpose |
| --- | --- |
| `config.py` | Data periods, algorithm parameters, training settings and seeds |
| `data/` | Data preparation; source inputs in `raw/`, prepared arrays in `processed/` |
| `models/` | Single-index model and fixed-direction training/loss evaluation |
| `algorithms/` | Optimization methods, with each complete iteration visible in its file |
| `experiments/` | Run AI/ZI experiments, save evaluations and plot saved results |
| `tests/` | Checks for implemented calculations and reproducibility |
| `docs/` | Formulation, user-managed notes and [references](docs/references.md) |
| `runs/` | Local experiment settings, results and checkpoints |

## Settings

Edit `config.py`. It starts from the agreed baseline with `iota1 = 1.0`.
For a different step coefficient, change that value in the same file.
Each new experiment saves its settings. Resuming uses that saved copy, so editing
`config.py` for another experiment does not alter an existing one.

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

## DF-SSQP iteration

`algorithms/df_ssqp.py` shows the complete first-order update in `step`: schedules,
perturbed losses, derivative estimates and averages, Jacobian safeguard, the SQP
system, merit/ratio parameters and the primal/dual update. `initialize()` creates
the configured normalized Gaussian start and initial state.

Using the training/validation dictionaries loaded above:

```python
from algorithms.df_ssqp import initialize, step
from models.objective import evaluate
import config


def loss(w, seed):
    return evaluate(w, seed, training, validation)


state = initialize()
state, trace = step(state, config.outer_seeds[0], "AI", loss)
# Call again with the returned state for the next iteration; use "ZI" for estimation.
```

The state contains `k`, `w`, `lam`, `g_bar`, `G_bar`, `tau`, and `nu`. The trace
contains the actual quantities computed at each stage, including clipping and
numerical-zero diagnostics. Each step uses two fresh loss evaluations with the same
training seed. Indexed sampling preserves the original perturbation/training seeds
and allows continuation directly from the saved state.

The iteration follows Na (2025), Algorithm 1, with the existing application choices:
unit-norm constraint, identity Hessian, AI/ZI Jacobians, singular-value clipping and
`psi=0`. A certified zero primal direction retains `nu` and completes the ordinary
state update. The numerical checks are at the bottom of the same file; they use
constraint/KKT roundoff bounds, without adding a tunable zero-step threshold.

## Run an experiment

Edit the parameters, `modes`, `outer_seeds`, `sqp_steps`, `evaluation_steps` and
`evaluation_seeds` in `config.py`, then run from the repository root:

```bash
python -m experiments.run --background
```

The command prints the new folder under `runs/`. Omit `--background` to run in the
terminal, or add `--output runs/my_experiment` to choose a new folder name.
Existing folders are never overwritten. AI/ZI runs execute sequentially;
background execution continues after the terminal closes.

Using the printed folder in place of `runs/my_experiment`:

```bash
python -m experiments.run --status runs/my_experiment
python -m experiments.run --pause runs/my_experiment
python -m experiments.run --resume runs/my_experiment --background
```

Pause finishes the current SQP iteration or independent evaluation fit before
stopping. Ctrl+C and SIGTERM do the same. After a forced shutdown, resume reuses
saved fits and starts an unfinished fit again with the same seed. Such interrupted
attempts are recorded separately because their unrecorded computation time is
unknown. A pause request left by a crash is honored once on restart; resume again
after the status becomes `paused`. `--status` also detects an inactive process.

For a short check, use a separate configuration with fewer steps/seeds. Set
`evaluation_steps` to an increasing list including zero and the final step.
`--stop-after 2` pauses after two new SQP steps in that invocation; it does not
change the experiment's saved budget. No power or sleep settings are changed.

### What is saved

| File | Contents |
| --- | --- |
| `settings.json` | Frozen configuration, code/data hashes, runtime versions and Git revision |
| `summary.csv` | Mean MSE and feasibility, with SE across available runs at each checkpoint; `runs` gives the actual count |
| `AI_<seed>/history.csv` (and `ZI_<seed>/`) | Feasibility, schedules, actual step, update norm, Jacobian singular value, clipping/zero-step diagnostics |
| Each run's `evaluations.csv` | Mean MSE over the reporting seeds at each saved direction; incomplete panels are marked |
| Each run's `history.jsonl` | Full state after every iteration, including w, multiplier and moving averages; this is the resume checkpoint |
| Each run's `fits.jsonl` and `costs.json` | Per-fit seeds, losses and recorded fit times; optimizer/reporting costs and failed/interrupted attempts kept separate |
| `status.json`, `run.log` | Current progress and background output |

Each history row's `k` is the number of completed steps; its step diagnostics
refer to the update that produced that state. CSV files are convenient views of
the durable journals. The full state and loss records remain the authoritative
results even if a process stops during a CSV export.

Every optimizer step uses two loss evaluations. Reporting uses
`len(evaluation_seeds)` fresh fits at every configured checkpoint, including k=0;
these additional fits do not update or select w. Reporting seeds are below
2**32 and optimizer training seeds are above it. Modes share the configured initial
state and indexed random streams. Summary SE is across run means, not across the
reporting seeds; only complete reporting panels enter that summary.

Resume requires the same numerical source, prepared data and runtime. A changed
`config.py` is allowed because its saved values are restored. Concurrent writers
are rejected. This runner currently executes DF-SSQP AI/ZI only; the `baselines`
setting is reserved for later comparison work. No final-test data is loaded.

Raw ZIPs and prepared monthly data are tracked in Git. Experiment outputs under
`runs/` and local environments remain excluded.


## Plot saved results

```bash
python -m experiments.plot --run runs/my_experiment
```

`experiments/plot.py` reads that experiment's `settings.json` and `summary.csv`.
It saves separate `figures/mse.pdf` and `figures/feasibility.pdf`, with PNG copies.
Colors, line width, font and figure size are collected at the top of the file.
No training is run and the saved experiment results are unchanged.

Lines show the run means and bands show ±1 SE across runs at the recorded evaluation
checkpoints. Feasibility uses a logarithmic axis when positive values exist; zero
values and nonpositive band bounds are omitted without an artificial floor. A single
run is drawn without an SE band. Actual available run counts and these display
conventions are saved in `figures/notes.txt`; partial results are explicitly identified.
