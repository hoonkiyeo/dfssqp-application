# DF-SSQP Application

Research code for applying derivative-free stochastic SQP (DF-SSQP) to a
constrained single-index learning problem.

## Setup

Run commands from the repository root:

```bash
conda env create -f environment.yml
conda activate dfssqp-application
```

## Project structure

```text
config.py       Experiment parameters, training settings and seeds
algorithms/     DF-SSQP, ZO-RSGD and MiSTP
models/         Single-index model, training objective and fixed baselines
data/           Raw data, preparation and processed arrays
experiments/    Experiment runners and plotting
runs/           Saved settings, results and checkpoints (local only)
docs/           Formulation, notes and references
tests/          Implementation checks
```

## Run DF-SSQP

Edit the settings in `config.py`, then run:

```bash
python -m experiments.run --background
```

Results are saved under `runs/`. Use the printed folder path in place of
`runs/my_experiment` below. Resuming uses that experiment's saved settings.

```bash
python -m experiments.run --status runs/my_experiment
python -m experiments.run --pause runs/my_experiment
python -m experiments.run --resume runs/my_experiment --background
```

To plot saved DF-SSQP results:

```bash
python -m experiments.plot --run runs/my_experiment
```

ZO-RSGD and MiSTP have separate runners in `experiments/`; configure their
`zo_rsgd_*` or `mistp_*` settings before use.

## Data and checks

Prepared data are included in `data/processed/`. To rebuild them from `data/raw/`
and run the implementation checks:

```bash
python -m data.prepare
python -m pytest tests -q
```

See [docs/references.md](docs/references.md) for the referenced papers.
