"""Run the single-sample ZO-RSGD baseline from config.py.

python -m experiments.zo_rsgd --background
python -m experiments.zo_rsgd --pause runs/<experiment>
python -m experiments.zo_rsgd --resume runs/<experiment> --background
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import config
from algorithms.zo_rsgd import initialize, step
from experiments import run as storage  # Existing file I/O and saved-fit tools.

ROOT = Path(__file__).resolve().parents[1]


def validate_settings():
    for name in ('zo_rsgd_step_coefficient', 'zo_rsgd_mu'):
        value = getattr(config, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive.')
    if config.zo_rsgd_m != 1:
        raise ValueError('This implementation uses exactly one sample (m=1).')
    for name in ('zo_rsgd_run_seeds', 'zo_rsgd_evaluation_seeds'):
        seeds = getattr(config, name)
        if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError(f'{name} must contain distinct nonnegative integers.')
    panel = config.zo_rsgd_evaluation_seeds
    if any(s >= 2**32 for s in panel):
        raise ValueError('Evaluation seeds must be below 2**32; optimizer seeds are above it.')
    steps, points = config.zo_rsgd_steps, config.zo_rsgd_evaluation_steps
    if type(steps) is not int or steps < 1:
        raise ValueError('zo_rsgd_steps must be a positive integer.')
    if (not points or any(type(k) is not int for k in points)
            or points != sorted(set(points)) or points[0] != 0 or points[-1] != steps):
        raise ValueError('zo_rsgd_evaluation_steps must increase from 0 to zo_rsgd_steps.')


def fingerprints():
    files = ('experiments/zo_rsgd.py', 'experiments/run.py',
             'algorithms/zo_rsgd.py', 'models/objective.py', 'models/single_index.py',
             'data/processed/training.npz', 'data/processed/validation.npz',
             'data/processed/metadata.json')
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}


def prepare(folder=None):
    """Freeze settings and budgets without evaluating or training any model."""
    validate_settings()
    storage.check_data_settings()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in config.zo_rsgd_experiment_name)
    folder = Path(folder).resolve() if folder else ROOT / config.runs_directory / f'{stamp}_{name}'
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True)
    settings = {k: v for k, v in vars(config).items() if not k.startswith('_') and not callable(v)}
    saved = {'config': settings, 'sha256': fingerprints(), 'runtime': storage.runtime(),
             'git_commit': revision.stdout.strip(), 'git_dirty': bool(dirty.stdout), 'created_utc': stamp,
             'parameters': {'eta': config.zo_rsgd_step_coefficient / np.sqrt(config.zo_rsgd_steps),
                            'mu': config.zo_rsgd_mu, 'm': config.zo_rsgd_m}}
    cases = len(config.zo_rsgd_run_seeds)
    optimization = cases * config.zo_rsgd_steps * 2
    evaluation = len(config.zo_rsgd_evaluation_seeds) * (
        1 + cases * (len(config.zo_rsgd_evaluation_steps) - 1))
    folder.mkdir(parents=True, exist_ok=False)
    storage.sync_directory(folder.parent)
    storage.save_json(folder / 'settings.json', saved)
    storage.save_json(folder / 'budget.json', {'optimization_fits': optimization,
                      'evaluation_fits': evaluation, 'total_fits': optimization + evaluation})
    storage.save_json(folder / 'status.json', {'state': 'prepared', 'pid': None, 'k': 0})
    return folder


def run_folder(folder, seed):
    return folder / f'run_{seed}'


def export_run(folder, history, completed, initial_completed):
    storage.write_csv(folder / 'history.csv', [row['metrics'] for row in history])
    rows = []
    for row in history:
        k = row['state']['k']
        if k not in config.zo_rsgd_evaluation_steps:
            continue
        panel = initial_completed if k == 0 else completed
        fits = [panel[f'eval:{k}:{seed}'] for seed in config.zo_rsgd_evaluation_seeds
                if f'eval:{k}:{seed}' in panel]
        if fits:
            rows.append({'k': k, 'evaluation_fits': len(fits),
                         'panel_complete': len(fits) == len(config.zo_rsgd_evaluation_seeds),
                         'mean_mse': float(np.mean([fit['mse'] for fit in fits])),
                         'feasibility': row['metrics']['feasibility']})
    storage.write_csv(folder / 'evaluations.csv', rows)


def export_summary(folder):
    """Average complete reporting panels; SE is across stochastic runs."""
    panels = {}
    for seed in config.zo_rsgd_run_seeds:
        path = run_folder(folder, seed) / 'evaluations.csv'
        if path.exists():
            with path.open() as file:
                for row in csv.DictReader(file):
                    if row['panel_complete'] == 'True':
                        panels.setdefault(int(row['k']), []).append(row)
    rows = []
    for k, panel in sorted(panels.items()):
        values = np.array([float(r['mean_mse']) for r in panel])
        se = (0.0 if k == 0 else float(values.std(ddof=1) / np.sqrt(len(values)))) if len(values) > 1 else ''
        rows.append({'k': k, 'runs': len(panel), 'mean_mse': float(values.mean()),
                     'se_mse': se,
                     'mean_feasibility': float(np.mean([float(r['feasibility']) for r in panel]))})
    storage.write_csv(folder / 'summary.csv', rows)
    costs = {role: {key: 0 for key in ('done', 'failed', 'interrupted', 'start')}
             for role in ('opt', 'eval')}
    for role in costs:
        costs[role]['recorded_fit_seconds'] = 0.0
    for path in folder.glob('**/fits.jsonl'):
        for event in storage.read_journal(path):
            role = event['key'].split(':')[0]
            costs[role][event['event']] += 1
            costs[role]['recorded_fit_seconds'] += event.get('seconds', 0.0)
    costs['total_done'] = costs['opt']['done'] + costs['eval']['done']
    costs['note'] = 'Shared initial fits counted once; failed/interrupted attempts are additional work.'
    storage.save_json(folder / 'costs.json', costs)


def execute(folder, stop_after=None):
    """Restore -> finish the current evaluation -> step -> save -> repeat."""
    folder = Path(folder).resolve()
    with (folder / 'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This experiment already has a running process.') from None
        saved = json.loads((folder / 'settings.json').read_text())
        if saved['sha256'] != fingerprints() or saved['runtime'] != storage.runtime():
            raise ValueError('Code, data or runtime changed. Restore saved inputs before resuming.')
        for key, value in saved['config'].items():
            setattr(config, key, value)
        validate_settings()
        eta = config.zo_rsgd_step_coefficient / np.sqrt(config.zo_rsgd_steps)
        mu = config.zo_rsgd_mu
        torch.set_num_threads(config.torch_threads)
        data = []
        for name in ('training', 'validation'):
            with np.load(ROOT / f'data/processed/{name}.npz', allow_pickle=False) as file:
                data.append({key: file[key] for key in ('z', 'u', 'y')})
        stopping, steps_this_session = False, 0
        status = {'state': 'running', 'pid': os.getpid(), 'eta': float(eta), 'mu': mu, 'outer_seed': None, 'k': 0}

        def request_pause(signum, frame):
            nonlocal stopping
            stopping = True

        def pause_requested():
            return stopping or (folder / 'PAUSE').exists()

        def publish(value):
            status.update(state=value, updated_utc=datetime.now(timezone.utc).isoformat())
            storage.save_json(folder / 'status.json', status)

        def pause():
            export_summary(folder)
            (folder / 'PAUSE').unlink(missing_ok=True)
            publish('paused')
            print(f'Paused: eta={status["eta"]}, run={status["outer_seed"]}, step={status["k"]}', flush=True)

        handlers = {s: signal.signal(s, request_pause) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            publish('running')
            initial = initialize()
            initial_folder = folder / 'initial'
            initial_folder.mkdir(exist_ok=True)
            storage.sync_directory(folder)
            initial_fit, initial_completed = storage.fit_evaluator(initial_folder, *data)
            for seed in config.zo_rsgd_evaluation_seeds:
                if pause_requested():
                    pause()
                    return
                initial_fit(f'eval:0:{seed}', initial['w'], seed)

            for outer_seed in config.zo_rsgd_run_seeds:
                run = run_folder(folder, outer_seed)
                run.mkdir(parents=True, exist_ok=True)
                storage.sync_directory(run.parent)
                storage.sync_directory(folder)
                history = storage.read_journal(run / 'history.jsonl')
                if not history:
                    record = {'state': initial, 'metrics': {'k': 0, 'eta': eta, 'mu': mu,
                              'feasibility': abs(float(initial['w'] @ initial['w']) - 1)}}
                    storage.append(run / 'history.jsonl', record)
                    history = [record]
                if ([r['state']['k'] for r in history] != list(range(len(history)))
                        or history[-1]['state']['k'] > config.zo_rsgd_steps):
                    raise ValueError('Non-contiguous iteration journal.')
                state = dict(history[-1]['state'])
                state['w'] = np.asarray(state['w'], dtype=np.float64)
                fit, completed = storage.fit_evaluator(run, *data)
                status.update(eta=eta, outer_seed=outer_seed, k=state['k'])
                publish('running')
                while True:
                    k = state['k']
                    if k > 0 and k in config.zo_rsgd_evaluation_steps:
                        for seed in config.zo_rsgd_evaluation_seeds:
                            if pause_requested():
                                break
                            fit(f'eval:{k}:{seed}', state['w'], seed)
                    if k in config.zo_rsgd_evaluation_steps or pause_requested():
                        export_run(run, history, completed, initial_completed)
                        export_summary(folder)
                    if pause_requested() or (stop_after is not None and steps_this_session >= stop_after):
                        export_run(run, history, completed, initial_completed)
                        pause()
                        return
                    if k == config.zo_rsgd_steps:
                        break
                    sides = iter(('current', 'probe'))

                    def loss(w, seed):
                        return fit(f'opt:{k}:{next(sides)}', w, seed)

                    previous_w = state['w']
                    state, trace = step(state, outer_seed, eta, mu, loss)
                    metrics = {'k': state['k'], 'eta': eta, 'mu': mu,
                               **{key: trace[key] for key in ('seed', 'L_current', 'L_probe')},
                               'update_norm': float(np.linalg.norm(state['w'] - previous_w)),
                               'gradient_norm': float(np.linalg.norm(trace['g_hat'])),
                               'feasibility': abs(float(state['w'] @ state['w']) - 1)}
                    record = {'state': state, 'metrics': metrics}
                    storage.append(run / 'history.jsonl', record)
                    history.append(record)
                    steps_this_session += 1
                    status['k'] = state['k']
                    publish('running')
                    if state['k'] in config.zo_rsgd_evaluation_steps:
                        print(f'eta={eta}, run={outer_seed}: {state["k"]}/{config.zo_rsgd_steps}', flush=True)
            publish('completed')
            print(f'Completed: {folder}', flush=True)
        except Exception as error:
            status['error'] = str(error)
            publish('failed')
            raise
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--output', type=Path)
    action.add_argument('--resume', type=Path)
    action.add_argument('--pause', type=Path)
    action.add_argument('--status', type=Path)
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Pause after this many new iterations this invocation.')
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        parser.error('--stop-after must be positive.')
    if args.pause or args.status:
        folder = (args.pause or args.status).resolve()
        if not (folder / 'settings.json').exists():
            parser.error('Not an experiment folder.')
        if args.pause:
            (folder / 'PAUSE').touch()
            print(f'Pause requested: {folder}')
        else:
            storage.show_status(folder)
        return
    folder = args.resume.resolve() if args.resume else prepare(args.output)
    print(f'Results: {folder}', flush=True)
    if args.background:
        command = [sys.executable, '-u', '-m', 'experiments.zo_rsgd', '--resume', str(folder)]
        if args.stop_after is not None:
            command += ['--stop-after', str(args.stop_after)]
        with (folder / 'run.log').open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'Background process {child.pid}; progress/errors in {folder / "run.log"}', flush=True)
    else:
        execute(folder, args.stop_after)


if __name__ == '__main__':
    main()
