"""Run AI/ZI from config.py; save every completed fit and iteration for resumption.

python -m experiments.run --background
python -m experiments.run --pause runs/<experiment>
python -m experiments.run --resume runs/<experiment> --background
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import config
from algorithms.df_ssqp import initialize, step
from models.objective import evaluate

ROOT = Path(__file__).resolve().parents[1]


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot save {type(value).__name__}")


def save_json(path, value):
    """Replace a small status/settings file only after its contents reach disk."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as file:
        json.dump(value, file, default=json_value, allow_nan=False, indent=2)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append(path, record):
    new = not path.exists()
    with path.open('a') as file:
        file.write(json.dumps(record, default=json_value, allow_nan=False) + '\n')
        file.flush()
        os.fsync(file.fileno())
    if new:
        sync_directory(path.parent)


def read_journal(path):
    """A complete line is a commit. Preserve/truncate only an interrupted last line."""
    if not path.exists():
        return []
    data = path.read_bytes()
    end = data.rfind(b'\n') + 1
    if end != len(data):
        backup = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
        with backup.open('wb') as file:
            file.write(data[end:])
            file.flush()
            os.fsync(file.fileno())
        sync_directory(path.parent)
        with path.open('r+b') as file:
            file.truncate(end)
            file.flush()
            os.fsync(file.fileno())
    return [json.loads(line) for line in data[:end].splitlines()]


def fingerprints():
    files = ('experiments/run.py', 'algorithms/df_ssqp.py', 'models/objective.py',
             'models/single_index.py', 'data/processed/training.npz',
             'data/processed/validation.npz', 'data/processed/metadata.json')
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}


def runtime():
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'torch': torch.__version__, 'machine': platform.machine(), 'system': platform.system()}


def validate_settings():
    if not config.modes or len(set(config.modes)) != len(config.modes) or not set(config.modes) <= {'AI', 'ZI'}:
        raise ValueError('modes must contain AI and/or ZI without duplicates.')
    for name in ('outer_seeds', 'evaluation_seeds'):
        seeds = getattr(config, name)
        if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError(f'{name} must be distinct nonnegative integers.')
    if any(s >= 2**32 for s in config.evaluation_seeds):
        raise ValueError('Evaluation seeds must be below 2**32; optimizer fit seeds are above it.')
    if type(config.sqp_steps) is not int or config.sqp_steps < 1:
        raise ValueError('sqp_steps must be a positive integer.')
    points = config.evaluation_steps
    if (any(type(k) is not int for k in points) or points != sorted(set(points))
            or not points or points[0] != 0 or points[-1] != config.sqp_steps):
        raise ValueError('evaluation_steps must increase from 0 to sqp_steps.')


def check_data_settings():
    metadata = json.loads((ROOT / 'data/processed/metadata.json').read_text())
    for key in ('training_months', 'validation_months', 'data_vintage',
                'portfolio_weighting', 'benchmark_ddof'):
        value = getattr(config, key)
        if metadata[key] != (list(value) if isinstance(value, tuple) else value):
            raise ValueError(f'Prepared data does not match {key}; run data.prepare first.')
    if len(metadata['industry_names']) != config.dimension:
        raise ValueError('Prepared data dimension differs from config.')
    for source in ('returns', 'calendar'):
        if metadata['sources'][source]['sha256'] != getattr(config, source + '_sha256'):
            raise ValueError('Prepared data source hash differs from config.')


def prepare(folder=None):
    """Freeze config and inputs once. Existing folders are never overwritten."""
    validate_settings()
    check_data_settings()
    settings = {key: value for key, value in vars(config).items()
                if not key.startswith('_') and not callable(value)}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in config.experiment_name)
    folder = Path(folder).resolve() if folder else ROOT / config.runs_directory / f'{stamp}_{name}'
    hashes = fingerprints()
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=False)
    folder.mkdir(parents=True, exist_ok=False)
    sync_directory(folder.parent)
    save_json(folder / 'settings.json', {'config': settings, 'sha256': hashes,
              'runtime': runtime(), 'git_commit': revision.stdout.strip(), 'created_utc': stamp})
    return folder


def fit_evaluator(folder, training, validation):
    """Reuse durable results; an interrupted attempt is counted separately, then retried."""
    path = folder / 'fits.jsonl'
    events = read_journal(path)
    pending, completed = {}, {}
    for event in events:
        if event['event'] == 'start':
            pending[event['id']] = event
        else:
            pending.pop(event['id'], None)
        if event['event'] == 'done':
            completed[event['key']] = event
    for event in pending.values():
        append(path, dict(event, event='interrupted'))
    if pending:
        print(f'{folder.name}: {len(pending)} interrupted fit attempt(s); cost is unknown.', flush=True)

    def fit(key, w, seed):
        digest = hashlib.sha256(np.asarray(w, dtype=np.float64).tobytes()).hexdigest()
        if key in completed:
            result = completed[key]
            if result['seed'] != seed or result['w_sha256'] != digest:
                raise ValueError(f'Saved fit does not match the requested w/seed: {key}')
            return result['mse']
        event = {'id': uuid.uuid4().hex, 'key': key, 'seed': int(seed), 'w_sha256': digest}
        append(path, dict(event, event='start'))
        start = time.perf_counter()
        try:
            mse = float(evaluate(w, seed, training, validation))
            if not np.isfinite(mse):
                raise FloatingPointError('Nonfinite loss.')
        except Exception as error:
            append(path, dict(event, event='failed', seconds=time.perf_counter()-start, error=str(error)))
            raise
        result = dict(event, event='done', mse=mse, seconds=time.perf_counter()-start)
        append(path, result)
        completed[key] = result
        return mse
    return fit, completed


def export_run(folder, history, completed):
    """CSV files are views of the journals, never a second checkpoint authority."""
    rows = [row['metrics'] for row in history]
    write_csv(folder / 'history.csv', rows)
    evaluations = []
    for row in history:
        k = row['state']['k']
        fits = [completed[f'eval:{k}:{seed}'] for seed in config.evaluation_seeds
                if f'eval:{k}:{seed}' in completed]
        if k in config.evaluation_steps and fits:
            losses = [fit['mse'] for fit in fits]
            evaluations.append({'k': k, 'evaluation_fits': len(fits),
                                'panel_complete': len(fits) == len(config.evaluation_seeds),
                                'mean_mse': float(np.mean(losses)),
                                'feasibility': row['metrics']['feasibility']})
    write_csv(folder / 'evaluations.csv', evaluations)
    events = read_journal(folder / 'fits.jsonl')
    costs = {}
    for role in ('opt', 'eval'):
        selected = [e for e in events if e['key'].startswith(role + ':')]
        costs[role] = {kind: sum(e['event'] == kind for e in selected)
                       for kind in ('start', 'done', 'failed', 'interrupted')}
        costs[role]['recorded_fit_seconds'] = sum(e.get('seconds', 0) for e in selected)
    costs['note'] = 'Interrupted attempt work is unknown; successful losses are reused.'
    save_json(folder / 'costs.json', costs)


def write_csv(path, rows):
    if not rows:
        return
    with path.with_suffix('.csv.tmp').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(path.with_suffix('.csv.tmp'), path)


def export_summary(folder):
    """Mean and SE across available runs, using only COMPLETE evaluation panels."""
    rows = []
    for mode in config.modes:
        panels = {}
        for seed in config.outer_seeds:
            path = folder / f'{mode}_{seed}' / 'evaluations.csv'
            if path.exists():
                with path.open() as file:
                    for row in csv.DictReader(file):
                        if row['panel_complete'] == 'True':
                            panels.setdefault(int(row['k']), []).append(row)
        for k, panel in sorted(panels.items()):
            values = np.array([float(r['mean_mse']) for r in panel])
            feasibility = np.array([float(r['feasibility']) for r in panel])
            rows.append({'mode': mode, 'k': k, 'runs': len(panel), 'mean_mse': float(values.mean()),
                         'se_mse': float(values.std(ddof=1)/np.sqrt(len(values))) if len(values) > 1 else '',
                         'mean_feasibility': float(feasibility.mean()),
                         'se_feasibility': float(feasibility.std(ddof=1)/np.sqrt(len(panel))) if len(panel) > 1 else ''})
    write_csv(folder / 'summary.csv', rows)


def execute(folder, stop_after=None):
    """Clear experiment flow: restore -> evaluate current w -> step -> save -> repeat."""
    with (folder / 'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This experiment already has a running process.') from None
        saved = json.loads((folder / 'settings.json').read_text())
        if saved['sha256'] != fingerprints() or saved['runtime'] != runtime():
            raise ValueError('Code, data or runtime changed. Restore the saved inputs before resuming.')
        for key, value in saved['config'].items():
            setattr(config, key, value)
        validate_settings()
        torch.set_num_threads(config.torch_threads)
        data = []
        for name in ('training', 'validation'):
            with np.load(ROOT / f'data/processed/{name}.npz', allow_pickle=False) as file:
                data.append({key: file[key] for key in ('z', 'u', 'y')})
        training, validation = data
        stopping = False

        def request_pause(signum, frame):
            nonlocal stopping
            stopping = True

        handlers = {s: signal.signal(s, request_pause) for s in (signal.SIGINT, signal.SIGTERM)}
        status = {'state': 'running', 'pid': os.getpid(), 'mode': None, 'outer_seed': None, 'k': 0}
        steps_this_session = 0

        def pause_requested():
            return stopping or (folder / 'PAUSE').exists()

        def publish(state):
            status.update(state=state, updated_utc=datetime.now(timezone.utc).isoformat())
            save_json(folder / 'status.json', status)

        try:
            publish('running')
            for mode in config.modes:
                for outer_seed in config.outer_seeds:
                    run = folder / f'{mode}_{outer_seed}'
                    run.mkdir(exist_ok=True)
                    sync_directory(folder)
                    history = read_journal(run / 'history.jsonl')
                    if not history:
                        state = initialize()
                        initial = {'state': state, 'metrics': {'k': 0, 'feasibility': abs(float(state['w'] @ state['w'])-1)}}
                        append(run / 'history.jsonl', initial)
                        history = [initial]
                    if ([r['state']['k'] for r in history] != list(range(len(history)))
                            or history[-1]['state']['k'] > config.sqp_steps):
                        raise ValueError('Non-contiguous iteration journal.')
                    state = dict(history[-1]['state'])
                    for key in ('w', 'lam', 'g_bar', 'G_bar'):
                        state[key] = np.asarray(state[key], dtype=np.float64)
                    fit, completed = fit_evaluator(run, training, validation)
                    status.update(mode=mode, outer_seed=outer_seed, k=state['k'])
                    publish('running')
                    while True:
                        # Finish reporting at the current saved w BEFORE advancing it.
                        k = state['k']
                        if k in config.evaluation_steps:
                            for seed in config.evaluation_seeds:
                                if pause_requested():
                                    break
                                fit(f'eval:{k}:{seed}', state['w'], seed)
                        if k in config.evaluation_steps or pause_requested():
                            export_run(run, history, completed)
                        if pause_requested() or (stop_after is not None and steps_this_session >= stop_after):
                            export_run(run, history, completed)
                            export_summary(folder)
                            (folder / 'PAUSE').unlink(missing_ok=True)
                            publish('paused')
                            print(f'Paused: {run.name}, step {k}', flush=True)
                            return
                        if k == config.sqp_steps:
                            break
                        side = iter(('plus', 'minus'))

                        def loss(w, seed, fit=fit, k=k, side=side):
                            return fit(f'opt:{k}:{next(side)}', w, seed)

                        state, trace = step(state, outer_seed, mode, loss)
                        metrics = {'k': state['k'], 'feasibility': abs(float(state['w'] @ state['w'])-1),
                                   **{key: trace[key] for key in ('alpha', 'beta', 'b', 'seed', 'L_plus', 'L_minus', 'tau', 'nu', 's')},
                                   'direction_norm': float(np.linalg.norm(trace['d'])),
                                   'update_norm': float(np.linalg.norm(trace['s']*trace['d'])),
                                   'predicted_reduction': trace['reduction'],
                                   'singular_value': float(trace['singular_values'][0]),
                                   'clipped': trace['clipped'], 'zero_step': trace['roundoff']['zero_step'],
                                   'kkt_residual': trace['roundoff']['kkt_residual_inf']}
                        record = {'state': state, 'metrics': metrics}
                        append(run / 'history.jsonl', record)
                        history.append(record)
                        steps_this_session += 1
                        status['k'] = state['k']
                        publish('running')
                        if state['k'] in config.evaluation_steps:
                            print(f'{run.name}: {state["k"]}/{config.sqp_steps} steps', flush=True)
                    export_summary(folder)
            publish('completed')
            print(f'Completed: {folder}', flush=True)
        except Exception as error:
            status['error'] = str(error)
            publish('failed')
            raise
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


def show_status(folder):
    path = folder / 'status.json'
    status = json.loads(path.read_text()) if path.exists() else {'state': 'not started'}
    with (folder / 'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            active = False
        except BlockingIOError:
            active = True
    status['process_active'] = active
    if not active and status['state'] == 'running':
        status['state'] = 'interrupted; resume from saved results'
    print(json.dumps(status, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--output', type=Path, help='New result folder (must not exist).')
    action.add_argument('--resume', type=Path, help='Resume using this folder\'s saved settings.')
    action.add_argument('--pause', type=Path, help='Pause after the current iteration/evaluation fit.')
    action.add_argument('--status', type=Path)
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Pause after this many new SQP steps this invocation.')
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
            show_status(folder)
        return
    folder = args.resume.resolve() if args.resume else prepare(args.output)
    print(f'Results: {folder}', flush=True)
    if args.background:
        command = [sys.executable, '-u', '-m', 'experiments.run', '--resume', str(folder)]
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
