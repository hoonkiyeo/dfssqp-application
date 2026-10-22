"""Evaluate fixed comparisons using an existing experiment's frozen settings.

python -m experiments.baselines --run runs/<experiment>
Repeat the same command to resume: completed fits are reused. No SQP steps run.
"""

import argparse
import fcntl
import hashlib
import json
import signal
from pathlib import Path

import numpy as np
import torch

import config
from models.baselines import aggregate_only, fixed_directions
from experiments.run import ROOT, fit_evaluator, read_journal, runtime, save_json, write_csv


def check_inputs(folder):
    """Check the model/data used by DF-SSQP, then pin this comparison's code too."""
    saved = json.loads((folder / 'settings.json').read_text())
    shared = ('models/single_index.py', 'models/objective.py', 'experiments/run.py',
              'data/processed/training.npz', 'data/processed/validation.npz',
              'data/processed/metadata.json')
    files = shared + ('models/baselines.py', 'experiments/baselines.py')
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    if saved['runtime'] != runtime() or any(saved['sha256'].get(name) != hashes[name] for name in shared):
        raise ValueError('Saved experiment model, data or runtime differs from the current inputs.')
    settings = saved['config']
    names, seeds = settings['baselines'], settings['evaluation_seeds']
    if not names or len(set(names)) != len(names) or not set(names) <= {'aggregate_only', 'mean', 'pca'}:
        raise ValueError('baselines must select aggregate_only, mean and/or pca without duplicates.')
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError('Use distinct reporting seeds below 2**32, separate from optimizer seeds.')
    manifest = {'parent_settings_sha256': hashlib.sha256((folder / 'settings.json').read_bytes()).hexdigest(),
                'sha256': hashes, 'runtime': saved['runtime']}
    return settings, manifest


def export_results(output, names, seeds, aggregate, completed):
    """A fixed direction has one reporting panel, not a panel per SQP run/step."""
    evaluations, summary = [], []
    for name in names:
        if name == 'aggregate_only':
            summary.append({'model': name, 'evaluation_fits': 0, 'mean_mse': aggregate['mse'],
                            'se_mse': '', 'panel_complete': True})
            continue
        values = []
        for seed in seeds:
            fit = completed.get(f'eval:{name}:{seed}')
            if fit is not None:
                values.append(fit['mse'])
                evaluations.append({'model': name, 'seed': seed, 'mse': fit['mse']})
        n = len(values)
        summary.append({'model': name, 'evaluation_fits': n,
                        'mean_mse': float(np.mean(values)) if n else '',
                        'se_mse': float(np.std(values, ddof=1)/np.sqrt(n)) if n > 1 else '',
                        'panel_complete': n == len(seeds)})
    write_csv(output / 'evaluations.csv', evaluations)
    write_csv(output / 'summary.csv', summary)
    events = read_journal(output / 'fits.jsonl')
    save_json(output / 'costs.json', {
        'neural_fits': {kind: sum(e['event'] == kind for e in events)
                       for kind in ('start', 'done', 'failed', 'interrupted')},
        'recorded_fit_seconds': sum(e.get('seconds', 0) for e in events),
        'aggregate_ols': 'deterministic; no neural fits or training seed',
        'note': 'Reporting only; no optimization. Interrupted-attempt work is unknown.',
    })


def execute(folder):
    """Restore -> construct fixed comparisons -> evaluate seeds -> save."""
    folder = Path(folder).resolve()
    settings, manifest = check_inputs(folder)
    output = folder / 'baselines'
    output.mkdir(exist_ok=True)
    with (output / 'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Baseline evaluation is already running here.') from None
        path = output / 'settings.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError('Baseline inputs changed; restore the saved inputs before resuming.')
        if not path.exists():
            save_json(path, manifest)

        previous = {key: getattr(config, key) for key in settings}
        previous_threads = torch.get_num_threads()
        stopping = False

        def request_pause(signum, frame):
            nonlocal stopping
            stopping = True

        handlers = {s: signal.signal(s, request_pause) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            for key, value in settings.items():
                setattr(config, key, value)
            torch.set_num_threads(config.torch_threads)
            data = []
            for split in ('training', 'validation'):
                with np.load(ROOT / f'data/processed/{split}.npz', allow_pickle=False) as file:
                    data.append({key: file[key] for key in ('z', 'u', 'y')})
            training, validation = data
            names, seeds = config.baselines, config.evaluation_seeds
            directions = fixed_directions(training) if any(n != 'aggregate_only' for n in names) else {}
            save_json(output / 'directions.json', {n: directions[n] for n in names if n in directions})
            aggregate = aggregate_only(training, validation) if 'aggregate_only' in names else None
            if aggregate is not None:
                save_json(output / 'aggregate_only.json', aggregate)
            fit, completed = fit_evaluator(output, training, validation)
            export_results(output, names, seeds, aggregate, completed)

            for name in names:
                if name == 'aggregate_only':
                    continue
                for seed in seeds:
                    if stopping:
                        print(f'Paused. Repeat the same command to resume: {output}', flush=True)
                        return output
                    fit(f'eval:{name}:{seed}', directions[name], seed)
                    export_results(output, names, seeds, aggregate, completed)
                print(f'{name}: {len(seeds)} reporting seeds complete.', flush=True)
            print(f'Baseline results: {output}', flush=True)
            return output
        finally:
            for key, value in previous.items():
                setattr(config, key, value)
            torch.set_num_threads(previous_threads)
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path, help='Existing DF-SSQP result folder.')
    args = parser.parse_args()
    execute(args.run)


if __name__ == '__main__':
    main()
