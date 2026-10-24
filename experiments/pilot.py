"""Prepare the common step-scale pilot without starting any training.

python -m experiments.pilot --output runs/<pilot>
python -m experiments.pilot --report runs/<pilot>

Each saved child campaign uses an existing runner; its command is recorded,
never executed here. Reporting reads the frozen protocol and completed panels.
"""

import argparse
import copy
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config
from experiments import run as df_runner
from experiments import zo_rsgd as zo_runner
from experiments import mistp as mistp_runner

ROOT = Path(__file__).resolve().parents[1]
WARNING = ('This pilot gives every candidate the same iteration count, not the same '
           'loss-evaluation budget: DF-SSQP and ZO-RSGD use two fits per iteration, '
           'whereas MiSTP uses three. It selects step coefficients within each method; '
           'it does not establish an overall method winner.')
SELECTION_NOTE = ('Inspect run variability and DF feasibility before adopting the lowest-MSE '
                  'provisional coefficient; no overall method winner is selected.')
PANEL_USE = ('Pilot evaluation seeds are separate from the main evaluation panel and '
             'are used to select coefficients. Their reported losses are development '
             'results, not a final independent assessment of the selected settings.')


def public_config():
    return {name: copy.deepcopy(value) for name, value in vars(config).items()
            if not name.startswith('_') and not callable(value)}


@contextmanager
def settings(**changes):
    """Restore every public config entry, even when a child preparation fails."""
    original = {name: copy.deepcopy(value) for name, value in vars(config).items()
                if not name.startswith('_')}
    try:
        for name, value in changes.items():
            setattr(config, name, copy.deepcopy(value))
        yield
    finally:
        for name in list(vars(config)):
            if not name.startswith('_') and name not in original:
                delattr(config, name)
        vars(config).update(original)


def fingerprints():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ('experiments/pilot.py', 'config.py')}


def validate_settings():
    if type(config.pilot_steps) is not int or config.pilot_steps < 1:
        raise ValueError('pilot_steps must be a positive integer.')
    for name in ('pilot_run_seeds', 'pilot_evaluation_seeds'):
        seeds = getattr(config, name)
        if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError(f'{name} must contain distinct nonnegative integers.')
    if set(config.pilot_run_seeds) & set(config.outer_seeds):
        raise ValueError('Pilot run seeds must be separate from the main run seeds.')
    panel = config.pilot_evaluation_seeds
    if any(s >= 2**32 for s in panel) or set(panel) & set(config.evaluation_seeds):
        raise ValueError('Pilot evaluation seeds must be below 2**32 and separate from the main panel.')
    if set(panel) & set(config.pilot_run_seeds):
        raise ValueError('Pilot run and evaluation seed labels must be separate.')
    points = config.pilot_evaluation_steps
    if (not points or any(type(k) is not int for k in points)
            or points != sorted(set(points)) or points[0] != 0 or points[-1] != config.pilot_steps):
        raise ValueError('pilot_evaluation_steps must increase from 0 to pilot_steps.')
    for name in ('pilot_df_iota1s', 'pilot_zo_coefficients', 'pilot_mistp_alphas'):
        values = getattr(config, name)
        if not values or len(set(values)) != len(values) or any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f'{name} must contain distinct finite positive coefficients.')


def prepare(folder=None):
    """Freeze seven child campaigns and exact budgets; do not evaluate any loss."""
    validate_settings()
    df_runner.check_data_settings()
    frozen = public_config()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in config.pilot_experiment_name)
    folder = Path(folder).resolve() if folder else ROOT / config.runs_directory / f'{stamp}_{name}'
    folder.mkdir(parents=True, exist_ok=False)
    df_runner.sync_directory(folder.parent)
    campaigns = []
    steps, seeds = frozen['pilot_steps'], frozen['pilot_run_seeds']
    panel, points = frozen['pilot_evaluation_seeds'], frozen['pilot_evaluation_steps']

    def save_campaign(name, method, coefficients, module, runner, changes):
        with settings(**changes):
            child = runner.prepare(folder / name)
        if method == 'DF-SSQP':
            cases = 2 * len(seeds)
            optimization = cases * steps * 2
            evaluation = cases * len(points) * len(panel)
            df_runner.save_json(child / 'budget.json', {
                'optimization_fits': optimization, 'evaluation_fits': evaluation,
                'total_fits': optimization + evaluation})
            df_runner.save_json(child / 'status.json', {'state': 'prepared', 'pid': None, 'k': 0})
        command = [sys.executable, '-m', module, '--resume', str(child), '--background']
        campaigns.append({'name': name, 'method': method, 'folder': name,
                          'coefficient': coefficients[0] if len(coefficients) == 1 else None,
                          'coefficients': coefficients, 'module': module, 'command': command,
                          'shell_command': shlex.join(command),
                          'settings_sha256': hashlib.sha256((child / 'settings.json').read_bytes()).hexdigest(),
                          'budget': json.loads((child / 'budget.json').read_text())})

    try:
        for coefficient in frozen['pilot_df_iota1s']:
            name = f'df_iota1_{coefficient:g}'
            save_campaign(name, 'DF-SSQP', [coefficient], 'experiments.run', df_runner, {
                'modes': ('AI', 'ZI'), 'iota1': coefficient, 'sqp_steps': steps,
                'outer_seeds': seeds, 'evaluation_seeds': panel,
                'evaluation_steps': points, 'experiment_name': name})
        for coefficient in frozen['pilot_zo_coefficients']:
            name = f'zo_c_{coefficient:g}'
            save_campaign(name, 'ZO-RSGD', [coefficient], 'experiments.zo_rsgd', zo_runner, {
                'zo_rsgd_step_coefficient': coefficient, 'zo_rsgd_steps': steps,
                'zo_rsgd_run_seeds': seeds, 'zo_rsgd_evaluation_seeds': panel,
                'zo_rsgd_evaluation_steps': points, 'zo_rsgd_experiment_name': name})
        save_campaign('mistp', 'MiSTP', list(frozen['pilot_mistp_alphas']),
                      'experiments.mistp', mistp_runner, {
                          'mistp_alphas': frozen['pilot_mistp_alphas'], 'mistp_steps': steps,
                          'mistp_run_seeds': seeds, 'mistp_evaluation_seeds': panel,
                          'mistp_evaluation_steps': points, 'mistp_experiment_name': 'mistp'})
        budget = {key: sum(c['budget'][key] for c in campaigns)
                  for key in ('optimization_fits', 'evaluation_fits', 'total_fits')}
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True)
        protocol = {'created_utc': stamp, 'git_commit': revision.stdout.strip(),
                    'git_dirty': bool(dirty.stdout), 'config': frozen, 'campaigns': campaigns,
                    'budget': budget, 'sha256': fingerprints(), 'runtime': df_runner.runtime(),
                    'warning': WARNING, 'panel_use': PANEL_USE, 'selection_note': SELECTION_NOTE,
                    'purpose': 'Select a step coefficient within each method under a common pilot protocol.'}
        df_runner.save_json(folder / 'protocol.json', protocol)
        df_runner.save_json(folder / 'budget.json', budget)
        df_runner.save_json(folder / 'status.json', {'state': 'prepared', 'pid': None})
    except Exception as error:
        df_runner.save_json(folder / 'status.json', {'state': 'preparation_failed', 'pid': None,
                                                  'error': str(error)})
        raise
    return folder


def report(folder):
    """Aggregate complete seed panels; coefficient choices remain provisional."""
    folder = Path(folder).resolve()
    protocol = json.loads((folder / 'protocol.json').read_text())
    frozen = protocol['config']
    seeds, points = frozen['pilot_run_seeds'], frozen['pilot_evaluation_steps']
    panel_size, final_step = len(frozen['pilot_evaluation_seeds']), frozen['pilot_steps']
    rows, expected = [], {}
    for campaign in protocol['campaigns']:
        child = folder / campaign['folder']
        digest = hashlib.sha256((child / 'settings.json').read_bytes()).hexdigest()
        if digest != campaign['settings_sha256']:
            raise ValueError(f'Frozen child settings changed: {campaign["name"]}')
        variants = ('AI', 'ZI') if campaign['method'] == 'DF-SSQP' else (None,)
        for variant in variants:
            method = f'DF-SSQP {variant}' if variant else campaign['method']
            expected.setdefault(method, []).extend(campaign['coefficients'])
            for coefficient in campaign['coefficients']:
                panels = {}
                for seed in seeds:
                    if variant:
                        run = child / f'{variant}_{seed}'
                    elif method == 'ZO-RSGD':
                        run = child / f'run_{seed}'
                    else:
                        run = child / f'alpha_{float(coefficient)}' / f'run_{seed}'
                    path = run / 'evaluations.csv'
                    if not path.exists():
                        continue
                    with path.open() as file:
                        entries = list(csv.DictReader(file))
                    seen = set()
                    for entry in entries:
                        k = int(entry['k'])
                        if k in seen or k not in points:
                            raise ValueError(f'Duplicate or unexpected checkpoint in {path}')
                        seen.add(k)
                        if entry['panel_complete'] != 'True' or int(entry['evaluation_fits']) != panel_size:
                            continue
                        mse, feasibility = float(entry['mean_mse']), float(entry['feasibility'])
                        if not np.isfinite(mse) or not np.isfinite(feasibility) or feasibility < 0:
                            raise ValueError(f'Invalid complete evaluation panel in {path}')
                        panels.setdefault(k, []).append((mse, feasibility))
                for k, values in sorted(panels.items()):
                    values = np.asarray(values)
                    means = values.mean(axis=0)
                    se = values.std(axis=0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else ('', '')
                    rows.append({'method': method, 'coefficient': coefficient, 'k': k,
                                 'runs': len(values), 'mean_mse': float(means[0]),
                                 'se_mse': float(se[0]) if len(values) > 1 else '',
                                 'mean_feasibility': float(means[1]),
                                 'se_feasibility': float(se[1]) if len(values) > 1 else ''})
    fields = ('method', 'coefficient', 'k', 'runs', 'mean_mse', 'se_mse',
              'mean_feasibility', 'se_feasibility')
    with (folder / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    methods = {}
    for method, coefficients in expected.items():
        final = [row for row in rows if row['method'] == method and row['k'] == final_step]
        complete = (len(final) == len(coefficients)
                    and all(row['runs'] == len(seeds) for row in final))
        methods[method] = {'complete': complete,
                           'provisional_coefficient': min(final, key=lambda row: row['mean_mse'])['coefficient']
                           if complete else None,
                           'final_results': final}
    selection = {'complete': all(value['complete'] for value in methods.values()),
                 'criterion': 'Lowest mean final-checkpoint evaluation MSE within each method.',
                 'provisional': True, 'warning': protocol['warning'],
                 'panel_use': protocol['panel_use'], 'selection_note': protocol['selection_note'],
                 'methods': methods}
    df_runner.save_json(folder / 'selection.json', selection)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--output', type=Path)
    action.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.report:
        selection = report(args.report)
        print(f'Report saved: {args.report.resolve()} (complete={selection["complete"]})')
    else:
        folder = prepare(args.output)
        print(f'Prepared only; no training started: {folder}')
        print(f'Child launch commands are recorded in {folder / "protocol.json"}.')


if __name__ == '__main__':
    main()
