"""Freeze the fixed-setting main comparison without starting training.

python -m experiments.prepare_comparison --output runs/<comparison>

Each saved command uses an existing resumable runner. No scheduler or optimizer
is introduced here. MiSTP's equal-budget endpoint is separate from its horizon.
"""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import config
from experiments import run as df_runner
from experiments import zo_rsgd as zo_runner
from experiments import mistp as mistp_runner
from experiments.pilot import public_config, settings

ROOT = Path(__file__).resolve().parents[1]


def fingerprints():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ('config.py', 'experiments/prepare_comparison.py', 'experiments/pilot.py')}


def validate_settings():
    for runner in (df_runner, zo_runner, mistp_runner):
        runner.validate_settings()
    if set(config.modes) != {'AI', 'ZI'}:
        raise ValueError('Main comparison requires both AI and ZI.')
    for prefix in ('zo_rsgd', 'mistp'):
        for suffix, common in (('steps', config.sqp_steps), ('run_seeds', config.outer_seeds),
                               ('evaluation_seeds', config.evaluation_seeds)):
            if getattr(config, prefix + '_' + suffix) != common:
                raise ValueError(f'{prefix} must share the main horizon, run seeds and evaluation seeds.')
    if config.zo_rsgd_evaluation_steps != config.evaluation_steps:
        raise ValueError('ZO-RSGD must share the DF reporting checkpoints.')
    matched_step = (2 * config.sqp_steps) // 3
    if (matched_step < 1 or len(config.mistp_alphas) != 1
            or config.mistp_evaluation_steps != sorted(set(config.evaluation_steps + [matched_step]))):
        raise ValueError('MiSTP requires one fixed alpha and the shared plus equal-budget checkpoints.')
    if (set(config.outer_seeds) & set(config.pilot_run_seeds)
            or set(config.evaluation_seeds) & set(config.pilot_evaluation_seeds)):
        raise ValueError('Main run and evaluation seeds must be separate from the pilot.')
    if set(config.outer_seeds) & set(config.evaluation_seeds):
        raise ValueError('Run and evaluation seed labels must be separate.')


def prepare(folder=None):
    """Save four independently resumable campaigns; never evaluate a loss."""
    validate_settings()
    df_runner.check_data_settings()
    frozen = public_config()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    folder = Path(folder).resolve() if folder else ROOT / config.runs_directory / f'{stamp}_main_comparison'
    folder.mkdir(parents=True, exist_ok=False)
    df_runner.sync_directory(folder.parent)
    campaigns = []
    cases, steps, panel = len(config.outer_seeds), config.sqp_steps, len(config.evaluation_seeds)
    matched_step = (2 * steps) // 3
    definitions = [
        ('df_ai', 'DF-SSQP AI', 'experiments.run', df_runner, {'modes': ('AI',)}, steps, None),
        ('df_zi', 'DF-SSQP ZI', 'experiments.run', df_runner, {'modes': ('ZI',)}, steps, None),
        ('zo_rsgd', 'ZO-RSGD', 'experiments.zo_rsgd', zo_runner, {}, steps, None),
        ('mistp', 'MiSTP', 'experiments.mistp', mistp_runner, {}, matched_step, steps),
    ]
    try:
        for name, method, module, runner, changes, primary_step, secondary_step in definitions:
            with settings(**changes):
                child = runner.prepare(folder / name)
            if method.startswith('DF-SSQP'):
                opt, evaluation = cases * steps * 2, cases * len(config.evaluation_steps) * panel
                df_runner.save_json(child / 'budget.json', {'optimization_fits': opt,
                                    'evaluation_fits': evaluation, 'total_fits': opt + evaluation})
                df_runner.save_json(child / 'status.json', {'state': 'prepared', 'pid': None, 'k': 0})
            command = [sys.executable, '-m', module, '--resume', str(child), '--background']
            campaigns.append({'name': name, 'method': method, 'folder': name, 'module': module,
                              'command': command, 'shell_command': shlex.join(command),
                              'primary_step': primary_step, 'secondary_step': secondary_step,
                              'primary_optimization_fits_per_run': primary_step * (3 if method == 'MiSTP' else 2),
                              'settings_sha256': hashlib.sha256((child / 'settings.json').read_bytes()).hexdigest(),
                              'budget': json.loads((child / 'budget.json').read_text())})
        budget = {key: sum(c['budget'][key] for c in campaigns)
                  for key in ('optimization_fits', 'evaluation_fits', 'total_fits')}
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True)
        protocol = {
            'created_utc': stamp, 'git_commit': revision.stdout.strip(), 'git_dirty': bool(dirty.stdout),
            'config': frozen, 'campaigns': campaigns, 'budget': budget,
            'sha256': fingerprints(), 'runtime': df_runner.runtime(), 'coefficients_fixed': True,
            'purpose': 'Confirm fixed pilot-selected settings on fresh paired stochastic runs.',
            'primary_comparison': 'Validation MSE and DF feasibility after 10,000 optimization fits per run (MiSTP: 9,999).',
            'secondary_comparison': 'MiSTP at 5,000 steps uses 15,000 fits; this is not the equal-budget endpoint.',
            'reporting': 'Average evaluation seeds within each run, then mean and SE across run means. Reporting does not update iterates or select coefficients.',
            'limitations': 'Fresh randomness on the existing validation months is not an independent temporal test. The reserved test split is not accessed.',
        }
        df_runner.save_json(folder / 'protocol.json', protocol)
        df_runner.save_json(folder / 'budget.json', budget)
        df_runner.save_json(folder / 'status.json', {'state': 'prepared', 'pid': None})
    except Exception as error:
        df_runner.save_json(folder / 'status.json', {'state': 'preparation_failed', 'pid': None,
                                                  'error': str(error)})
        raise
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    folder = prepare(args.output)
    print(f'Prepared only; no training started: {folder}')
    print(f'Launch commands: {folder / "protocol.json"}')


if __name__ == '__main__':
    main()
