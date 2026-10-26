"""Extend saved trajectories without training neural models or launching workers."""

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import config
from experiments import extend_comparison, prepare_comparison
from experiments import run as df_runner
from experiments import mistp as mistp_runner
from experiments import zo_rsgd as zo_runner
from experiments.pilot import settings


RUNNERS = {'df_ai': df_runner, 'df_zi': df_runner,
           'zo_rsgd': zo_runner, 'mistp': mistp_runner}
RUN_SEEDS = [67_102_501, 67_102_502]
EVAL_SEEDS = [48_102_501, 48_102_502]


def read_json(path):
    return json.loads(path.read_text())


def file_hashes(folder):
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.rglob('*') if path.is_file()}


def scientific_results(folder):
    """Discard fit timestamps/IDs while retaining every state, loss and seed."""
    result = {}
    for path in sorted(folder.rglob('history.jsonl')):
        result[str(path.relative_to(folder))] = df_runner.read_journal(path)
    for path in sorted(folder.rglob('fits.jsonl')):
        result[str(path.relative_to(folder))] = {
            event['key']: (event['seed'], event['w_sha256'], event['mse'])
            for event in df_runner.read_journal(path) if event['event'] == 'done'}
    return result


@pytest.fixture
def completed_source(monkeypatch, tmp_path):
    """Four native completed campaigns with a cheap scalar objective."""
    real_root = df_runner.ROOT
    for name in ('algorithms', 'models', 'experiments'):
        shutil.copytree(real_root / name, tmp_path / name,
                        ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('config.py', 'requirements.txt', 'environment.yml'):
        shutil.copy2(real_root / name, tmp_path / name)
    data = tmp_path / 'data/processed'
    data.mkdir(parents=True)
    shutil.copy2(real_root / 'data/processed/metadata.json', data / 'metadata.json')
    for name in ('training', 'validation'):
        np.savez(data / f'{name}.npz', z=np.zeros((64, 48)),
                 u=np.zeros(64), y=np.zeros(64))
    for module in (extend_comparison, prepare_comparison,
                   df_runner, zo_runner, mistp_runner):
        monkeypatch.setattr(module, 'ROOT', tmp_path, raising=False)
    # The global configuration is mutable in native resume; restore every entry.
    original = {key: copy.deepcopy(value) for key, value in vars(config).items()
                if not key.startswith('_') and not callable(value)}
    for key, value in original.items():
        monkeypatch.setattr(config, key, value)
    values = {
        'modes': ('AI', 'ZI'), 'sqp_steps': 3, 'outer_seeds': RUN_SEEDS,
        'evaluation_seeds': EVAL_SEEDS, 'evaluation_steps': [0, 3],
        'zo_rsgd_steps': 3, 'zo_rsgd_run_seeds': RUN_SEEDS,
        'zo_rsgd_evaluation_seeds': EVAL_SEEDS,
        'zo_rsgd_evaluation_steps': [0, 3],
        'zo_rsgd_step_coefficient': 1., 'zo_rsgd_mu': .01,
        'zo_rsgd_m': 1, 'mistp_alphas': (.01,), 'mistp_steps': 3,
        'mistp_run_seeds': RUN_SEEDS, 'mistp_evaluation_seeds': EVAL_SEEDS,
        'mistp_evaluation_steps': [0, 2, 3],
    }
    for key, value in values.items():
        monkeypatch.setattr(config, key, copy.deepcopy(value))
    calls = []
    target = np.linspace(-1., 1., 48)

    def cheap_loss(w, seed, training, validation):
        calls.append((np.asarray(w).copy(), int(seed)))
        return float(np.mean((np.asarray(w) - target)**2) + (seed % 17)*.001)

    monkeypatch.setattr(df_runner, 'evaluate', cheap_loss)
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(stdout='fixture', stderr='', returncode=0))
    source = prepare_comparison.prepare(tmp_path / 'source')
    for name, runner in RUNNERS.items():
        runner.execute(source / name)
    df_runner.save_json(source / 'status.json', {'state': 'completed', 'pid': None})
    # Preserve the exact source snapshot needed to audit later code amendments.
    snapshot = source / 'source_snapshot'
    snapshot.mkdir()
    names = set()
    for name in RUNNERS:
        names.update(read_json(source / name / 'settings.json')['sha256'])
    names.update(('config.py', 'experiments/prepare_comparison.py'))
    manifest = {}
    for name in names:
        path = tmp_path / name
        if not path.is_file() or name.startswith('data/'):
            continue
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    df_runner.save_json(snapshot / 'manifest.json', manifest)
    return SimpleNamespace(root=tmp_path, source=source, calls=calls,
                           cheap_loss=cheap_loss)


def test_preparation_copies_three_prefixes_and_keeps_zo_fresh(completed_source, monkeypatch):
    case = completed_source
    before, call_count = file_hashes(case.source), len(case.calls)

    def forbidden(*args, **kwargs):
        pytest.fail('Preparation must not execute a runner, fit a model or launch a worker.')

    for runner in set(RUNNERS.values()):
        monkeypatch.setattr(runner, 'execute', forbidden)
    monkeypatch.setattr(df_runner, 'evaluate', forbidden)
    # Native preparation can inspect Git, but it must not spawn an experiment.
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(stdout='fixture', stderr='', returncode=0))
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    target = extend_comparison.prepare(case.source, case.root / 'extended',
                                       steps=6, report_every=2)
    assert target == case.root / 'extended'
    assert file_hashes(case.source) == before
    assert len(case.calls) == call_count
    assert not list((target / 'zo_rsgd').rglob('history.jsonl'))
    assert not list((target / 'zo_rsgd').rglob('fits.jsonl'))
    for name in ('df_ai', 'df_zi', 'mistp'):
        original = case.source / name
        copied = target / name
        for source_file in list(original.rglob('history.jsonl')) + list(original.rglob('fits.jsonl')):
            relative = source_file.relative_to(original)
            destination = copied / relative
            assert destination.read_bytes() == source_file.read_bytes()
            assert destination.stat().st_ino != source_file.stat().st_ino
    settings_zo = read_json(target / 'zo_rsgd/settings.json')
    assert settings_zo['parameters']['eta'] == pytest.approx(1 / np.sqrt(6))
    assert settings_zo['parameters']['eta'] != read_json(
        case.source / 'zo_rsgd/settings.json')['parameters']['eta']
    assert settings_zo['config']['zo_rsgd_steps'] == 6
    assert settings_zo['config']['zo_rsgd_step_coefficient'] == 1.
    for name in RUNNERS:
        saved = read_json(target / name / 'settings.json')['config']
        prefix = '' if name.startswith('df_') else name + '_'
        horizon = 'sqp_steps' if not prefix else prefix + 'steps'
        assert saved[horizon] == 6
        assert saved[prefix + 'evaluation_steps'] == (
            [0, 2, 3, 4, 6] if name == 'mistp' else [0, 3, 4, 6])
        assert saved[prefix + ('outer_seeds' if not prefix else 'run_seeds')] == RUN_SEEDS
        assert saved[prefix + 'evaluation_seeds'] == EVAL_SEEDS


def test_extension_budgets_distinguish_reuse_from_new_work(completed_source):
    case = completed_source
    target = extend_comparison.prepare(case.source, case.root / 'budgeted',
                                       steps=6, report_every=2)
    expected = {'df_ai': (24, 16, 40), 'df_zi': (24, 16, 40),
                'zo_rsgd': (24, 14, 38), 'mistp': (36, 18, 54)}
    for name, (optimization, evaluation, total) in expected.items():
        assert read_json(target / name / 'budget.json') == {
            'optimization_fits': optimization, 'evaluation_fits': evaluation,
            'total_fits': total}
    protocol = read_json(target / 'protocol.json')
    entries = {entry['name']: entry for entry in protocol['campaigns']}
    for name in RUNNERS:
        inherited = entries[name]['inherited_budget']
        added = entries[name]['additional_budget']
        cumulative = read_json(target / name / 'budget.json')
        for key in ('optimization_fits', 'evaluation_fits', 'total_fits'):
            assert inherited[key] + added[key] == cumulative[key]
        if name == 'zo_rsgd':
            assert inherited['total_fits'] == 0
            assert added == cumulative
        else:
            assert inherited == read_json(case.source / name / 'budget.json')
    assert sum(entry['additional_budget']['total_fits'] for entry in entries.values()) == 104


@pytest.mark.parametrize('name', RUNNERS)
def test_native_resume_matches_continuous_run_and_is_idempotent(completed_source, name):
    case = completed_source
    before = file_hashes(case.source)
    target = extend_comparison.prepare(case.source, case.root / 'extended',
                                       steps=6, report_every=2)
    runner = RUNNERS[name]
    child = target / name
    saved = read_json(child / 'settings.json')
    reference = case.root / f'reference_{name}'
    with settings(**saved['config']):
        runner.prepare(reference)
    # ZO is a fresh N=6 experiment; its native eta must be recomputed from N=6.
    if name == 'zo_rsgd':
        assert read_json(reference / 'settings.json')['parameters']['eta'] == pytest.approx(1 / np.sqrt(6))
    runner.execute(reference)
    expected = scientific_results(reference)
    calls_before_resume = len(case.calls)
    runner.execute(child, stop_after=1)
    assert read_json(child / 'status.json')['state'] == 'paused'
    runner.execute(child)
    assert scientific_results(child) == expected
    new_calls = len(case.calls) - calls_before_resume
    assert new_calls == {'df_ai': 20, 'df_zi': 20, 'zo_rsgd': 38, 'mistp': 26}[name]
    count = len(case.calls)
    runner.execute(child)
    assert len(case.calls) == count
    assert scientific_results(child) == expected
    assert file_hashes(case.source) == before
    assert read_json(child / 'status.json')['state'] == 'completed'


@pytest.mark.parametrize('steps', [0, 2, 3, 3.5, True])
def test_non_increasing_or_invalid_horizon_is_rejected(completed_source, steps):
    case = completed_source
    before = file_hashes(case.source)
    target = case.root / 'invalid'
    with pytest.raises((ValueError, TypeError)):
        extend_comparison.prepare(case.source, target, steps=steps, report_every=2)
    assert file_hashes(case.source) == before
    assert not target.exists()


def test_existing_target_is_never_overwritten(completed_source):
    case = completed_source
    target = case.root / 'existing'
    target.mkdir()
    marker = target / 'keep.txt'
    marker.write_text('user content')
    with pytest.raises((ValueError, FileExistsError)):
        extend_comparison.prepare(case.source, target, steps=6, report_every=2)
    assert marker.read_text() == 'user content'


@pytest.mark.parametrize('problem', ['missing_campaign', 'incomplete_history',
                                     'changed_data', 'changed_runtime',
                                     'incomplete_panel', 'method_seed_mismatch'])
def test_invalid_source_is_rejected_before_output(completed_source, problem):
    case = completed_source
    if problem == 'missing_campaign':
        shutil.rmtree(case.source / 'df_zi')
    elif problem == 'incomplete_history':
        path = next((case.source / 'df_ai').rglob('history.jsonl'))
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b''.join(lines[:-1]))
    elif problem == 'changed_data':
        path = case.root / 'data/processed/training.npz'
        path.write_bytes(path.read_bytes() + b'changed input')
    elif problem == 'incomplete_panel':
        path = next((case.source / 'df_ai').rglob('evaluations.csv'))
        path.write_text(path.read_text().replace('True', 'False', 1))
    elif problem == 'method_seed_mismatch':
        path = case.source / 'zo_rsgd/settings.json'
        saved = read_json(path)
        saved['config']['zo_rsgd_run_seeds'] = [999]
        df_runner.save_json(path, saved)
    else:
        path = case.source / 'df_ai/settings.json'
        saved = read_json(path)
        saved['runtime']['numpy'] = 'different-runtime'
        df_runner.save_json(path, saved)
    target = case.root / 'invalid'
    before = file_hashes(case.source)
    with pytest.raises((ValueError, FileNotFoundError)):
        extend_comparison.prepare(case.source, target, steps=6, report_every=2)
    assert not target.exists()
    assert file_hashes(case.source) == before


def test_ten_thousand_target_uses_saved_protocol_and_fresh_zo_step(completed_source, monkeypatch):
    case = completed_source
    before_calls = len(case.calls)
    monkeypatch.setattr(config, 'iota1', 999.)
    monkeypatch.setattr(config, 'mistp_alphas', (9.,))
    monkeypatch.setattr(config, 'zo_rsgd_step_coefficient', 9.)
    monkeypatch.setattr(config, 'outer_seeds', [999])
    monkeypatch.setattr(config, 'evaluation_seeds', [998])
    target = extend_comparison.prepare(case.source, case.root / 'ten_thousand',
                                       steps=10_000, report_every=500)
    assert len(case.calls) == before_calls
    for name in RUNNERS:
        frozen = read_json(target / name / 'settings.json')
        original = read_json(case.source / name / 'settings.json')
        for key in ('iota1', 'iota2', 'iota3', 'p1', 'p2', 'p3',
                    'initialization_seed', 'training_steps', 'learning_rate',
                    'hidden_width', 'batch_size', 'standardize'):
            assert frozen['config'][key] == original['config'][key]
    zo = read_json(target / 'zo_rsgd/settings.json')
    assert zo['config']['zo_rsgd_step_coefficient'] == 1.
    assert zo['parameters'] == {'eta': .01, 'mu': .01, 'm': 1}
    mistp = read_json(target / 'mistp/settings.json')['config']
    assert mistp['mistp_alphas'] == [.01]
    assert 6666 in mistp['mistp_evaluation_steps']
    assert mistp['mistp_evaluation_steps'][-1] == 10_000
    assert not list((target / 'zo_rsgd').rglob('history.jsonl'))


def test_comment_amendment_records_old_and_current_source_fingerprints(completed_source):
    case = completed_source
    file = case.root / 'algorithms/mistp.py'
    before = file_hashes(case.source)
    old = read_json(case.source / 'mistp/settings.json')['sha256']['algorithms/mistp.py']
    file.write_text(file.read_text() + '\n# Clarify the existing update without changing its calculation.\n')
    current = hashlib.sha256(file.read_bytes()).hexdigest()
    assert current != old
    target = extend_comparison.prepare(case.source, case.root / 'amended',
                                       steps=6, report_every=2)
    protocol = read_json(target / 'protocol.json')
    entry = next(item for item in protocol['campaigns'] if item['name'] == 'mistp')
    assert entry['source_changes']['algorithms/mistp.py'] == {
        'source_sha256': old, 'prepared_sha256': current}
    assert read_json(target / 'mistp/settings.json')['sha256']['algorithms/mistp.py'] == current
    assert file_hashes(case.source) == before
