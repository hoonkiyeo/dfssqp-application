"""MiSTP runner checks: cheap deterministic losses, no training experiment."""
import csv
import fcntl
import json
from pathlib import Path

import numpy as np
import pytest

import config
from experiments import mistp as runner
from experiments import run as storage


@pytest.fixture
def small(monkeypatch, tmp_path):
    values = {
        'mistp_alphas': (0.01, 0.1, 1.0),
        'mistp_steps': 3,
        'mistp_run_seeds': [71, 72, 73],
        'mistp_evaluation_seeds': [101, 102],
        'mistp_evaluation_steps': [0, 1, 3],
        'mistp_experiment_name': 'pilot_test',
        'evaluation_seeds': [201, 202],
    }
    for key, value in values.items():
        monkeypatch.setattr(config, key, value)
    metadata = (runner.ROOT / 'data/processed/metadata.json').read_text()
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(storage, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'fingerprints', lambda: {'inputs': 'fixed'})
    data = tmp_path / 'data/processed'
    data.mkdir(parents=True)
    (data / 'metadata.json').write_text(metadata)
    for name in ('training', 'validation'):
        np.savez(data / f'{name}.npz', z=np.zeros((64, 48)), u=np.zeros(64), y=np.zeros(64))
    calls = []
    target = np.linspace(-1, 1, 48)

    def loss(w, seed, training, validation):
        calls.append((np.asarray(w).copy(), int(seed)))
        return float(np.mean((np.asarray(w)-target)**2) + (seed % 17)*0.001)

    monkeypatch.setattr(storage, 'evaluate', loss)
    return tmp_path, calls, loss


def scientific_results(folder):
    result = {}
    for directory in sorted(folder.glob('alpha_*/run_*')):
        result[str(directory.relative_to(folder))] = {
            'history': storage.read_journal(directory / 'history.jsonl'),
            'fits': {e['key']: (e['seed'], e['w_sha256'], e['mse'])
                     for e in storage.read_journal(directory / 'fits.jsonl') if e['event'] == 'done'},
        }
    initial = folder / 'initial/fits.jsonl'
    result['initial'] = {
        e['key']: (e['seed'], e['w_sha256'], e['mse'])
        for e in storage.read_journal(initial) if e['event'] == 'done'
    }
    return result


def done_events(folder):
    return [e for path in folder.rglob('fits.jsonl')
            for e in storage.read_journal(path) if e['event'] == 'done']


def test_frozen_config_resume_and_full_cost(small, monkeypatch):
    root, calls, _ = small
    continuous = runner.prepare(root / 'continuous')
    runner.execute(continuous)
    expected = scientific_results(continuous)
    # 9 cases * 3 steps * 3 optimization fits, shared initial 2, and 9 * 2 * 2 report fits.
    assert len(calls) == 119
    budget = json.loads((continuous / 'budget.json').read_text())
    assert budget['optimization_fits'] == 81
    assert budget['evaluation_fits'] == 38
    assert budget['total_fits'] == 119
    costs = json.loads((continuous / 'costs.json').read_text())
    assert costs['opt']['done'] == 81 and costs['eval']['done'] == 38
    assert costs['total_done'] == 119
    paused = runner.prepare(root / 'paused')
    runner.execute(paused, stop_after=1)
    assert json.loads((paused / 'status.json').read_text())['state'] == 'paused'
    selection = json.loads((paused / 'selection.json').read_text())
    assert selection['complete'] is False and selection['alpha_with_lowest_mean'] is None
    # Resumption restores the saved scientific settings, not the edited config.
    monkeypatch.setattr(config, 'mistp_steps', 99)
    monkeypatch.setattr(config, 'mistp_alphas', (99.,))
    monkeypatch.setattr(config, 'mistp_run_seeds', [999])
    monkeypatch.setattr(config, 'mistp_evaluation_seeds', [998])
    runner.execute(paused)
    assert len(calls) == 238
    assert scientific_results(paused) == expected
    assert (paused / 'summary.csv').read_text() == (continuous / 'summary.csv').read_text()
    runner.execute(paused)
    assert len(calls) == 238


def test_initial_cache_seeds_and_alpha_sharing(small):
    root, calls, _ = small
    folder = runner.prepare(root / 'sharing')
    runner.execute(folder)
    events = done_events(folder)
    assert len([e for e in events if e['seed'] < 2**32]) == 38
    assert len(storage.read_journal(folder / 'initial/fits.jsonl')) == 4  # 2 start + 2 done
    histories = {}
    for directory in folder.glob('alpha_*/run_*'):
        alpha, seed = float(directory.parent.name.removeprefix('alpha_')), int(directory.name.removeprefix('run_'))
        histories[alpha, seed] = storage.read_journal(directory / 'history.jsonl')
    initial_w = None
    for records in histories.values():
        assert [r['state']['k'] for r in records] == [0, 1, 2, 3]
        w0 = records[0]['state']['w']
        assert initial_w is None or initial_w == w0
        initial_w = w0
    # The same iteration index/outer seed has the same direction and training seed across alpha.
    # Full direction is inferred exactly from recorded candidate fits using core replay below.
    from algorithms.mistp import step
    for seed in config.mistp_run_seeds:
        for k in range(config.mistp_steps):
            traces = [step(histories[alpha, seed][k]['state'], seed, alpha, lambda w, s: 0.)[1]
                      for alpha in config.mistp_alphas]
            for trace in traces[1:]:
                np.testing.assert_array_equal(trace['s'], traces[0]['s'])
                assert trace['seed'] == traces[0]['seed'] >= 2**32
    # Every iteration retrains all three candidates with one shared fresh training seed.
    for directory in folder.glob('alpha_*/run_*'):
        completed = {e['key']: e for e in storage.read_journal(directory / 'fits.jsonl')
                     if e['event'] == 'done' and e['key'].startswith('opt:')}
        assert len(completed) == 3 * config.mistp_steps
        seeds = []
        for k in range(config.mistp_steps):
            fits = [completed[f'opt:{k}:{candidate}']
                    for candidate in ('current', 'plus', 'minus')]
            assert len({fit['seed'] for fit in fits}) == 1
            seeds.append(fits[0]['seed'])
        assert len(set(seeds)) == config.mistp_steps
    assert not set(config.mistp_evaluation_seeds) & set(config.evaluation_seeds)
    assert all(e['seed'] >= 2**32 or e['seed'] in config.mistp_evaluation_seeds for e in events)
    with (folder / 'summary.csv').open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 9
    assert all(int(row['runs']) == 3 for row in rows)
    assert all(float(row['mean_feasibility']) < 1e-14 for row in rows)
    for row in rows:
        if int(row['k']) == 0:
            assert float(row['se_mse']) == 0


@pytest.mark.parametrize('interrupt_call', [2, 3, 4, 5, 6, 7])
def test_interrupted_initial_current_plus_minus_or_reporting_reuses_done_fits(small, monkeypatch, interrupt_call):
    root, calls, loss = small
    reference = runner.prepare(root / 'reference')
    runner.execute(reference)
    interrupted = runner.prepare(root / 'interrupted')
    count = 0

    def fail(*args):
        nonlocal count
        count += 1
        if count == interrupt_call:
            raise KeyboardInterrupt('simulated hard stop')
        return loss(*args)

    monkeypatch.setattr(storage, 'evaluate', fail)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(interrupted)
    monkeypatch.setattr(storage, 'evaluate', loss)
    runner.execute(interrupted)
    assert scientific_results(reference) == scientific_results(interrupted)
    assert len(calls) == 238
    events = [e for path in interrupted.rglob('fits.jsonl') for e in storage.read_journal(path)]
    assert sum(e['event'] == 'done' for e in events) == 119
    assert sum(e['event'] == 'interrupted' for e in events) == 1
    assert sum(e['event'] == 'start' for e in events) == 120


@pytest.mark.parametrize('alphas', [(0.01, 0.1, 1.0), (0.1,)])
def test_reporting_panel_does_not_change_search(small, monkeypatch, alphas):
    root, _, _ = small
    monkeypatch.setattr(config, 'mistp_alphas', alphas)
    dense = runner.prepare(root / 'dense')
    runner.execute(dense)
    monkeypatch.setattr(config, 'mistp_evaluation_steps', [0, 3])
    monkeypatch.setattr(config, 'mistp_evaluation_seeds', [103])
    sparse = runner.prepare(root / 'sparse')
    runner.execute(sparse)
    a, b = scientific_results(dense), scientific_results(sparse)
    for name in a:
        if name == 'initial':
            continue
        assert a[name]['history'] == b[name]['history']
        assert {k: v for k, v in a[name]['fits'].items() if k.startswith('opt:')} == {
            k: v for k, v in b[name]['fits'].items() if k.startswith('opt:')}


def test_locks_changed_inputs_and_failed_fit(small, monkeypatch):
    root, calls, loss = small
    folder = runner.prepare(root / 'protected')
    with (folder / 'runner.lock').open('a') as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='already'):
            runner.execute(folder)
    monkeypatch.setattr(runner, 'fingerprints', lambda: {'inputs': 'changed'})
    with pytest.raises(ValueError, match='changed'):
        runner.execute(folder)
    assert not calls
    monkeypatch.setattr(runner, 'fingerprints', lambda: {'inputs': 'fixed'})
    monkeypatch.setattr(storage, 'evaluate', lambda *args: float('nan'))
    with pytest.raises(FloatingPointError):
        runner.execute(folder)
    assert json.loads((folder / 'status.json').read_text())['state'] == 'failed'
    monkeypatch.setattr(storage, 'evaluate', loss)
    runner.execute(folder)
    assert len(calls) == 119
    costs = json.loads((folder / 'costs.json').read_text())
    assert costs['eval']['failed'] == 1


def test_pause_file_partial_history_and_final_selection(small, monkeypatch):
    root, calls, loss = small
    reference = runner.prepare(root / 'reference')
    runner.execute(reference)
    folder = runner.prepare(root / 'partial')
    runner.execute(folder, stop_after=1)
    path = next(folder.glob('alpha_*/run_*/history.jsonl'))
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[0] + lines[1][:40])
    runner.execute(folder)
    assert scientific_results(folder) == scientific_results(reference)
    assert len(calls) == 238
    assert len(list(path.parent.glob('history.jsonl.partial-*'))) == 1
    selection = json.loads((folder / 'selection.json').read_text())
    with (folder / 'summary.csv').open() as file:
        finals = [row for row in csv.DictReader(file) if int(row['k']) == 3]
    expected = min(finals, key=lambda row: float(row['mean_mse']))
    assert float(selection['alpha_with_lowest_mean']) == float(expected['alpha'])


@pytest.mark.parametrize('name,value', [
    ('mistp_evaluation_seeds', [201]),
    ('mistp_evaluation_seeds', [2**32]),
    ('mistp_evaluation_seeds', [101, 101]),
    ('mistp_run_seeds', [71, 71]),
    ('mistp_alphas', [0.]),
    ('mistp_alphas', [np.nan]),
    ('mistp_alphas', [0.1, 0.1]),
    ('mistp_steps', 0),
    ('mistp_steps', None),
    ('mistp_run_seeds', []),
    ('mistp_evaluation_seeds', []),
    ('mistp_evaluation_steps', []),
    ('mistp_evaluation_steps', [0, 2]),
])
def test_bad_protocol_is_rejected_without_work(small, monkeypatch, name, value):
    root, calls, _ = small
    monkeypatch.setattr(config, name, value)
    with pytest.raises(ValueError):
        runner.prepare(root / 'bad')
    assert not calls and not (root / 'bad').exists()


def test_pause_request_during_initial_panel_reuses_result(small, monkeypatch):
    root, calls, loss = small
    folder = runner.prepare(root / 'pause_file')

    def pausing_loss(*args):
        value = loss(*args)
        (folder / 'PAUSE').touch()
        return value

    monkeypatch.setattr(storage, 'evaluate', pausing_loss)
    runner.execute(folder)
    assert len(calls) == 1
    assert json.loads((folder / 'status.json').read_text())['state'] == 'paused'
    assert not (folder / 'PAUSE').exists()
    monkeypatch.setattr(storage, 'evaluate', loss)
    runner.execute(folder)
    assert len(calls) == 119
    assert json.loads((folder / 'status.json').read_text())['state'] == 'completed'


def test_fixed_alpha_production_panel_budget_and_resume(small, monkeypatch):
    root, calls, _ = small
    monkeypatch.setattr(config, 'mistp_alphas', (0.1,))
    monkeypatch.setattr(config, 'mistp_evaluation_seeds', list(config.evaluation_seeds))
    continuous = runner.prepare(root / 'fixed_continuous')
    assert not calls
    runner.execute(continuous)
    expected = scientific_results(continuous)
    # 3 runs * 3 steps * 3 fits + 2 initial fits + 3 runs * 2 checkpoints * 2 fits.
    assert len(calls) == 41
    assert json.loads((continuous / 'budget.json').read_text()) == {
        'optimization_fits': 27, 'evaluation_fits': 14, 'total_fits': 41}
    selection = json.loads((continuous / 'selection.json').read_text())
    assert selection['mode'] == 'fixed_alpha' and selection['complete']
    assert selection['fixed_alpha'] == 0.1 and selection['alpha_with_lowest_mean'] is None
    assert len(selection['final_results']) == 1
    assert selection['final_results'][0]['runs'] == 3
    assert all(seed >= 2**32 or seed in config.evaluation_seeds for _, seed in calls)

    paused = runner.prepare(root / 'fixed_paused')
    runner.execute(paused, stop_after=1)
    selection = json.loads((paused / 'selection.json').read_text())
    assert selection['mode'] == 'fixed_alpha' and not selection['complete']
    assert selection['fixed_alpha'] == 0.1 and selection['alpha_with_lowest_mean'] is None
    # Saved fixed-alpha settings must win over a later multi-alpha configuration.
    monkeypatch.setattr(config, 'mistp_alphas', (9.0, 99.0))
    monkeypatch.setattr(config, 'mistp_steps', 99)
    monkeypatch.setattr(config, 'mistp_evaluation_seeds', [999])
    runner.execute(paused)
    assert scientific_results(paused) == expected
    assert (paused / 'summary.csv').read_text() == (continuous / 'summary.csv').read_text()
    assert (paused / 'selection.json').read_text() == (continuous / 'selection.json').read_text()
    assert len(calls) == 82
    runner.execute(paused)
    assert len(calls) == 82


def test_old_random_search_source_identity_is_rejected_without_work(small):
    root, calls, _ = small
    folder = runner.prepare(root / 'wrong_method')
    path = folder / 'settings.json'
    saved = json.loads(path.read_text())
    saved['sha256'] = {'algorithms/random_search.py': 'old-two-point-source'}
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='changed'):
        runner.execute(folder)
    assert not calls
    assert not list(folder.rglob('fits.jsonl'))


def test_unconfigured_pilot_cannot_prepare_or_fit(small, monkeypatch):
    root, calls, _ = small
    monkeypatch.setattr(config, 'mistp_alphas', (1.0, 0.1, 0.01))
    monkeypatch.setattr(config, 'mistp_steps', None)
    monkeypatch.setattr(config, 'mistp_run_seeds', [])
    monkeypatch.setattr(config, 'mistp_evaluation_seeds', [])
    monkeypatch.setattr(config, 'mistp_evaluation_steps', [])
    with pytest.raises(ValueError):
        runner.prepare(root / 'unconfigured')
    assert not calls
    assert not (root / 'unconfigured').exists()
