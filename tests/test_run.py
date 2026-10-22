"""Runner checks: same mathematics through interruption, without expensive training."""

import fcntl
import json

import numpy as np
import pytest

import config
from experiments import run


@pytest.fixture
def small(monkeypatch, tmp_path):
    settings = {'modes': ['AI', 'ZI'], 'outer_seeds': [56100901], 'sqp_steps': 3,
                'evaluation_steps': [0, 1, 3], 'evaluation_seeds': [46100901, 46100902]}
    for key, value in settings.items():
        monkeypatch.setattr(config, key, value)
    metadata = (run.ROOT / 'data/processed/metadata.json').read_text()
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'fingerprints', lambda: {'inputs': 'fixed'})
    data = tmp_path / 'data/processed'
    data.mkdir(parents=True)
    (data / 'metadata.json').write_text(metadata)
    for name in ('training', 'validation'):
        np.savez(data / f'{name}.npz', z=np.zeros((64, 48)), u=np.zeros(64), y=np.zeros(64))
    calls = []

    def loss(w, seed, training, validation):
        calls.append((w.copy(), seed))
        return float(np.mean((w - np.linspace(-1, 1, len(w)))**2) + (seed % 17)*.001)

    monkeypatch.setattr(run, 'evaluate', loss)
    return tmp_path, calls, loss


def scientific_results(folder):
    result = {}
    for directory in sorted(folder.glob('*_*')):
        if directory.is_dir():
            result[directory.name] = {
                'history': run.read_journal(directory / 'history.jsonl'),
                'fits': {e['key']: (e['seed'], e['w_sha256'], e['mse'])
                         for e in run.read_journal(directory / 'fits.jsonl') if e['event'] == 'done'},
            }
    return result


def test_pause_resume_preserves_settings_and_all_results(small, monkeypatch):
    root, calls, _ = small
    continuous = run.prepare(root / 'continuous')
    run.execute(continuous)
    expected_calls = len(calls)
    paused = run.prepare(root / 'paused')
    run.execute(paused, stop_after=1)
    assert json.loads((paused / 'status.json').read_text())['state'] == 'paused'
    # A new experiment's config must not change the existing experiment on resume.
    monkeypatch.setattr(config, 'iota1', 10.)
    monkeypatch.setattr(config, 'sqp_steps', 99)
    run.execute(paused)
    assert len(calls) == 2*expected_calls == 48
    assert scientific_results(continuous) == scientific_results(paused)
    assert (continuous / 'summary.csv').read_text() == (paused / 'summary.csv').read_text()
    run.execute(paused)
    assert len(calls) == 48  # A completed resume is a numerical no-op.
    for mode in ('AI', 'ZI'):
        costs = json.loads((paused / f'{mode}_56100901/costs.json').read_text())
        assert costs['opt']['done'] == 6 and costs['eval']['done'] == 6
        assert costs['opt']['interrupted'] == costs['eval']['interrupted'] == 0


@pytest.mark.parametrize('interrupt_call', [2, 4])
def test_unfinished_panel_or_probe_pair_reuses_completed_fits(small, monkeypatch, interrupt_call):
    root, calls, loss = small
    monkeypatch.setattr(config, 'modes', ['ZI'])
    reference = run.prepare(root / 'reference')
    run.execute(reference)
    interrupted = run.prepare(root / 'interrupted')
    count = 0

    def interrupted_loss(*args):
        nonlocal count
        count += 1
        if count == interrupt_call:
            raise KeyboardInterrupt('simulated process termination before result save')
        return loss(*args)

    monkeypatch.setattr(run, 'evaluate', interrupted_loss)
    with pytest.raises(KeyboardInterrupt):
        run.execute(interrupted)
    monkeypatch.setattr(run, 'evaluate', loss)
    run.execute(interrupted)
    assert scientific_results(reference) == scientific_results(interrupted)
    events = run.read_journal(interrupted / 'ZI_56100901/fits.jsonl')
    assert sum(e['event'] == 'done' for e in events) == 12
    assert sum(e['event'] == 'start' for e in events) == 13
    assert sum(e['event'] == 'interrupted' for e in events) == 1
    assert len(calls) == 24  # Durable earlier losses were not recomputed.


def test_partial_final_record_is_preserved_and_completed_fits_replayed(small):
    root, calls, _ = small
    folder = run.prepare(root / 'partial')
    run.execute(folder, stop_after=1)
    path = folder / 'AI_56100901/history.jsonl'
    # Simulate a killed state commit after the completed pair reached disk.
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[0] + lines[1][:40])
    run.execute(folder)
    assert len(calls) == 24
    assert len(list(path.parent.glob('history.jsonl.partial-*'))) == 1
    assert len(run.read_journal(path)) == 4


def test_lock_and_changed_inputs_prevent_work(small, monkeypatch):
    root, calls, _ = small
    folder = run.prepare(root / 'locked')
    with (folder / 'runner.lock').open('a') as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='already'):
            run.execute(folder)
    monkeypatch.setattr(run, 'fingerprints', lambda: {'inputs': 'changed'})
    with pytest.raises(ValueError, match='changed'):
        run.execute(folder)
    assert not calls


def test_completed_corruption_is_not_discarded(small):
    root, _, _ = small
    path = root / 'broken.jsonl'
    path.write_bytes(b'{"ok":1}\nBROKEN\n')
    with pytest.raises(json.JSONDecodeError):
        run.read_journal(path)
    assert path.read_bytes() == b'{"ok":1}\nBROKEN\n'


def test_failed_fit_stays_visible_and_can_be_retried(small, monkeypatch):
    root, _, loss = small
    folder = run.prepare(root / 'failed')
    monkeypatch.setattr(run, 'evaluate', lambda *args: float('nan'))
    with pytest.raises(FloatingPointError):
        run.execute(folder)
    assert json.loads((folder / 'status.json').read_text())['state'] == 'failed'
    monkeypatch.setattr(run, 'evaluate', loss)
    run.execute(folder)
    costs = json.loads((folder / 'AI_56100901/costs.json').read_text())
    assert costs['eval']['failed'] == 1 and costs['eval']['done'] == 6
    assert costs['eval']['interrupted'] == 0


def test_pause_file_and_seed_separation(small, monkeypatch):
    root, calls, loss = small
    folder = run.prepare(root / 'pause_file')

    def pausing_loss(*args):
        value = loss(*args)
        (folder / 'PAUSE').touch()
        return value

    monkeypatch.setattr(run, 'evaluate', pausing_loss)
    run.execute(folder)
    assert len(calls) == 1 and not (folder / 'PAUSE').exists()
    monkeypatch.setattr(run, 'evaluate', loss)
    run.execute(folder)
    assert len(calls) == 24
    for directory in folder.glob('*_*'):
        if directory.is_dir():
            for e in run.read_journal(directory / 'fits.jsonl'):
                assert (e['seed'] < 2**32) == e['key'].startswith('eval:')
    monkeypatch.setattr(config, 'evaluation_seeds', [2**32])
    with pytest.raises(ValueError, match='below'):
        run.prepare(root / 'invalid')


def test_data_period_changes_require_preparation(small, monkeypatch):
    root, calls, _ = small
    monkeypatch.setattr(config, 'training_months', ('1970-01', '2003-09'))
    with pytest.raises(ValueError, match='data.prepare'):
        run.prepare(root / 'wrong_period')
    assert not calls and not (root / 'wrong_period').exists()


def test_reporting_panel_does_not_change_optimizer(small, monkeypatch):
    root, _, _ = small
    dense = run.prepare(root / 'dense')
    run.execute(dense)
    monkeypatch.setattr(config, 'evaluation_steps', [0, 3])
    sparse = run.prepare(root / 'sparse')
    run.execute(sparse)
    for name, reference in scientific_results(dense).items():
        other = scientific_results(sparse)[name]
        assert reference['history'] == other['history']
        assert {k: v for k, v in reference['fits'].items() if k.startswith('opt:')} == {
            k: v for k, v in other['fits'].items() if k.startswith('opt:')}
    ai = scientific_results(dense)['AI_56100901']['history']
    zi = scientific_results(dense)['ZI_56100901']['history']
    assert [r['metrics']['seed'] for r in ai[1:]] == [r['metrics']['seed'] for r in zi[1:]]


def test_bad_history_index_stops_instead_of_continuing(small):
    root, calls, _ = small
    folder = run.prepare(root / 'bad_index')
    run.execute(folder, stop_after=1)
    path = folder / 'AI_56100901/history.jsonl'
    records = run.read_journal(path)
    records[-1]['state']['k'] = 99
    path.write_text(''.join(json.dumps(row, default=run.json_value)+'\n' for row in records))
    count = len(calls)
    with pytest.raises(ValueError, match='Non-contiguous'):
        run.execute(folder)
    assert len(calls) == count
