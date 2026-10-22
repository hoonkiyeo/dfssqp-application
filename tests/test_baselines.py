"""Independent baseline formulas, frozen evaluation and resumption checks."""

import csv
import fcntl
import json
import os
import signal

import numpy as np
import pytest

import config
from experiments import baselines, run
from models.baselines import aggregate_only, fixed_directions


def test_ols_fits_training_only_with_intercept():
    training = {'u': np.array([-2., -1., 0., 1., 2.]), 'y': np.array([-4., -1., 2., 5., 8.])}
    validation = {'u': np.array([3., 4.]), 'y': np.array([12., 12.])}
    result = aggregate_only(training, validation)
    np.testing.assert_allclose(result['coefficients'], [2., 3.], atol=2e-15)
    np.testing.assert_allclose(result['validation_predictions'], [11., 14.], atol=2e-15)
    assert result['mse'] == pytest.approx(2.5)
    other = aggregate_only(training, dict(validation, y=np.array([-900., 900.])))
    np.testing.assert_array_equal(result['coefficients'], other['coefficients'])
    np.testing.assert_array_equal(result['validation_predictions'], other['validation_predictions'])
    assert other['mse'] != result['mse']


def test_mean_and_pca_against_covariance_eigenvector():
    z = np.array([[10., -4., 2.], [17., -2., 5.], [13., -6., 1.], [15., -1., 3.]])
    original = z.copy()
    directions = fixed_directions({'z': z})
    np.testing.assert_array_equal(directions['mean'], np.ones(3)/np.sqrt(3))
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(z, rowvar=False))
    expected = eigenvectors[:, -1]
    if expected[np.argmax(abs(expected))] < 0:
        expected *= -1
    np.testing.assert_allclose(directions['pca'], expected, atol=1e-14)
    assert directions['pca'] @ np.cov(z, rowvar=False) @ directions['pca'] == pytest.approx(eigenvalues[-1])
    translated = fixed_directions({'z': z + np.array([30., 20., 10.])})
    np.testing.assert_array_equal(translated['pca'], directions['pca'])
    np.testing.assert_array_equal(z, original)


def test_baseline_invalid_data():
    with pytest.raises(ValueError, match='finite training'):
        fixed_directions({'z': np.array([[1., np.nan], [2., 3.]])})
    with pytest.raises(ValueError, match='u/y'):
        aggregate_only({'u': [1., 2.], 'y': [1.]}, {'u': [1.], 'y': [1.]})


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    # Real prepared data and model; a small inner budget keeps routine tests short.
    monkeypatch.setattr(config, 'training_steps', 4)
    monkeypatch.setattr(config, 'evaluation_seeds', [901, 902])
    folder = run.prepare(tmp_path / 'existing')
    history = folder / 'AI_123' / 'history.jsonl'
    history.parent.mkdir()
    history.write_text('existing DF-SSQP history must not change\n')
    (folder / 'summary.csv').write_text('existing DF-SSQP summary must not change\n')
    monkeypatch.setattr(run, 'step', lambda *a: pytest.fail('DF-SSQP must not run'))
    return folder


def read_rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def test_frozen_settings_pause_resume_reuses_fits_and_keeps_df_results(experiment, monkeypatch):
    before = {p.relative_to(experiment): p.read_bytes() for p in experiment.rglob('*') if p.is_file()}
    monkeypatch.setattr(config, 'training_steps', 17)
    monkeypatch.setattr(config, 'evaluation_seeds', [1000])
    original = run.evaluate
    calls = []

    def evaluate(w, seed, training, validation):
        assert config.training_steps == 4 and config.evaluation_seeds == [901, 902]
        calls.append((seed, w.copy()))
        mse = original(w, seed, training, validation)
        if len(calls) == 1:
            os.kill(os.getpid(), signal.SIGTERM)
        return mse

    monkeypatch.setattr(run, 'evaluate', evaluate)
    output = baselines.execute(experiment)
    assert len(calls) == 1
    rows = {r['model']: r for r in read_rows(output / 'summary.csv')}
    assert rows['mean']['panel_complete'] == 'False' and rows['mean']['evaluation_fits'] == '1'
    assert rows['pca']['mean_mse'] == ''
    baselines.execute(experiment)
    assert [s for s, _ in calls] == [901, 902, 901, 902]
    results_before = (output / 'fits.jsonl').read_bytes()
    baselines.execute(experiment)
    assert len(calls) == 4 and (output / 'fits.jsonl').read_bytes() == results_before
    assert config.training_steps == 17 and config.evaluation_seeds == [1000]
    assert all((experiment / p).read_bytes() == b for p, b in before.items())
    rows = {r['model']: r for r in read_rows(output / 'summary.csv')}
    assert all(r['panel_complete'] == 'True' for r in rows.values())
    assert rows['aggregate_only']['evaluation_fits'] == '0'
    assert rows['aggregate_only']['se_mse'] == ''
    per_seed = read_rows(output / 'evaluations.csv')
    for name in ('mean', 'pca'):
        values = [float(r['mse']) for r in per_seed if r['model'] == name]
        assert float(rows[name]['mean_mse']) == np.mean(values)
        assert float(rows[name]['se_mse']) == np.std(values, ddof=1)/np.sqrt(2)
    costs = json.loads((output / 'costs.json').read_text())
    assert costs['neural_fits']['done'] == 4 and costs['neural_fits']['start'] == 4


@pytest.mark.parametrize('change', ['model_hash', 'runtime', 'seeds', 'baseline_names'])
def test_incompatible_saved_inputs_rejected_before_training(experiment, monkeypatch, change):
    path = experiment / 'settings.json'
    saved = json.loads(path.read_text())
    if change == 'model_hash':
        saved['sha256']['models/objective.py'] = 'different'
    elif change == 'runtime':
        saved['runtime']['torch'] = 'different'
    elif change == 'seeds':
        saved['config']['evaluation_seeds'] = [2**32]
    else:
        saved['config']['baselines'] = ['unrecognized']
    path.write_text(json.dumps(saved))
    monkeypatch.setattr(run, 'evaluate', lambda *a: pytest.fail('Invalid experiment must not train'))
    with pytest.raises(ValueError):
        baselines.execute(experiment)


def test_baseline_provenance_and_writer_lock(experiment, monkeypatch):
    output = baselines.execute(experiment)
    monkeypatch.setattr(run, 'evaluate', lambda *a: pytest.fail('Completed/invalid baseline must not train'))
    with (output / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='already running'):
            baselines.execute(experiment)
    path = output / 'settings.json'
    manifest = json.loads(path.read_text())
    manifest['sha256']['models/baselines.py'] = 'different'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Baseline inputs changed'):
        baselines.execute(experiment)


def test_failed_fit_resumes_without_repeating_completed_fit(experiment, monkeypatch):
    original = run.evaluate
    calls = []

    def evaluate(w, seed, training, validation):
        calls.append(seed)
        if len(calls) == 2:
            raise RuntimeError('simulated interruption')
        return original(w, seed, training, validation)

    monkeypatch.setattr(run, 'evaluate', evaluate)
    with pytest.raises(RuntimeError, match='simulated'):
        baselines.execute(experiment)
    baselines.execute(experiment)
    assert calls == [901, 902, 902, 901, 902]
    events = run.read_journal(experiment / 'baselines/fits.jsonl')
    assert sum(e['event'] == 'done' for e in events) == 4
    assert sum(e['event'] == 'failed' for e in events) == 1
