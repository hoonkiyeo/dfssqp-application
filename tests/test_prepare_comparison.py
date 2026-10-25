"""Freeze the selected main comparison; never train or launch a worker."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import config
from experiments import prepare_comparison


RUN_SEEDS = list(range(67_102_501, 67_102_511))
PANEL = list(range(48_102_501, 48_102_521))
POINTS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500,
          1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
MISTP_POINTS = sorted(POINTS + [3333])


def public_config():
    return {key: copy.deepcopy(value) for key, value in vars(config).items()
            if not key.startswith('_') and not callable(value)}


def read_json(path):
    return json.loads(path.read_text())


@pytest.fixture
def prepared_context(monkeypatch, tmp_path):
    for key, value in public_config().items():
        monkeypatch.setattr(config, key, copy.deepcopy(value))
    expected = {
        'modes': ('AI', 'ZI'), 'iota1': 10., 'sqp_steps': 5000,
        'outer_seeds': RUN_SEEDS, 'evaluation_seeds': PANEL,
        'evaluation_steps': POINTS,
        'zo_rsgd_step_coefficient': 1., 'zo_rsgd_mu': .01, 'zo_rsgd_m': 1,
        'zo_rsgd_steps': 5000, 'zo_rsgd_run_seeds': RUN_SEEDS,
        'zo_rsgd_evaluation_seeds': PANEL, 'zo_rsgd_evaluation_steps': POINTS,
        'mistp_alphas': (.01,), 'mistp_steps': 5000,
        'mistp_run_seeds': RUN_SEEDS, 'mistp_evaluation_seeds': PANEL,
        'mistp_evaluation_steps': MISTP_POINTS,
    }
    for key, value in expected.items():
        monkeypatch.setattr(config, key, copy.deepcopy(value))

    modules = (prepare_comparison, prepare_comparison.df_runner,
               prepare_comparison.zo_runner, prepare_comparison.mistp_runner)
    metadata = (prepare_comparison.df_runner.ROOT / 'data/processed/metadata.json').read_text()
    for module in modules:
        monkeypatch.setattr(module, 'ROOT', tmp_path)
        monkeypatch.setattr(module, 'fingerprints', lambda: {'fixture': 'fixed'})
    monkeypatch.setattr(prepare_comparison.df_runner, 'check_data_settings', lambda: None)
    data = tmp_path / 'data/processed'
    data.mkdir(parents=True)
    (data / 'metadata.json').write_text(metadata)
    for name in ('training', 'validation'):
        np.savez(data / f'{name}.npz', z=np.zeros((64, 48)), u=np.zeros(64), y=np.zeros(64))

    def forbidden(*args, **kwargs):
        pytest.fail('Main comparison preparation must not train or launch any work.')

    for module in modules[1:]:
        monkeypatch.setattr(module, 'execute', forbidden)
        if hasattr(module, 'evaluate'):
            monkeypatch.setattr(module, 'evaluate', forbidden)
    return tmp_path


def test_prepare_freezes_four_selected_campaigns_without_training(prepared_context):
    before = public_config()
    folder = prepare_comparison.prepare(prepared_context / 'comparison')
    assert public_config() == before
    protocol = read_json(folder / 'protocol.json')
    assert read_json(folder / 'status.json')['state'] == 'prepared'
    entries = protocol['campaigns']
    assert [entry['name'] for entry in entries] == ['df_ai', 'df_zi', 'zo_rsgd', 'mistp']
    assert len(entries) == 4
    assert protocol['config']['outer_seeds'] == RUN_SEEDS
    assert protocol['config']['evaluation_seeds'] == PANEL
    assert not list(folder.rglob('fits.jsonl'))
    assert not list(folder.rglob('history.jsonl'))
    assert not list(folder.rglob('worker.lock'))
    assert not list(folder.rglob('selection.json'))
    assert 'criterion' not in protocol
    assert 'selection' not in protocol
    for entry in entries:
        child = folder / entry['folder']
        assert read_json(child / 'status.json')['state'] == 'prepared'
        assert '--resume' in entry['command'] and '--background' in entry['command']
        assert entry['module'] in entry['command']
        assert str(child) in entry['command']
        assert entry['shell_command']
        assert entry['settings_sha256'] == hashlib.sha256((child / 'settings.json').read_bytes()).hexdigest()


def test_main_budget_separates_primary_matched_and_secondary_mistp_cost(prepared_context):
    folder = prepare_comparison.prepare(prepared_context / 'budgets')
    protocol = read_json(folder / 'protocol.json')
    expected = {
        'df_ai': (100000, 3800, 103800),
        'df_zi': (100000, 3800, 103800),
        'zo_rsgd': (100000, 3620, 103620),
        'mistp': (150000, 3820, 153820),
    }
    for entry in protocol['campaigns']:
        optimization, evaluation, total = expected[entry['name']]
        assert entry['budget'] == {
            'optimization_fits': optimization, 'evaluation_fits': evaluation, 'total_fits': total}
        assert read_json(folder / entry['folder'] / 'budget.json') == entry['budget']
        if entry['name'] == 'mistp':
            assert entry['primary_step'] == 3333
            assert entry['primary_optimization_fits_per_run'] == 9999
            assert entry['secondary_step'] == 5000
        else:
            assert entry['primary_step'] == 5000
            assert entry['primary_optimization_fits_per_run'] == 10000
            assert entry['secondary_step'] is None
    assert protocol['budget'] == {
        'optimization_fits': 450000, 'evaluation_fits': 15040, 'total_fits': 465040}
    assert read_json(folder / 'budget.json') == protocol['budget']


def test_frozen_methods_share_fresh_streams_and_preserve_inner_model(prepared_context):
    before = public_config()
    folder = prepare_comparison.prepare(prepared_context / 'settings')
    assert not set(RUN_SEEDS) & set(before['pilot_run_seeds'])
    assert not set(PANEL) & set(before['pilot_evaluation_seeds'])
    for entry in read_json(folder / 'protocol.json')['campaigns']:
        saved = read_json(folder / entry['folder'] / 'settings.json')
        child = saved['config']
        for key in ('initialization_seed', 'dimension', 'training_steps', 'learning_rate',
                    'iota2', 'iota3', 'p1', 'p2', 'p3', 'sigma', 'epsilon', 'L_f', 'L_c',
                    'hidden_width', 'batch_size', 'standardize'):
            assert child[key] == before[key]
        if entry['name'].startswith('df_'):
            assert child['modes'] == [entry['name'][-2:].upper()]
            assert child['iota1'] == 10.
            assert child['sqp_steps'] == 5000
            keys = ('outer_seeds', 'evaluation_seeds', 'evaluation_steps')
            points = POINTS
        elif entry['name'] == 'zo_rsgd':
            assert child['zo_rsgd_step_coefficient'] == 1.
            assert child['zo_rsgd_mu'] == .01 and child['zo_rsgd_m'] == 1
            assert child['zo_rsgd_steps'] == 5000
            assert saved['parameters']['eta'] == pytest.approx(1 / np.sqrt(5000))
            keys = ('zo_rsgd_run_seeds', 'zo_rsgd_evaluation_seeds', 'zo_rsgd_evaluation_steps')
            points = POINTS
        else:
            assert child['mistp_alphas'] == [.01]
            assert child['mistp_steps'] == 5000
            keys = ('mistp_run_seeds', 'mistp_evaluation_seeds', 'mistp_evaluation_steps')
            points = MISTP_POINTS
        assert child[keys[0]] == RUN_SEEDS
        assert child[keys[1]] == PANEL
        assert child[keys[2]] == points
    assert public_config() == before


def test_child_failure_restores_config_and_does_not_train(prepared_context, monkeypatch):
    before = public_config()

    def fail(*args, **kwargs):
        config.iota1 = 999.
        config.outer_seeds.append(999)
        config.temporary_test_setting = 'must disappear'
        raise RuntimeError('simulated preparation failure')

    monkeypatch.setattr(prepare_comparison.zo_runner, 'prepare', fail)
    folder = prepared_context / 'failed'
    with pytest.raises(RuntimeError, match='simulated'):
        prepare_comparison.prepare(folder)
    assert public_config() == before
    assert not hasattr(config, 'temporary_test_setting')
    assert not list(folder.rglob('fits.jsonl'))


@pytest.mark.parametrize('changes', [
    {'zo_rsgd_run_seeds': list(range(67_102_601, 67_102_611))},
    {'mistp_run_seeds': list(range(67_102_601, 67_102_611))},
    {'zo_rsgd_evaluation_seeds': list(range(48_102_601, 48_102_621))},
    {'mistp_evaluation_seeds': list(range(48_102_601, 48_102_621))},
    {'outer_seeds': RUN_SEEDS[:-1] + [RUN_SEEDS[0]]},
    {'evaluation_seeds': PANEL[:-1] + [2**32]},
    {'zo_rsgd_steps': 4999},
    {'mistp_steps': 4999},
    {'mistp_alphas': (.01, .001)},
    {'mistp_evaluation_steps': POINTS},
    {'outer_seeds': [66_102_401], 'zo_rsgd_run_seeds': [66_102_401],
     'mistp_run_seeds': [66_102_401]},
    {'evaluation_seeds': [47_102_401], 'zo_rsgd_evaluation_seeds': [47_102_401],
     'mistp_evaluation_seeds': [47_102_401]},
])
def test_invalid_main_protocol_rejected_before_output(prepared_context, monkeypatch, changes):
    for key, value in changes.items():
        monkeypatch.setattr(config, key, copy.deepcopy(value))
    before = public_config()
    folder = prepared_context / 'invalid'
    with pytest.raises(ValueError):
        prepare_comparison.prepare(folder)
    assert not folder.exists()
    assert public_config() == before


def test_existing_output_is_never_overwritten(prepared_context):
    folder = prepare_comparison.prepare(prepared_context / 'existing')
    protocol_bytes = (folder / 'protocol.json').read_bytes()
    before = public_config()
    with pytest.raises(FileExistsError):
        prepare_comparison.prepare(folder)
    assert (folder / 'protocol.json').read_bytes() == protocol_bytes
    assert public_config() == before
