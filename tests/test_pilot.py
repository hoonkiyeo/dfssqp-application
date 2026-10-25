"""Common step-scale pilot preparation/reporting; synthetic losses only."""

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

import config
from algorithms import df_ssqp, mistp, zo_rsgd
from experiments import pilot


def public_config():
    return {key: copy.deepcopy(value) for key, value in vars(config).items()
            if not key.startswith('_') and not callable(value)}


@pytest.fixture
def prepared_context(monkeypatch, tmp_path):
    # Existing runners restore their frozen config on execute; isolate every entry.
    for key, value in public_config().items():
        monkeypatch.setattr(config, key, copy.deepcopy(value))
    settings = {
        'outer_seeds': list(range(56_100_901, 56_100_911)),
        'evaluation_seeds': list(range(46_100_901, 46_100_921)),
        'pilot_steps': 5000,
        'pilot_run_seeds': [66_102_401, 66_102_402, 66_102_403],
        'pilot_evaluation_seeds': list(range(47_102_401, 47_102_421)),
        'pilot_evaluation_steps': [0, 100, 500, 1000, 2000, 3500, 5000],
        'pilot_df_iota1s': (1., 10., 100.),
        'pilot_zo_coefficients': (.01, .1, 1.),
        'pilot_mistp_alphas': (1., .1, .01),
        'pilot_experiment_name': 'step_scale_pilot',
    }
    for key, value in settings.items():
        monkeypatch.setattr(config, key, value, raising=False)
    metadata = (pilot.df_runner.ROOT / 'data/processed/metadata.json').read_text()
    for module in (pilot, pilot.df_runner, pilot.zo_runner, pilot.mistp_runner):
        monkeypatch.setattr(module, 'ROOT', tmp_path)
        monkeypatch.setattr(module, 'fingerprints', lambda: {'fixture': 'fixed'})
    data = tmp_path / 'data/processed'
    data.mkdir(parents=True)
    (data / 'metadata.json').write_text(metadata)
    for name in ('training', 'validation'):
        np.savez(data / f'{name}.npz', z=np.zeros((64, 48)), u=np.zeros(64), y=np.zeros(64))

    def no_training(*args, **kwargs):
        pytest.fail('Pilot preparation/reporting must not train a model.')

    monkeypatch.setattr(pilot.df_runner, 'evaluate', no_training)
    return tmp_path


def read_json(path):
    return json.loads(path.read_text())


def campaigns(folder):
    return read_json(folder / 'protocol.json')['campaigns']


def test_prepare_freezes_seven_campaigns_without_starting_work(prepared_context, monkeypatch):
    for module in (pilot.df_runner, pilot.zo_runner, pilot.mistp_runner):
        monkeypatch.setattr(module, 'execute', lambda *args, **kwargs: pytest.fail('Do not execute.'))
    before = public_config()
    folder = pilot.prepare(prepared_context / 'prepared')
    assert public_config() == before
    assert read_json(folder / 'status.json')['state'] == 'prepared'
    assert not list(folder.rglob('fits.jsonl'))
    assert not list(folder.rglob('history.jsonl'))
    protocol = read_json(folder / 'protocol.json')
    assert len(protocol['campaigns']) == 7
    assert [entry['method'] for entry in protocol['campaigns']].count('DF-SSQP') == 3
    assert [entry['method'] for entry in protocol['campaigns']].count('ZO-RSGD') == 3
    assert [entry['method'] for entry in protocol['campaigns']].count('MiSTP') == 1
    assert protocol['config']['pilot_run_seeds'] == before['pilot_run_seeds']
    # Pilot is not a method ranking: MiSTP pays for a third fit per iteration.
    assert 'not the same' in protocol['warning']
    assert 'three' in protocol['warning']
    for entry in protocol['campaigns']:
        assert read_json(folder / entry['folder'] / 'status.json')['state'] == 'prepared'
        assert '--resume' in entry['command'] and '--background' in entry['command']
        assert entry['module'] in entry['command']


def test_exact_full_pilot_budget(prepared_context):
    folder = pilot.prepare(prepared_context / 'budgets')
    entries = campaigns(folder)
    expected_by_method = {
        'DF-SSQP': (3 * 2 * 3 * 5000 * 2, 3 * 2 * 3 * 7 * 20),
        'ZO-RSGD': (3 * 3 * 5000 * 2, 3 * (1 + 3 * 6) * 20),
        'MiSTP': (3 * 3 * 5000 * 3, (1 + 3 * 3 * 6) * 20),
    }
    for method, (optimization, reporting) in expected_by_method.items():
        selected = [entry for entry in entries if entry['method'] == method]
        assert sum(entry['budget']['optimization_fits'] for entry in selected) == optimization
        assert sum(entry['budget']['evaluation_fits'] for entry in selected) == reporting
    assert read_json(folder / 'budget.json') == {
        'optimization_fits': 405000, 'evaluation_fits': 4760, 'total_fits': 409760}
    assert read_json(folder / 'protocol.json')['budget'] == read_json(folder / 'budget.json')


def test_saved_child_settings_share_protocol_and_only_target_coefficient(prepared_context):
    before = public_config()
    folder = pilot.prepare(prepared_context / 'frozen')
    for entry in campaigns(folder):
        saved = read_json(folder / entry['folder'] / 'settings.json')
        child = saved['config']
        for key in ('initialization_seed', 'dimension', 'training_steps', 'learning_rate',
                    'iota2', 'iota3', 'p1', 'p2', 'p3', 'zo_rsgd_mu', 'zo_rsgd_m'):
            assert child[key] == before[key]
        if entry['method'] == 'DF-SSQP':
            assert child['modes'] == ['AI', 'ZI']
            assert child['iota1'] == entry['coefficient']
            assert child['sqp_steps'] == 5000
            keys = ('outer_seeds', 'evaluation_seeds', 'evaluation_steps')
        elif entry['method'] == 'ZO-RSGD':
            assert child['zo_rsgd_step_coefficient'] == entry['coefficient']
            assert child['zo_rsgd_steps'] == 5000
            assert saved['parameters']['eta'] == pytest.approx(entry['coefficient'] / np.sqrt(5000))
            assert child['iota1'] == before['iota1']
            keys = ('zo_rsgd_run_seeds', 'zo_rsgd_evaluation_seeds', 'zo_rsgd_evaluation_steps')
        else:
            assert child['mistp_alphas'] == [1., .1, .01]
            assert child['mistp_steps'] == 5000
            assert child['iota1'] == before['iota1']
            keys = ('mistp_run_seeds', 'mistp_evaluation_seeds', 'mistp_evaluation_steps')
        assert child[keys[0]] == before['pilot_run_seeds']
        assert child[keys[1]] == before['pilot_evaluation_seeds']
        assert child[keys[2]] == before['pilot_evaluation_steps']
    assert not set(before['pilot_run_seeds']) & set(before['outer_seeds'])
    assert not set(before['pilot_evaluation_seeds']) & set(before['evaluation_seeds'])
    assert public_config() == before


def test_failed_child_prepare_restores_all_public_config(prepared_context, monkeypatch):
    before = public_config()

    def fail(*args, **kwargs):
        config.iota1 = 999.
        config.outer_seeds.append(999)
        config.temporary_test_setting = 'must disappear'
        raise RuntimeError('simulated preparation failure')

    monkeypatch.setattr(pilot.zo_runner, 'prepare', fail)
    with pytest.raises(RuntimeError, match='simulated'):
        pilot.prepare(prepared_context / 'failed_child')
    assert public_config() == before
    assert not hasattr(config, 'temporary_test_setting')
    assert not list((prepared_context / 'failed_child').rglob('fits.jsonl'))


@pytest.mark.parametrize('key,value', [
    ('pilot_steps', 0), ('pilot_steps', 3.5),
    ('pilot_run_seeds', []), ('pilot_run_seeds', [1, 1]),
    ('pilot_run_seeds', [-1]), ('pilot_run_seeds', [56_100_901]),
    ('pilot_evaluation_seeds', []), ('pilot_evaluation_seeds', [2**32]),
    ('pilot_evaluation_seeds', [46_100_901]),
    ('pilot_evaluation_seeds', [47_102_401, 47_102_401]),
    ('pilot_evaluation_seeds', [66_102_401]),
    ('pilot_evaluation_steps', [0, 100]),
    ('pilot_evaluation_steps', [0, 5000, 100]),
    ('pilot_df_iota1s', []), ('pilot_df_iota1s', [0.]),
    ('pilot_df_iota1s', [1., 1.]), ('pilot_df_iota1s', [float('inf')]),
    ('pilot_zo_coefficients', [-.1]), ('pilot_mistp_alphas', [0.]),
])
def test_invalid_protocol_fails_before_partial_output(prepared_context, monkeypatch, key, value):
    monkeypatch.setattr(config, key, value)
    before = json.dumps(public_config(), sort_keys=True)
    folder = prepared_context / 'invalid'
    with pytest.raises(ValueError):
        pilot.prepare(folder)
    assert not folder.exists()
    assert json.dumps(public_config(), sort_keys=True) == before


def run_directories(folder):
    protocol = read_json(folder / 'protocol.json')
    for entry in protocol['campaigns']:
        child = folder / entry['folder']
        for coefficient in entry['coefficients']:
            variants = ('AI', 'ZI') if entry['method'] == 'DF-SSQP' else (None,)
            for variant in variants:
                method = f'DF-SSQP {variant}' if variant else entry['method']
                for seed in protocol['config']['pilot_run_seeds']:
                    if variant:
                        run = child / f'{variant}_{seed}'
                    elif entry['method'] == 'ZO-RSGD':
                        run = child / f'run_{seed}'
                    else:
                        run = child / f'alpha_{float(coefficient)}' / f'run_{seed}'
                    yield method, coefficient, seed, run


def fake_panels(folder, partial=None, missing=None, wrong_count=None):
    protocol = read_json(folder / 'protocol.json')
    settings = protocol['config']
    final = settings['pilot_steps']
    n_eval = len(settings['pilot_evaluation_seeds'])
    offsets = [0., 1., 4.]
    expected = {}
    for method, coefficient, seed, run in run_directories(folder):
        key = (method, coefficient, seed)
        if key == missing:
            continue
        seed_index = settings['pilot_run_seeds'].index(seed)
        optimum = 10. if method.startswith('DF-SSQP') else .1
        mse = 10. + np.log10(coefficient / optimum)**2 + offsets[seed_index]
        feasibility = .001 * (seed_index + 1) if method.startswith('DF-SSQP') else 1e-16
        complete = key != partial
        count = n_eval - 1 if key == wrong_count else n_eval
        row = {'k': final, 'evaluation_fits': count, 'panel_complete': complete,
               'mean_mse': mse, 'feasibility': feasibility}
        run.mkdir(parents=True, exist_ok=True)
        with (run / 'evaluations.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        if complete and count == n_eval:
            expected.setdefault((method, coefficient), []).append((mse, feasibility))
    return expected


def read_summary(folder):
    with (folder / 'summary.csv').open() as file:
        return list(csv.DictReader(file))


def test_report_aggregates_run_means_and_selects_only_with_complete_candidates(prepared_context):
    folder = pilot.prepare(prepared_context / 'complete')
    expected = fake_panels(folder)
    before = public_config()
    selection = pilot.report(folder)
    assert public_config() == before
    assert selection['complete'] and selection['provisional']
    assert selection == read_json(folder / 'selection.json')
    for row in read_summary(folder):
        values = np.array(expected[row['method'], float(row['coefficient'])])
        assert int(row['runs']) == 3
        assert float(row['mean_mse']) == pytest.approx(values[:, 0].mean())
        assert float(row['se_mse']) == pytest.approx(values[:, 0].std(ddof=1) / np.sqrt(3))
        assert float(row['mean_feasibility']) == pytest.approx(values[:, 1].mean())
        assert float(row['se_feasibility']) == pytest.approx(values[:, 1].std(ddof=1) / np.sqrt(3))
        # Twenty fits form each run mean; they are not sixty independent runs.
        wrong_se = np.repeat(values[:, 0], 20).std(ddof=1) / np.sqrt(60)
        assert not np.isclose(float(row['se_mse']), wrong_se)
    for method, result in selection['methods'].items():
        assert result['complete']
        assert result['provisional_coefficient'] == (10. if method.startswith('DF-SSQP') else .1)


@pytest.mark.parametrize('partial_kind', ['partial', 'missing', 'wrong_count'])
def test_incomplete_seed_panels_are_excluded_and_prevent_selection(prepared_context, partial_kind):
    folder = pilot.prepare(prepared_context / partial_kind)
    key = ('DF-SSQP AI', 1., config.pilot_run_seeds[-1])
    expected = fake_panels(folder, **{partial_kind: key})
    selection = pilot.report(folder)
    assert not selection['complete']
    assert not selection['methods']['DF-SSQP AI']['complete']
    assert selection['methods']['DF-SSQP AI']['provisional_coefficient'] is None
    assert all(value['complete'] for method, value in selection['methods'].items() if method != 'DF-SSQP AI')
    row = next(row for row in read_summary(folder)
               if row['method'] == 'DF-SSQP AI' and float(row['coefficient']) == 1.)
    assert int(row['runs']) == 2
    assert float(row['mean_mse']) == pytest.approx(np.mean(np.array(expected['DF-SSQP AI', 1.])[:, 0]))


def test_no_results_cannot_select_any_coefficient(prepared_context):
    folder = pilot.prepare(prepared_context / 'empty')
    selection = pilot.report(folder)
    assert not selection['complete']
    assert read_summary(folder) == []
    assert all(not item['complete'] and item['provisional_coefficient'] is None
               for item in selection['methods'].values())


def test_report_uses_frozen_protocol_and_checks_child_integrity(prepared_context, monkeypatch):
    folder = pilot.prepare(prepared_context / 'frozen_report')
    fake_panels(folder)
    monkeypatch.setattr(config, 'pilot_steps', 999)
    monkeypatch.setattr(config, 'pilot_run_seeds', [999])
    monkeypatch.setattr(config, 'pilot_evaluation_seeds', [998])
    before = public_config()
    assert pilot.report(folder)['complete']
    assert public_config() == before
    child = folder / campaigns(folder)[0]['folder'] / 'settings.json'
    saved = read_json(child)
    saved['config']['iota1'] = 999.
    child.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='changed'):
        pilot.report(folder)


@pytest.mark.parametrize('k', [0, 1, 7])
def test_methods_share_training_seed_and_gaussian_source(k):
    states = [df_ssqp.initialize(), zo_rsgd.initialize(), mistp.initialize()]
    for state in states:
        state['k'] = k
    df_trace = df_ssqp.step(states[0], 66_102_401, 'AI', lambda w, seed: 1.)[1]
    zo_trace = zo_rsgd.step(states[1], 66_102_401, .1, .01, lambda w, seed: 1.)[1]
    mistp_trace = mistp.step(states[2], 66_102_401, .1, lambda w, seed: 1.)[1]
    assert df_trace['seed'] == zo_trace['seed'] == mistp_trace['seed'] >= 2**32
    w = states[0]['w']
    streams = np.random.SeedSequence([66_102_401, k, 20260910]).spawn(2)
    gaussian = np.random.default_rng(streams[0]).normal(size=len(w))
    np.testing.assert_array_equal(mistp_trace['s'], gaussian / np.linalg.norm(gaussian))
    np.testing.assert_array_equal(zo_trace['u'], gaussian - (w @ gaussian) * w)
    for state in states[1:]:
        np.testing.assert_array_equal(state['w'], w)
    assert set(np.unique(df_trace['delta'])) <= {-1., 1.}
    assert df_trace['seed'] not in config.pilot_evaluation_seeds


def scientific_results(folder):
    output = {}
    for path in sorted(folder.rglob('history.jsonl')):
        output[str(path.relative_to(folder))] = pilot.df_runner.read_journal(path)
    for path in sorted(folder.rglob('fits.jsonl')):
        output[str(path.relative_to(folder))] = {
            item['key']: (item['seed'], item['w_sha256'], item['mse'])
            for item in pilot.df_runner.read_journal(path) if item['event'] == 'done'}
    return output


def execute_campaigns(folder, paused=False):
    modules = {'DF-SSQP': pilot.df_runner, 'ZO-RSGD': pilot.zo_runner, 'MiSTP': pilot.mistp_runner}
    for entry in campaigns(folder):
        original = public_config()
        try:
            module = modules[entry['method']]
            child = folder / entry['folder']
            if paused:
                module.execute(child, stop_after=1)
            module.execute(child)
        finally:
            vars(config).update(original)


def test_existing_runners_execute_frozen_coefficients_and_resume_identically(prepared_context, monkeypatch):
    for key, value in {
        'pilot_steps': 3, 'pilot_run_seeds': [71, 72, 73],
        'pilot_evaluation_seeds': [101, 102], 'pilot_evaluation_steps': [0, 1, 3],
    }.items():
        monkeypatch.setattr(config, key, value)
    target = np.linspace(-1., 1., 48)
    calls = []

    def synthetic(w, seed, training, validation):
        calls.append(int(seed))
        return float(np.mean((np.asarray(w) - target)**2) + (seed % 17) * .001)

    monkeypatch.setattr(pilot.df_runner, 'evaluate', synthetic)
    full = pilot.prepare(prepared_context / 'synthetic_full')
    resumed = pilot.prepare(prepared_context / 'synthetic_resumed')
    before = public_config()
    execute_campaigns(full)
    execute_campaigns(resumed, paused=True)
    assert public_config() == before
    assert scientific_results(full) == scientific_results(resumed)
    assert len(calls) == 2 * read_json(full / 'budget.json')['total_fits']
    assert pilot.report(full) == pilot.report(resumed)
    assert read_json(full / 'selection.json')['complete']
    for entry in campaigns(full):
        child = full / entry['folder']
        histories = [pilot.df_runner.read_journal(path) for path in child.rglob('history.jsonl')]
        assert histories
        for history in histories:
            first_metrics = history[1]['metrics']
            if entry['method'] == 'DF-SSQP':
                assert first_metrics['alpha'] == entry['coefficient']
            elif entry['method'] == 'ZO-RSGD':
                assert first_metrics['eta'] == pytest.approx(entry['coefficient'] / np.sqrt(3))
            else:
                assert first_metrics['alpha'] in entry['coefficients']


def test_additional_pilot_freezes_only_new_candidates_and_method_specific_budget(prepared_context):
    before = public_config()
    folder = pilot.prepare_additional(prepared_context / 'additional')
    assert public_config() == before
    entries = campaigns(folder)
    assert len(entries) == 3
    expected_coefficients = {'DF-SSQP': [20.], 'ZO-RSGD': [10.], 'MiSTP': [.001]}
    expected_budgets = {'DF-SSQP': (60000, 840), 'ZO-RSGD': (30000, 380),
                        'MiSTP': (45000, 440)}
    for entry in entries:
        method = entry['method']
        assert entry['coefficients'] == expected_coefficients[method]
        child = read_json(folder / entry['folder'] / 'settings.json')['config']
        if method == 'DF-SSQP':
            keys = ('outer_seeds', 'evaluation_seeds', 'evaluation_steps')
            assert child['iota1'] == 20.
            assert child['modes'] == ['AI', 'ZI']
        elif method == 'ZO-RSGD':
            keys = ('zo_rsgd_run_seeds', 'zo_rsgd_evaluation_seeds', 'zo_rsgd_evaluation_steps')
            saved = read_json(folder / entry['folder'] / 'settings.json')
            assert saved['parameters']['eta'] == pytest.approx(10. / np.sqrt(5000))
            assert saved['parameters']['mu'] == .01
            assert saved['parameters']['m'] == 1
        else:
            keys = ('mistp_run_seeds', 'mistp_evaluation_seeds', 'mistp_evaluation_steps')
            assert child['mistp_steps'] == 5000
            assert child['mistp_alphas'] == [.001]
        assert child[keys[0]] == before['pilot_run_seeds']
        assert child[keys[1]] == before['pilot_evaluation_seeds']
        expected_points = [0, 100, 500, 1000, 2000, 3500, 5000]
        if method == 'MiSTP':
            expected_points = [0, 100, 500, 1000, 2000, 3333, 3500, 5000]
        assert child[keys[2]] == expected_points
        assert child['initialization_seed'] == before['initialization_seed']
        for key in ('training_steps', 'learning_rate', 'batch_size', 'iota2', 'iota3',
                    'p1', 'p2', 'p3'):
            assert child[key] == before[key]
        opt, report_fits = expected_budgets[method]
        assert entry['budget'] == {'optimization_fits': opt,
                                   'evaluation_fits': report_fits,
                                   'total_fits': opt + report_fits}
    assert read_json(folder / 'budget.json') == {
        'optimization_fits': 135000, 'evaluation_fits': 1660, 'total_fits': 136660}
    assert not list(folder.rglob('fits.jsonl'))
    assert not list(folder.rglob('history.jsonl'))


def test_additional_report_accepts_mistp_equal_budget_checkpoint_without_early_selection(
        prepared_context, monkeypatch):
    folder = pilot.prepare_additional(prepared_context / 'additional_reporting')
    missing_seed = config.pilot_run_seeds[-1]
    fake_panels(folder, missing=('MiSTP', .001, missing_seed))
    mistp_entry = next(entry for entry in campaigns(folder) if entry['method'] == 'MiSTP')
    missing_run = None
    for index, seed in enumerate(config.pilot_run_seeds):
        run = folder / mistp_entry['folder'] / f'alpha_{float(.001)}' / f'run_{seed}'
        run.mkdir(parents=True, exist_ok=True)
        path = run / 'evaluations.csv'
        rows = []
        if path.exists():
            with path.open() as file:
                rows = list(csv.DictReader(file))
        rows.insert(0, {'k': 3333, 'evaluation_fits': 20, 'panel_complete': True,
                        'mean_mse': 4. + index, 'feasibility': 1e-16})
        with path.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if seed == missing_seed:
            missing_run = path
    # Reporting must use the saved child checkpoints, not changed live settings.
    monkeypatch.setattr(config, 'pilot_mistp_evaluation_steps', [0, 5000])
    before = public_config()
    selection = pilot.report(folder)
    assert public_config() == before
    assert not selection['complete']
    assert not selection['methods']['MiSTP']['complete']
    assert selection['methods']['MiSTP']['provisional_coefficient'] is None
    intermediate = next(row for row in read_summary(folder)
                        if row['method'] == 'MiSTP' and int(row['k']) == 3333)
    assert int(intermediate['runs']) == 3
    assert float(intermediate['mean_mse']) == pytest.approx(5.)
    assert float(intermediate['se_mse']) == pytest.approx(1. / np.sqrt(3))
    final = next(row for row in read_summary(folder)
                 if row['method'] == 'MiSTP' and int(row['k']) == 5000)
    assert int(final['runs']) == 2
    assert all(value['complete'] for method, value in selection['methods'].items()
               if method != 'MiSTP')
    with missing_run.open('a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=('k', 'evaluation_fits', 'panel_complete',
                                                 'mean_mse', 'feasibility'))
        writer.writerow({'k': 5000, 'evaluation_fits': 20, 'panel_complete': True,
                          'mean_mse': 9., 'feasibility': 1e-16})
    complete = pilot.report(folder)
    assert complete['complete']
    assert complete['methods']['MiSTP']['provisional_coefficient'] == .001


@pytest.mark.parametrize('points', [
    [0, 3333],
    [0, 3333, 3333, 5000],
    [0, 5000, 3333],
    [0, 3333., 5000],
])
def test_invalid_method_specific_mistp_points_fail_before_preparation(
        prepared_context, monkeypatch, points):
    monkeypatch.setattr(config, 'pilot_mistp_evaluation_steps', points)
    before = public_config()
    folder = prepared_context / 'invalid_mistp_points'
    with pytest.raises(ValueError, match='[Mm]i[Ss][Tt][Pp]|mistp'):
        pilot.prepare(folder)
    assert not folder.exists()
    assert public_config() == before
