"""Check saved coordinates, SE bands and meaningful small-sample/log-axis cases."""

import csv
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.plot import BASELINES, COLORS, draw_metric, load_baselines, load_results, plot


def saved_results(folder, counts=(3, 3, 3)):
    settings = {'modes': ['AI', 'ZI'], 'outer_seeds': [10, 11, 12],
                'evaluation_steps': [0, 100, 5000], 'sqp_steps': 5000, 'iota1': 1.,
                'evaluation_seeds': [20, 21, 22], 'baselines': list(BASELINES)}
    (folder / 'settings.json').write_text(json.dumps({'config': settings}))
    rows = []
    for mode in settings['modes']:
        for k, n, mean, se in zip(settings['evaluation_steps'], counts, [.5, .4, .3], [.02, .01, .005]):
            rows.append({'mode': mode, 'k': k, 'runs': n, 'mean_mse': mean,
                         'se_mse': se if n > 1 else '',
                         'mean_feasibility': mean/10, 'se_feasibility': se/10 if n > 1 else ''})
    with (folder / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return rows


@pytest.mark.parametrize('metric', ['mse', 'feasibility'])
def test_lines_and_band_coordinates_preserve_saved_values(tmp_path, metric):
    saved_results(tmp_path)
    _, curves = load_results(tmp_path)
    fig, ax = draw_metric(curves, metric)
    try:
        assert [line.get_label() for line in ax.lines] == ['DF-SSQP AI', 'DF-SSQP ZI']
        assert ax.get_xlabel() == 'SQP steps' and not ax.get_title()
        assert ax.get_yscale() == ('log' if metric == 'feasibility' else 'linear')
        for mode, line, band in zip(('AI', 'ZI'), ax.lines, ax.collections):
            curve = curves[mode]
            np.testing.assert_array_equal(line.get_xdata(), [0, 100, 5000])
            np.testing.assert_array_equal(line.get_ydata(), curve['mean_' + metric])
            assert line.get_color() == COLORS[mode]
            vertices = band.get_paths()[0].vertices
            for k, mean, se in zip(curve['k'], curve['mean_' + metric], curve['se_' + metric]):
                y = vertices[vertices[:, 0] == k, 1]
                assert min(y) == mean-se and max(y) == mean+se
    finally:
        plt.close(fig)


def test_one_run_has_no_band_and_log_zero_is_not_replaced(tmp_path):
    saved_results(tmp_path, counts=(1, 1, 1))
    _, curves = load_results(tmp_path)
    for curve in curves.values():
        curve['mean_feasibility'] = np.array([0., 0., 1e-16])
    fig, ax = draw_metric(curves, 'feasibility')
    try:
        assert not ax.collections
        for line in ax.lines:
            np.testing.assert_array_equal(line.get_ydata(), [np.nan, np.nan, 1e-16])
            assert line.get_marker() == 'o'
        assert ax.get_yscale() == 'log'
    finally:
        plt.close(fig)


def test_nonpositive_band_bound_and_all_zero_case(tmp_path):
    saved_results(tmp_path)
    _, curves = load_results(tmp_path)
    for curve in curves.values():
        curve['mean_feasibility'] = np.array([0., .1, .2])
        curve['se_feasibility'] = np.array([0., .1, .01])
    fig, ax = draw_metric(curves, 'feasibility')
    try:
        for band in ax.collections:
            assert all(np.all(path.vertices[:, 1] > 0) for path in band.get_paths())
        assert np.isnan(ax.lines[0].get_ydata()[0])
        assert ax.lines[0].get_ydata()[1] == .1
    finally:
        plt.close(fig)
    for curve in curves.values():
        curve['mean_feasibility'][:] = 0
        curve['se_feasibility'][:] = 0
    fig, ax = draw_metric(curves, 'feasibility')
    try:
        assert ax.get_yscale() == 'linear'
        assert np.all(ax.lines[0].get_ydata() == 0)
    finally:
        plt.close(fig)


def test_output_and_partial_counts_without_changing_inputs(tmp_path):
    saved_results(tmp_path, counts=(3, 2, 1))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    output = plot(tmp_path)
    assert {p.name for p in output.iterdir()} == {'mse.pdf', 'mse.png', 'feasibility.pdf', 'feasibility.png', 'notes.txt'}
    assert (output / 'mse.pdf').read_bytes().startswith(b'%PDF')
    assert (output / 'feasibility.png').read_bytes().startswith(b'\x89PNG')
    notes = (output / 'notes.txt').read_text()
    assert 'AI: 1-3/3 runs' in notes and 'partial results' in notes
    assert all((tmp_path/name).read_bytes() == value for name, value in before.items())
    assert not plt.get_fignums()


def test_bad_statistics_and_duplicate_checkpoints_are_rejected(tmp_path):
    saved_results(tmp_path)
    path = tmp_path / 'summary.csv'
    text = path.read_text()
    path.write_text(text.replace('0.02', '-0.02', 1))
    with pytest.raises(ValueError, match='Invalid mean/SE'):
        load_results(tmp_path)
    path.write_text(text + text.splitlines()[1] + '\n')
    with pytest.raises(ValueError, match='Duplicate'):
        load_results(tmp_path)


def saved_baselines(folder, mean_count=3):
    rows = [dict(model='aggregate_only', evaluation_fits=0, mean_mse=.48,
                 se_mse='', panel_complete=True),
            dict(model='mean', evaluation_fits=mean_count, mean_mse=.54 if mean_count else '',
                 se_mse=.02 if mean_count > 1 else '', panel_complete=mean_count == 3),
            dict(model='pca', evaluation_fits=3, mean_mse=.51,
                 se_mse=.01, panel_complete=True)]
    directory = folder / 'baselines'
    directory.mkdir(exist_ok=True)
    parent_hash = hashlib.sha256((folder / 'settings.json').read_bytes()).hexdigest()
    (directory / 'settings.json').write_text(json.dumps({'parent_settings_sha256': parent_hash}))
    with (directory / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_fixed_baselines_add_mse_lines_without_changing_curves_or_bands(tmp_path):
    saved_results(tmp_path)
    saved_baselines(tmp_path)
    settings, curves = load_results(tmp_path)
    baselines = load_baselines(tmp_path, settings)
    original, ax_original = draw_metric(curves, 'mse')
    combined, ax = draw_metric(curves, 'mse', baselines)
    try:
        assert [line.get_label() for line in ax.lines] == [
            'DF-SSQP AI', 'DF-SSQP ZI', 'Aggregate-only', 'Mean', 'PCA']
        assert len(ax.collections) == len(ax_original.collections) == 2
        for line, old_line in zip(ax.lines[:2], ax_original.lines):
            np.testing.assert_array_equal(line.get_xydata(), old_line.get_xydata())
        for band, old_band in zip(ax.collections, ax_original.collections):
            np.testing.assert_array_equal(band.get_paths()[0].vertices,
                                          old_band.get_paths()[0].vertices)
        for name, line in zip(BASELINES, ax.lines[2:]):
            np.testing.assert_array_equal(line.get_ydata(), [baselines[name]['mean_mse']] * 2)
            assert line.get_color() == BASELINES[name][1]
            assert line.get_linestyle() == BASELINES[name][2]
    finally:
        plt.close(original)
        plt.close(combined)


def test_baselines_leave_feasibility_figure_exactly_unchanged(tmp_path):
    saved_results(tmp_path)
    saved_baselines(tmp_path)
    settings, curves = load_results(tmp_path)
    baseline = load_baselines(tmp_path, settings)
    original, _ = draw_metric(curves, 'feasibility')
    combined, _ = draw_metric(curves, 'feasibility', baseline)
    try:
        original.canvas.draw()
        combined.canvas.draw()
        np.testing.assert_array_equal(np.asarray(original.canvas.buffer_rgba()),
                                      np.asarray(combined.canvas.buffer_rgba()))
    finally:
        plt.close(original)
        plt.close(combined)


@pytest.mark.parametrize('mean_count', [0, 1, 2])
def test_incomplete_baseline_is_omitted_and_recorded_without_changing_inputs(tmp_path, mean_count):
    saved_results(tmp_path)
    saved_baselines(tmp_path, mean_count)
    settings, curves = load_results(tmp_path)
    baselines = load_baselines(tmp_path, settings)
    fig, ax = draw_metric(curves, 'mse', baselines)
    try:
        assert 'Mean' not in [line.get_label() for line in ax.lines]
        assert 'PCA' in [line.get_label() for line in ax.lines]
    finally:
        plt.close(fig)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    output = plot(tmp_path)
    notes = (output / 'notes.txt').read_text()
    assert f'Mean: incomplete evaluation panel ({mean_count} fits); omitted.' in notes
    assert 'across run means' in notes and 'reporting seeds' in notes
    assert 'without uncertainty bands' in notes
    assert all((tmp_path/name).read_bytes() == value for name, value in before.items())


@pytest.mark.parametrize('old,new', [
    ('aggregate_only', 'unknown'), ('pca', 'mean'),
    ('0,0.48,,True', '1,0.48,,True'),
    ('3,0.54,0.02,True', '2,0.54,0.02,True'),
    ('3,0.54,0.02,True', '4,0.54,0.02,False'),
    ('0.54', '-0.54'), ('0.54', 'nan'), ('0.02', '-0.02'), ('0.02', 'nan'),
    ('3,0.54,0.02,True', '3,0.54,,True'), ('True', 'maybe')])
def test_invalid_baseline_summary_is_rejected(tmp_path, old, new):
    saved_results(tmp_path)
    saved_baselines(tmp_path)
    settings, _ = load_results(tmp_path)
    path = tmp_path / 'baselines' / 'summary.csv'
    path.write_text(path.read_text().replace(old, new, 1))
    with pytest.raises(ValueError):
        load_baselines(tmp_path, settings)


def test_baselines_from_another_seed_panel_cannot_be_plotted(tmp_path):
    saved_results(tmp_path)
    saved_baselines(tmp_path)
    path = tmp_path / 'settings.json'
    saved = json.loads(path.read_text())
    saved['config']['evaluation_seeds'] = [30, 31, 32]
    path.write_text(json.dumps(saved))
    settings, _ = load_results(tmp_path)
    with pytest.raises(ValueError, match='do not match'):
        load_baselines(tmp_path, settings)
    (tmp_path / 'baselines' / 'settings.json').unlink()
    with pytest.raises(ValueError, match='settings are missing'):
        load_baselines(tmp_path, settings)


def test_baseline_must_be_selected_in_frozen_settings(tmp_path):
    saved_results(tmp_path)
    path = tmp_path / 'settings.json'
    saved = json.loads(path.read_text())
    saved['config']['baselines'] = ['aggregate_only', 'mean']
    path.write_text(json.dumps(saved))
    saved_baselines(tmp_path)
    settings, _ = load_results(tmp_path)
    with pytest.raises(ValueError, match='unconfigured'):
        load_baselines(tmp_path, settings)
