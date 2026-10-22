"""Check saved coordinates, SE bands and meaningful small-sample/log-axis cases."""

import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.plot import COLORS, draw_metric, load_results, plot


def saved_results(folder, counts=(3, 3, 3)):
    settings = {'modes': ['AI', 'ZI'], 'outer_seeds': [10, 11, 12],
                'evaluation_steps': [0, 100, 5000], 'sqp_steps': 5000, 'iota1': 1.}
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
