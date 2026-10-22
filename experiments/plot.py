"""Plot saved AI/ZI summaries; no training or optimization is run.

python -m experiments.plot --run runs/<experiment>
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # Select the headless backend before pyplot.

# Existing document style. Change presentation here, not the saved experiment.
COLORS = {'AI': '#166287', 'ZI': '#b85424'}
FIGURE_SIZE = (7.5, 4.3)
LINE_WIDTH = 2.0
BAND_ALPHA = 0.32
PNG_DPI = 300
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 11,
         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'}


def load_results(folder):
    """Read the frozen settings and the runner's already-computed mean/SE values."""
    settings = json.loads((folder / 'settings.json').read_text())['config']
    curves = {}
    with (folder / 'summary.csv').open(newline='') as file:
        for row in csv.DictReader(file):
            mode, k, n = row['mode'], int(row['k']), int(row['runs'])
            if (mode not in COLORS or mode not in settings['modes']
                    or k not in settings['evaluation_steps']
                    or not 1 <= n <= len(settings['outer_seeds'])):
                raise ValueError('Summary mode, checkpoint or run count disagrees with saved settings.')
            values = {'k': k, 'runs': n}
            for metric in ('mse', 'feasibility'):
                mean = float(row['mean_' + metric])
                se = float(row['se_' + metric]) if row['se_' + metric] else np.nan
                if not np.isfinite(mean) or mean < 0 or (n > 1 and (not np.isfinite(se) or se < 0)):
                    raise ValueError(f'Invalid mean/SE for {mode}, step {k}, {metric}.')
                values['mean_' + metric] = mean
                values['se_' + metric] = se if n > 1 else np.nan
            curves.setdefault(mode, []).append(values)
    if not curves:
        raise ValueError('No completed evaluation panels are available to plot.')
    for mode, rows in curves.items():
        rows.sort(key=lambda row: row['k'])
        if len({row['k'] for row in rows}) != len(rows):
            raise ValueError(f'Duplicate summary checkpoint for {mode}.')
        curves[mode] = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    return settings, curves


def draw_metric(curves, metric):
    """One mean curve and ±1 SE band per mode; keep the saved checkpoint coordinates."""
    log_scale = metric == 'feasibility' and any(
        np.any(curve['mean_feasibility'] > 0) for curve in curves.values())
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout='constrained')
        for mode in ('AI', 'ZI'):
            if mode not in curves:
                continue
            curve = curves[mode]
            x, mean, se = curve['k'], curve['mean_' + metric], curve['se_' + metric]
            lower, upper = mean - se, mean + se
            line = mean.copy()
            if log_scale:
                # Log axes cannot show zero/negative values. Do not invent a floor.
                line = np.where(mean > 0, mean, np.nan)
                lower = np.where(lower > 0, lower, np.nan)
                upper = np.where(upper > 0, upper, np.nan)
            if np.any(np.isfinite(lower) & np.isfinite(upper)):
                ax.fill_between(x, lower, upper, color=COLORS[mode], alpha=BAND_ALPHA,
                                linewidth=0, zorder=2)
            ax.plot(x, line, color=COLORS[mode], lw=LINE_WIDTH, label=f'DF-SSQP {mode}',
                    marker='o' if np.count_nonzero(np.isfinite(line)) == 1 else None, markersize=3, clip_on=False, zorder=3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#B7BEC5')
        ax.tick_params(colors='#35414C', length=3)
        ax.grid(color='#B7BEC5', alpha=.22, linewidth=.7)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(1, max(curve['k'][-1] for curve in curves.values())))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.set_xlabel('SQP steps', labelpad=8)
        ax.set_ylabel('Validation MSE' if metric == 'mse' else r'Feasibility $|w^\top w-1|$', labelpad=8)
        if log_scale:
            ax.set_yscale('log')
        else:
            ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))
        ax.legend(loc='best' if metric == 'feasibility' else 'upper right', frameon=False, fontsize=9.5,
                  handlelength=3.1, labelspacing=.55, borderaxespad=.4)
    return fig, ax


def plot(folder):
    settings, curves = load_results(folder)
    output = folder / 'figures'
    output.mkdir(exist_ok=True)
    for metric in ('mse', 'feasibility'):
        fig, _ = draw_metric(curves, metric)
        try:
            with plt.rc_context(STYLE):
                fig.savefig(output / f'{metric}.pdf')
                fig.savefig(output / f'{metric}.png', dpi=PNG_DPI)
        finally:
            plt.close(fig)

    # Keep figure legends simple; statistical details and actual counts live here.
    notes = ['Lines: mean across stochastic runs; shading: +/-1 SE across run means.',
             'Curves connect saved evaluation checkpoints; no smoothing is applied.',
             'MSE per run is averaged over the independent reporting seeds.',
             'One run has no estimable across-run SE and is drawn without a band.',
             'Feasibility uses a log axis when positive residuals exist. Nonpositive',
             'values/band bounds are omitted from that display, never replaced by a floor.',
             f"Step coefficient iota1: {settings['iota1']}."]
    for mode in settings['modes']:
        curve = curves.get(mode)
        if curve is None:
            notes.append(f'{mode}: no completed evaluation panels yet.')
            continue
        counts = curve['runs']
        n = str(counts[0]) if np.all(counts == counts[0]) else f'{counts.min()}-{counts.max()}'
        complete = (np.array_equal(curve['k'], settings['evaluation_steps'])
                    and np.all(counts == len(settings['outer_seeds'])))
        notes.append(f"{mode}: {n}/{len(settings['outer_seeds'])} runs per checkpoint; "
                     f"through step {curve['k'][-1]}/{settings['sqp_steps']}; "
                     f"{'all configured panels present' if complete else 'partial results'}.")
    (output / 'notes.txt').write_text('\n'.join(notes) + '\n')
    print('\n'.join(notes))
    print(f'Figures: {output}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Saved experiment folder.')
    args = parser.parse_args()
    plot(args.run.resolve())


if __name__ == '__main__':
    main()
