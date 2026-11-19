"""Lock a challenger on validation and export comparable design-level test results."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean


CANDIDATES = ('flip_training', 'placement_context', 'context_tta',
              'flip_context_ensemble', 'incumbent_context_ensemble')
METRICS = ('ap', 'mean_layout_ap', 'pearson', 'mae', 'ssim', 'nrms', 'hotspot_mae',
           'precision', 'recall', 'f1', 'precision_top5', 'recall_top5',
           'hotspot_fraction', 'predicted_hotspot_fraction', 'zero_map_mae')
COUNTS = ('pixels', 'true_hotspots', 'hotspots_in_top5', 'tiles_inspected',
          'layout_ap_layouts', 'pearson_layouts', 'nrms_layouts')


def collect(paths, split):
    folds = {}
    for path in paths:
        report = json.loads(path.read_text())
        if report['split'] != split or not report['calibrated']:
            raise ValueError(f'{path} must be a calibrated {split} report')
        design = report['test_design']
        expected = report['val_design'] if split == 'validation' else design
        if report['design'] != expected or design == report['val_design']:
            raise ValueError(f'{path} has inconsistent design IDs')
        if design not in folds:
            folds[design] = {**report, 'methods': {}}
        fold = folds[design]
        for field in ('design', 'val_design', 'label_threshold', 'manifest_sha256', 'layouts', 'device'):
            if fold[field] != report[field]:
                raise ValueError(f'{path} disagrees on {field} for {design}')
        for name, metrics in report['methods'].items():
            if name in fold['methods']:
                raise ValueError(f'Duplicate {name} for {design} in {path}')
            if fold['methods']:
                first = next(iter(fold['methods'].values()))
                for field in ('pixels', 'true_hotspots', 'tiles_inspected', 'layout_ap_layouts'):
                    if first[field] != metrics[field]:
                        raise ValueError(f'{name} in {path} disagrees on {field}')
            fold['methods'][name] = metrics
    return folds


def select(paths, output):
    folds = collect(paths, 'validation')
    if len(folds) < 2:
        raise ValueError('Selection needs at least two different validation folds')
    scores = {name: {metric: fmean(fold['methods'][name][metric] for fold in folds.values())
                     for metric in ('ap', 'mae')} for name in CANDIDATES}
    winner = min(scores, key=lambda name: (-scores[name]['ap'], scores[name]['mae']))
    result = dict(selected=winner, rule='highest equal-design validation AP, then lowest MAE',
                  validation_designs=[fold['val_design'] for fold in folds.values()],
                  reports={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  candidates=scores)
    with output.open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(json.dumps(result, indent=2))


def export(paths, confirmation, output):
    folds = collect(paths, 'test')
    if confirmation not in folds or len(folds) < 3:
        raise ValueError('Supply both development folds and the separate confirmation fold')
    fields = ('scope', 'test_designs', 'validation_designs', 'design_count', 'layouts', 'method',
              'forward_passes', *METRICS, 'label_threshold', 'score_threshold', *COUNTS)
    with output.open('x', newline='') as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        for design, fold in folds.items():
            for name, metrics in fold['methods'].items():
                row = {key: metrics[key] for key in (*METRICS, *COUNTS, 'forward_passes', 'score_threshold')}
                writer.writerow(dict(scope='confirmation' if design == confirmation else 'development',
                                     test_designs=design, validation_designs=fold['val_design'], design_count=1,
                                     layouts=fold['layouts'], method=name, label_threshold=fold['label_threshold'], **row))
        for scope, designs in [('development_mean', [d for d in folds if d != confirmation]),
                               ('all_design_mean', list(folds))]:
            group = [folds[d] for d in designs]
            shared = set.intersection(*(set(f['methods']) for f in group))
            for name in sorted(shared):
                values = [f['methods'][name] for f in group]
                means = {k: fmean(m[k] for m in values) if all(m[k] is not None for m in values) else None
                         for k in METRICS}
                writer.writerow(dict(scope=scope, test_designs='+'.join(designs),
                                     validation_designs='+'.join(f['val_design'] for f in group),
                                     design_count=len(designs), layouts=sum(f['layouts'] for f in group),
                                     method=name, forward_passes=values[0]['forward_passes'], **means))


def plot(paths, confirmation, selection, secondary_selection, output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    folds = collect(paths, 'test')
    winner = json.loads(selection.read_text())['selected']
    labels = {'placement_context': 'Context U-Net', 'context_tta': 'Four-view context U-Net',
              'flip_training': 'Flip-trained U-Net', 'flip_context_ensemble': 'Flip + context ensemble',
              'incumbent_context_ensemble': 'Three-model context ensemble'}
    methods = [('rudy', 'RUDY', '#858d93'), ('checkpoint_ensemble', 'Previous ensemble', '#48698a'),
               (winner, labels[winner], '#007d73')]
    if secondary_selection:
        secondary = json.loads(secondary_selection.read_text())['secondary']
        methods.append((secondary, labels[secondary] + ' (secondary)', '#ae601b'))
    designs = [d for d in folds if d != confirmation] + [confirmation]
    panels = [('ap', 'Pooled hotspot AP ↑'), ('mean_layout_ap', 'Mean within-layout AP ↑'),
              ('mae', 'Map MAE ↓')]
    fig, axes = plt.subplots(1, len(panels), figsize=(11.4, 5.0), sharey=True)
    fig.subplots_adjust(top=.71, bottom=.19, left=.17, right=.985, wspace=.17)
    for axis, (metric, title) in zip(axes, panels):
        highest = 0
        step = .7 / len(methods)
        for i, (name, _, color) in enumerate(methods):
            offset = step * (i - (len(methods)-1)/2)
            values = [folds[d]['methods'][name][metric] for d in designs]
            bars = axis.barh([j + offset for j in range(len(designs))], values, height=.82*step, color=color)
            axis.bar_label(bars, fmt='%.4f' if metric == 'mae' else '%.3f', padding=3, fontsize=8)
            highest = max(highest, max(values))
        axis.set_xlim(0, highest*1.28)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.set_title(title, loc='left', fontsize=11, fontweight='bold')
        axis.grid(axis='x', color='#dbe1e5', linewidth=.6)
        axis.set_axisbelow(True)
        axis.spines[['left', 'right', 'top']].set_visible(False)
        axis.spines['bottom'].set_color('#bbc3c9')
        axis.tick_params(length=0, labelsize=9)
    design_labels = [f"{d}\n{folds[d]['layouts']:,} layouts" + (' · confirmation' if d == confirmation else '')
                     for d in designs]
    axes[0].set_yticks(range(len(designs)), design_labels)
    axes[0].set_ylim(len(designs)-.5, -.5)
    fig.suptitle('Placement-based congestion on held-out circuit designs', x=.03, y=.97,
                 ha='left', fontsize=17, fontweight='bold')
    fig.legend(handles=[Patch(color=color, label=label) for _, label, color in methods],
               loc='upper left', bbox_to_anchor=(.025, .90), frameon=False, ncol=len(methods), fontsize=10)
    fig.text(.03, .045, 'CircuitNet-N28  ·  Three placement channels  ·  Training seed 42  ·  '
             'Hotspot labels use the threshold from each training fold.', fontsize=9, color='#49525a')
    with output.open('xb') as file:
        fig.savefig(file, format='png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('select', 'table', 'figure'))
    parser.add_argument('reports', type=Path, nargs='+')
    parser.add_argument('--confirmation-design', default='zero-riscy-a')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--selection', type=Path, help='Locked validation selection for the figure')
    parser.add_argument('--secondary-selection', type=Path, help='Registered secondary comparator for the figure')
    args = parser.parse_args()
    if args.action == 'select':
        select(args.reports, args.out)
    elif args.action == 'table':
        export(args.reports, args.confirmation_design, args.out)
    elif args.selection and args.out.suffix.lower() == '.png':
        plot(args.reports, args.confirmation_design, args.selection, args.secondary_selection, args.out)
    else:
        parser.error('figure requires --selection and a PNG output path')


if __name__ == '__main__':
    main()
