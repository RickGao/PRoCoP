"""Compare RUDY and saved score maps on a shared design-held-out split."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ranking import average_precision, top5_counts
from train import Maps, training_hotspot_threshold


def evaluate_maps(dataset, predictions, threshold, show_progress):
    # Reserve one float32 score per tile for exact AP, with no score quantization.
    methods = {'RUDY': None, **predictions}
    score_chunks = {name: [] for name in methods}
    counts = {name: dict(hits=0, inspected=0, error=0.0) for name in methods}
    hotspot_chunks = []
    with tqdm(range(len(dataset)), desc='Evaluate layouts', unit='layout',
              disable=not show_progress) as layouts:
        for index in layouts:
            features, label = dataset[index]
            label = label.numpy()[0]
            hotspots = label >= threshold
            hotspot_chunks.append(hotspots.ravel())
            filename = Path(dataset.rows[index]['label']).name
            for name, directory in methods.items():
                if directory is None:
                    scores = features.numpy()[1]
                else:
                    path = directory / filename
                    scores = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
                    if scores.shape == (*label.shape, 1):
                        scores = scores[..., 0]
                    if scores.shape != label.shape or not np.isfinite(scores).all():
                        raise ValueError(f'{path}: expected finite scores with shape {label.shape}')
                hits, inspected = top5_counts(scores, hotspots)
                count = counts[name]
                count['hits'] += hits
                count['inspected'] += inspected
                count['error'] += float(np.abs(scores - label).sum(dtype=np.float64))
                score_chunks[name].append(scores.ravel().copy())

    hotspots = np.concatenate(hotspot_chunks)
    positives = int(np.count_nonzero(hotspots))
    results = {}
    for name, count in counts.items():
        scores = np.concatenate(score_chunks.pop(name))
        results[name] = {
            'ap': average_precision(scores, hotspots),
            'mae': count['error'] / hotspots.size,
            'recall_top5': count['hits'] / positives if positives else None,
            'precision_top5': count['hits'] / count['inspected'],
            'hotspots_in_top5': count['hits'],
            'tiles_inspected': count['inspected'],
        }
    return dict(layouts=len(dataset), tiles=hotspots.size, true_hotspots=positives, methods=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--test-design', required=True)
    parser.add_argument('--val-design', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, help='Reuse a saved training-derived label threshold')
    parser.add_argument('--prediction', action='append', default=[], metavar='METHOD=DIR',
                        help='Layout score arrays named after the manifest label files')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args()

    if args.test_design == args.val_design:
        parser.error('The validation and test designs must differ')
    if args.out.exists():
        parser.error('Choose a new output file')
    if args.threshold is not None and not 0 < args.threshold <= 1:
        parser.error('--threshold must be in (0, 1]')

    predictions = {}
    for assignment in args.prediction:
        name, separator, directory = assignment.partition('=')
        if not (name and separator and directory) or name == 'RUDY' or name in predictions:
            parser.error('Use a distinct METHOD=DIR for each additional predictor')
        predictions[name] = Path(directory)

    root = args.manifest.resolve().parent
    with args.manifest.open(newline='') as file:
        rows = list(csv.DictReader(file))
    train = [row for row in rows if row['design'] not in (args.test_design, args.val_design)]
    val = [row for row in rows if row['design'] == args.val_design]
    test = [row for row in rows if row['design'] == args.test_design]
    if not all((train, val, test)):
        parser.error('The train, validation, and test partitions must all have layouts')

    threshold = args.threshold
    if threshold is None:
        threshold = training_hotspot_threshold(Maps(train, root), args.seed,
                                              show_progress=not args.no_progress)
    report = dict(test_design=args.test_design, val_design=args.val_design,
                  train_designs=sorted({row['design'] for row in train}),
                  train_layouts=len(train), val_layouts=len(val), label_threshold=threshold,
                  threshold_source='supplied' if args.threshold is not None else 'training_labels',
                  seed=args.seed, manifest=str(args.manifest),
                  aggregation='pooled tile counts within design, with a separate budget per layout',
                  budget='ceil(5% of tiles); ties by row-major index')
    report.update(evaluate_maps(Maps(test, root), predictions, threshold, not args.no_progress))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
