"""Compare hotspot training and inference methods on one saved CircuitNet fold."""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

from fusion import HOTSPOT_GRID, MAP_CONTEXT_GRID, dual_prediction, fit_fusion_weights, weighted_prediction
from train import build_model, evaluate, make_loader, select_device


# Checkpoint roles also identify exactly which validation threshold can be reused.
METHOD_ROLES = {
    'rudy': (), 'fcn': ('fcn',), 'unet': ('unet',), 'dual_head': ('baseline',),
    'pearson_loss': ('pearson',), 'hard_negative_ranking': ('ranking',),
    'flip_tta': ('pearson',), 'rudy_blend': ('pearson',),
    'checkpoint_ensemble': ('baseline', 'pearson'), 'flip_training': ('flip',),
    'placement_context': ('context',), 'context_tta': ('context',),
    'flip_context_ensemble': ('flip', 'context'),
    'incumbent_context_ensemble': ('baseline', 'pearson', 'context'),
    'validation_weighted_ensemble': ('baseline', 'pearson', 'context'),
    'validation_map_ensemble': ('baseline', 'pearson', 'context'),
    'context_without_baseline': ('pearson', 'context'),
    'context_without_pearson': ('baseline', 'context'),
    'spatial_pair': ('pearson', 'spatial'),
    'spatial_ap_pair': ('pearson', 'spatial_ap'),
    'spatial_context': ('spatial',), 'spatial_ap_context': ('spatial_ap',),
    'spatial_consensus': ('pearson', 'context', 'spatial_ap'),
    'layout_centered_pair': ('pearson', 'context'),
    'cross_head_pair': ('pearson', 'context'),
}

TUNED_METHODS = {'validation_weighted_ensemble', 'validation_map_ensemble'}
OPT_IN_METHODS = TUNED_METHODS | {'context_without_baseline', 'context_without_pearson',
                                  'spatial_pair', 'spatial_ap_pair', 'spatial_context', 'spatial_ap_context',
                                  'layout_centered_pair', 'cross_head_pair', 'spatial_consensus'}


def flip_ensemble(model, features):
    congestion, logits = model(features)
    scores = logits.sigmoid()
    for axes in ((-1,), (-2,), (-2, -1)):
        flipped_map, flipped_logits = model(features.flip(axes))
        congestion = congestion + flipped_map.flip(axes)
        scores = scores + flipped_logits.sigmoid().flip(axes)
    return congestion / 4, scores / 4


def rudy_blend(model, features):
    congestion, logits = model(features)
    rudy = features[:, 1:2]
    spatial = (-2, -1)
    rudy_std, rudy_mean = torch.std_mean(rudy, dim=spatial, correction=0, keepdim=True)
    logit_std, logit_mean = torch.std_mean(logits, dim=spatial, correction=0, keepdim=True)
    scaled_rudy = (rudy - rudy_mean) / rudy_std.clamp_min(1e-6) * logit_std + logit_mean
    return congestion, (0.75 * logits + 0.25 * scaled_rudy).sigmoid()


def rudy_only(_model, features):
    rudy = features[:, 1:2]
    return rudy, rudy


def load_network(run, config, device):
    if not (run / 'validation_metrics.json').is_file():
        raise ValueError(f'No completed validation-selected checkpoint in {run}')
    architecture = build_model(config['model'], config.get('bottleneck_dilation', 1))
    model = architecture.to(device)
    if device == 'cuda':
        model.to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(run / 'best.pt', map_location=device, weights_only=True))
    return model.eval()


def mean_pair(first, second):
    return (first[0] + second[0]) / 2, (first[1] + second[1]) / 2


def method_predictor(name, models, fusion_weights=None):
    """Return the main module, optional prediction function and inference passes."""
    def incumbent(pearson, features):
        congestion, scores = dual_prediction(pearson, features)
        _, baseline_scores = dual_prediction(models['baseline'], features)
        return congestion, (scores + baseline_scores) / 2

    def flip_context(context, features):
        return mean_pair(dual_prediction(context, features), dual_prediction(models['flip'], features))

    def incumbent_context(context, features):
        return mean_pair(dual_prediction(context, features), incumbent(models['pearson'], features))

    def spatial_consensus(spatial, features):
        pearson_map, pearson_score = dual_prediction(models['pearson'], features)
        context_map, context_score = dual_prediction(models['context'], features)
        spatial_map, spatial_score = dual_prediction(spatial, features)
        reference = (context_map + pearson_map) / 2, (2 * context_score + pearson_score) / 3
        challenger = (spatial_map + pearson_map) / 2, (2 * spatial_score + pearson_score) / 3
        return mean_pair(reference, challenger)

    def context_pair(context, features):
        # Leave one member out and preserve the original 2:1 context-to-partner ratio.
        context_map, context_score = dual_prediction(context, features)
        partner = 'baseline' if name == 'context_without_pearson' else 'pearson'
        other_map, other_score = dual_prediction(models[partner], features)
        congestion = (context_map + other_map) / 2 if partner == 'pearson' else context_map
        scores = (2 * context_score + other_score) / 3
        if name in ('layout_centered_pair', 'cross_head_pair'):
            logits = torch.logit(scores, eps=1e-6)
            axes = (-2, -1)
            if name == 'layout_centered_pair':
                scores = (logits - 0.5 * logits.mean(dim=axes, keepdim=True)).sigmoid()
            else:
                # Put the continuous regression signal on this layout's hotspot-logit scale.
                map_std, map_mean = torch.std_mean(congestion, dim=axes, correction=0, keepdim=True)
                log_std, log_mean = torch.std_mean(logits, dim=axes, correction=0, keepdim=True)
                aligned = (congestion - map_mean) / map_std.clamp_min(1e-6) * log_std + log_mean
                aligned = torch.where(map_std > 1e-6, aligned, logits)
                scores = (0.75 * logits + 0.25 * aligned).sigmoid()
        return congestion, scores

    if name == 'rudy':
        return next(iter(models.values())), rudy_only, 0
    if name in TUNED_METHODS:
        if (not isinstance(fusion_weights, dict)
                or tuple(fusion_weights.get('hotspot') or ()) not in HOTSPOT_GRID
                or fusion_weights.get('map_context') not in MAP_CONTEXT_GRID):
            raise ValueError('Expected the weights selected on validation for this three-model ensemble')
        if name == 'validation_map_ensemble':
            fusion_weights = dict(hotspot=HOTSPOT_GRID[0], map_context=fusion_weights['map_context'])
        return models['context'], lambda _, x: weighted_prediction(models, x, fusion_weights), 3
    composites = {'flip_tta': ('pearson', flip_ensemble, 4),
                  'context_tta': ('context', flip_ensemble, 4),
                  'rudy_blend': ('pearson', rudy_blend, 1),
                  'checkpoint_ensemble': ('pearson', incumbent, 2),
                  'context_without_baseline': ('context', context_pair, 2),
                  'context_without_pearson': ('context', context_pair, 2),
                  'spatial_pair': ('spatial', context_pair, 2),
                  'spatial_ap_pair': ('spatial_ap', context_pair, 2),
                  'layout_centered_pair': ('context', context_pair, 2),
                  'cross_head_pair': ('context', context_pair, 2),
                  'spatial_consensus': ('spatial_ap', spatial_consensus, 3),
                  'flip_context_ensemble': ('context', flip_context, 2),
                  'incumbent_context_ensemble': ('context', incumbent_context, 3)}
    if name in composites:
        role, function, passes = composites[name]
        return models[role], function, passes
    role, = METHOD_ROLES[name]
    return models[role], None, 1


def metric_table(methods, columns):
    columns = [(title, key, spec) for title, key, spec in columns
               if any(key in metrics for metrics in methods.values())]
    if not columns:
        return []
    show_names = len(methods) > 1
    headers = (['Method'] if show_names else []) + [title for title, _, _ in columns]
    rows = []
    for name, metrics in methods.items():
        row = [name] if show_names else []
        for _, key, spec in columns:
            value = metrics.get(key)
            row.append(format(value, spec) if value is not None else '—')
        rows.append(row)
    widths = [max(map(len, values)) for values in zip(headers, *rows)]

    def line(cells):
        return '  ' + '  '.join(cell.ljust(width) if show_names and index == 0 else cell.rjust(width)
                                for index, (cell, width) in enumerate(zip(cells, widths)))

    return [line(headers), line(['─' * width for width in widths]), *(line(row) for row in rows)]


def print_report(report, path):
    methods = report['methods']
    use_color = sys.stdout.isatty() and 'NO_COLOR' not in os.environ and os.environ.get('TERM') != 'dumb'

    def heading(text):
        return f'\033[1;36m{text}\033[0m' if use_color else text

    split = '测试' if report['split'] == 'test' else '验证'
    lines = [heading(f"{split}结果 · {report['design']}"),
             f"{report['layouts']:,} 份布局  ·  {report['device'].upper()}"]
    if len(methods) == 1:
        name, metrics = next(iter(methods.items()))
        label = '三模型融合 (spatial_consensus)' if name == 'spatial_consensus' else name
        lines.append(f"模型  {label}  ·  {metrics['forward_passes']} 次网络前向")

    detection = '固定阈值报警'
    if report.get('calibrated'):
        detection += f"  ·  由 {report['val_design']} 验证集选定阈值"
    sections = (
        ('热点排序', [('Hotspot AP ↑', 'ap', '.5f'), ('Layout AP ↑', 'mean_layout_ap', '.5f'),
                    ('P@Top5% ↑', 'precision_top5', '.2%'), ('R@Top5% ↑', 'recall_top5', '.2%')]),
        ('拥塞图质量', [('MAE ↓', 'mae', '.5f'), ('Pearson ↑', 'pearson', '.5f'),
                     ('SSIM ↑', 'ssim', '.5f'), ('NRMS ↓', 'nrms', '.5f')]),
        (detection, [('Threshold', 'score_threshold', '.5f'), ('Precision ↑', 'precision', '.2%'),
                     ('Recall ↑', 'recall', '.2%'), ('F1 ↑', 'f1', '.2%')]),
    )
    for title, columns in sections:
        table = metric_table(methods, columns)
        if table:
            lines.extend(['', heading(title), *table])
    lines.extend(['', f'完整结果  {path}'])
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ('baseline', 'pearson', 'ranking', 'flip', 'context', 'fcn', 'unet', 'spatial', 'spatial_ap'):
        parser.add_argument('--' + role.replace('_', '-'), type=Path, metavar='RUN_DIR')
    parser.add_argument('--methods', choices=METHOD_ROLES, nargs='+',
                        help='Specify the weight search or leave-one-member-out controls explicitly.')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--split', choices=['validation', 'test'])
    mode.add_argument('--show-results', type=Path, metavar='REPORT.json',
                      help='Display an existing comparison report in the terminal')
    parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
    parser.add_argument('--out', type=Path, help='New JSON report for a validation or test run')
    parser.add_argument('--raw', action='store_true', help='Print each method name and metrics JSON on one line')
    parser.add_argument('--calibrated', action='store_true',
                        help='Include map metrics and select the F1 operating point on validation')
    parser.add_argument('--thresholds-from', type=Path, metavar='VALIDATION_JSON',
                        help='Use the operating points from this validation report for the test split')
    args = parser.parse_args()
    if args.show_results is not None:
        try:
            report = json.loads(args.show_results.read_text())
            if args.raw:
                for name, metrics in report['methods'].items():
                    print(name, json.dumps(metrics))
            else:
                print_report(report, args.show_results)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            parser.error(f'Cannot display comparison report {args.show_results} ({error})')
        return
    if args.out is None:
        parser.error('--split requires --out for the new report')
    if args.out.exists():
        parser.error('Choose a new output file')

    calibrated = args.calibrated or args.thresholds_from is not None
    if calibrated and ((args.split == 'test') != (args.thresholds_from is not None)):
        parser.error('Calibrated test evaluation requires --thresholds-from on the test split only')
    architectures = dict(baseline='unet-hotspot', pearson='unet-hotspot', ranking='unet-hotspot',
                         flip='unet-hotspot', context='unet-context', fcn='fcn', unet='unet',
                         spatial='unet-spatial', spatial_ap='unet-spatial')
    runs = {role: getattr(args, role) for role in architectures if getattr(args, role)}
    if not runs:
        parser.error('Supply at least one checkpoint with the saved fold and manifest')
    available = [name for name, roles in METHOD_ROLES.items() if all(r in runs for r in roles)]
    names = args.methods or [name for name in available if name not in OPT_IN_METHODS]
    if len(set(names)) != len(names) or any(name not in available for name in names):
        parser.error(f'Provide the required checkpoints and distinct methods. Available: {available}')
    tune_fusion = bool(TUNED_METHODS.intersection(names))
    if tune_fusion and not calibrated:
        parser.error('The weighted ensemble requires --calibrated on validation or --thresholds-from on test')
    needed = set().union(*(METHOD_ROLES[name] for name in names))
    if not needed:  # RUDY only needs a saved fold and the evaluation module.
        needed.add(next(iter(runs)))
    runs = {role: path for role, path in runs.items() if role in needed}
    configs = {role: json.loads((run / 'config.json').read_text()) for role, run in runs.items()}
    anchor = next(iter(runs))
    config = configs[anchor]
    manifest_path = runs[anchor] / 'manifest.csv'
    manifest = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest).hexdigest()
    hashes = {}
    same_fold = ('threshold', 'val_design', 'test_design', 'split_counts')
    for role, run in runs.items():
        other = configs[role]
        if ((run / 'manifest.csv').read_bytes() != manifest or other['model'] != architectures[role]
                or any(other[key] != config[key] for key in same_fold)):
            parser.error(f'{role} checkpoint has an incompatible architecture, manifest or fold')
        if role in ('spatial', 'spatial_ap') and ((other.get('ap_loss_weight', 0) > 0) != (role == 'spatial_ap')):
            parser.error(f'{role} checkpoint has an incompatible AP-loss setting')
        hashes[role] = hashlib.sha256((run / 'best.pt').read_bytes()).hexdigest()

    calibration = None
    if args.thresholds_from:
        calibration = json.loads(args.thresholds_from.read_text())
        if (calibration.get('split') != 'validation' or not calibration.get('calibrated')
                or calibration.get('design') != config['val_design']
                or calibration.get('test_design') != config['test_design']
                or calibration.get('label_threshold') != config['threshold']
                or calibration.get('manifest_sha256') != manifest_hash
                or any(calibration['checkpoint_sha256'].get(role) != hashes[role] for role in needed)
                or any(name not in calibration['methods'] for name in names)):
            parser.error('Calibration report must match the validation design, manifest, methods and weights')

    torch.set_num_threads(4)
    device = select_device(args.device)
    models = {role: load_network(run, configs[role], device) for role, run in runs.items()}
    design = config['val_design' if args.split == 'validation' else 'test_design']
    with manifest_path.open(newline='') as file:
        rows = [row for row in csv.DictReader(file) if row['design'] == design]
    if not rows:
        parser.error(f'No layouts for {design}')
    data_root = Path(config['data_root'])
    if not data_root.is_dir():
        data_root = Path(config['manifest']).resolve().parent
    loader = make_loader(rows, data_root, config['batch_size'], device, config['workers'])

    fusion_weights = None
    sweep = None
    if tune_fusion:
        if calibration:
            fusion_weights = calibration.get('fusion_weights')
        else:
            fusion_weights, sweep = fit_fusion_weights(models, loader, device, config['threshold'],
                                                       tune_hotspots='validation_weighted_ensemble' in names)
    proof = ('pearson', 'ap', 'precision_top5', 'recall_top5', 'mae',
             'tiles_inspected', 'true_hotspots', 'hotspots_in_top5')
    show_progress = sys.stdout.isatty() and sys.stderr.isatty() and not args.raw
    results = {}
    for index, name in enumerate(names, 1):
        model, predictor, passes = method_predictor(name, models, fusion_weights)
        cutoff = calibration['methods'][name]['score_threshold'] if calibration else None
        result = evaluate(model, loader, device, config['threshold'], include_ap=True,
                          desc='推理' if len(names) == 1 else f'推理 {index}/{len(names)}', show_progress=show_progress,
                          include_top5=True, predictor=predictor, score_threshold=cutoff,
                          calibrate=calibrated and args.split == 'validation', include_map_quality=calibrated,
                          include_layout_ap=calibrated)
        metrics = result if calibrated else {key: result[key] for key in proof}
        if calibrated and name == 'rudy':
            metrics['ranking_score'] = 'rudy'
        results[name] = dict(forward_passes=passes, **metrics)
        if name in TUNED_METHODS:
            results[name]['hotspot_weights'] = (list(HOTSPOT_GRID[0]) if name == 'validation_map_ensemble'
                                               else fusion_weights['hotspot'])
            results[name]['map_context_weight'] = fusion_weights['map_context']
        if args.raw:
            print(name, json.dumps(results[name]), flush=True)

    report = dict(split=args.split, design=design, layouts=len(rows), device=device,
                  test_design=config['test_design'], val_design=config['val_design'],
                  label_threshold=config['threshold'],
                  manifest_sha256=manifest_hash, checkpoint_sha256=hashes, calibrated=calibrated,
                  thresholds_from=str(args.thresholds_from) if args.thresholds_from else None,
                  runs={role: str(path) for role, path in runs.items()},
                  methods=results)
    if tune_fusion:
        report['fusion_weights'] = fusion_weights
    if sweep is not None:
        report['validation_sweep'] = sweep
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    if not args.raw:
        print_report(report, args.out)


if __name__ == '__main__':
    main()
