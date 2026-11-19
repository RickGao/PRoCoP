"""Compare hotspot training and inference methods on one saved CircuitNet fold."""

import argparse
import csv
import hashlib
import json
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
}

TUNED_METHODS = {'validation_weighted_ensemble', 'validation_map_ensemble'}
OPT_IN_METHODS = TUNED_METHODS | {'context_without_baseline', 'context_without_pearson'}


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

    def context_pair(context, features):
        # Leave one member out and preserve the original 2:1 context-to-partner ratio.
        context_map, context_score = dual_prediction(context, features)
        partner = 'pearson' if name == 'context_without_baseline' else 'baseline'
        other_map, other_score = dual_prediction(models[partner], features)
        congestion = (context_map + other_map) / 2 if partner == 'pearson' else context_map
        return congestion, (2 * context_score + other_score) / 3

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
                  'flip_context_ensemble': ('context', flip_context, 2),
                  'incumbent_context_ensemble': ('context', incumbent_context, 3)}
    if name in composites:
        role, function, passes = composites[name]
        return models[role], function, passes
    role, = METHOD_ROLES[name]
    return models[role], None, 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ('baseline', 'pearson', 'ranking', 'flip', 'context', 'fcn', 'unet'):
        parser.add_argument('--' + role, type=Path, metavar='RUN_DIR')
    parser.add_argument('--methods', choices=METHOD_ROLES, nargs='+',
                        help='Specify the weight search or leave-one-member-out controls explicitly.')
    parser.add_argument('--split', choices=['validation', 'test'], required=True)
    parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--calibrated', action='store_true',
                        help='Include map metrics and select the F1 operating point on validation')
    parser.add_argument('--thresholds-from', type=Path, metavar='VALIDATION_JSON',
                        help='Use the operating points from this validation report for the test split')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Choose a new output file')

    calibrated = args.calibrated or args.thresholds_from is not None
    if calibrated and ((args.split == 'test') != (args.thresholds_from is not None)):
        parser.error('Calibrated test evaluation requires --thresholds-from on the test split only')
    architectures = dict(baseline='unet-hotspot', pearson='unet-hotspot', ranking='unet-hotspot',
                         flip='unet-hotspot', context='unet-context', fcn='fcn', unet='unet')
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
    loader = make_loader(rows, Path(config['data_root']), config['batch_size'], device, config['workers'])

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
    results = {}
    for name in names:
        model, predictor, passes = method_predictor(name, models, fusion_weights)
        cutoff = calibration['methods'][name]['score_threshold'] if calibration else None
        result = evaluate(model, loader, device, config['threshold'], include_ap=True,
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


if __name__ == '__main__':
    main()
