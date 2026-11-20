"""Train one design-held-out fold from explicit, audited sample manifests."""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import avg_pool2d, binary_cross_entropy_with_logits, softplus
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import ContextHotspotUNet, FCN, HotspotUNet, SpatialContextHotspotUNet, UNet
from ranking import layout_average_precisions, ranking_metrics, top5_counts


MODELS = {'fcn': FCN, 'unet': UNet, 'unet-hotspot': HotspotUNet,
          'unet-context': ContextHotspotUNet, 'unet-spatial': SpatialContextHotspotUNet}


class Maps(Dataset):
    def __init__(self, rows, root):
        self.rows, self.root = rows, root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        arrays = []
        for key, channels in [('feature', 3), ('label', 1)]:
            a = np.load(self.root / r[key], allow_pickle=False)
            if a.ndim != 3 or a.shape[-1] != channels:
                raise ValueError(f'{r[key]}: expected HWC with {channels} channels, got {a.shape}')
            if not np.isfinite(a).all() or a.min() < -1e-6 or a.max() > 1.000001:
                raise ValueError(f'{r[key]}: expected finite normalized values in [0,1]')
            arrays.append(torch.from_numpy(a.astype(np.float32).transpose(2, 0, 1).copy()))
        if arrays[0].shape[1:] != arrays[1].shape[1:]:
            raise ValueError('Feature/label spatial mismatch')
        return tuple(arrays)


def weighted_mse(pred, target, alpha, threshold):
    w = 1 + alpha * (target >= threshold).to(target.dtype)
    return (w * (pred-target).square()).sum() / w.sum()


def training_hotspot_positive_weight(rows, root, threshold):
    positives = tiles = 0
    for row in rows:
        label = np.load(root / row['label'], mmap_mode='r', allow_pickle=False)
        positives += np.count_nonzero(label >= threshold)
        tiles += label.size
    if not 0 < positives < tiles:
        raise ValueError('The hotspot head needs positive and negative training tiles')
    return (tiles - positives) / positives


def hotspot_loss(logits, target, threshold, positive_weight):
    mask = (target >= threshold).to(logits.dtype)
    weight = mask.numel() + (positive_weight - 1) * mask.sum()
    loss = binary_cross_entropy_with_logits(logits, mask, pos_weight=positive_weight, reduction='sum')
    return loss / weight


def hard_negative_ranking_loss(logits, target, threshold):
    scores = logits.flatten(1)
    positive = target.flatten(1) >= threshold
    positives = positive.sum(dim=1)
    negatives = scores.shape[1] - positives
    budget = (scores.shape[1] + 19) // 20
    hard = scores.masked_fill(positive, -torch.inf).topk(budget, dim=1, sorted=False).values
    hard = hard.masked_fill(~hard.isfinite(), 0)
    hard_mean = hard.sum(dim=1) / negatives.clamp(min=1, max=budget)
    cost = softplus(hard_mean[:, None] - scores).masked_fill(~positive, 0)
    per_layout = cost.sum(dim=1) / positives.clamp_min(1)
    valid = (positives > 0) & (negatives > 0)
    return per_layout.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)


def soft_ap_loss(logits, target, threshold):
    """Approximate pooled tile AP with triangular bins in descending score order."""
    scores, labels = logits.flatten(), target.flatten()
    if scores.numel() > 16384:
        sample = torch.randint(scores.numel(), (16384,), device=scores.device)
        scores, labels = scores[sample], labels[sample]
    centers = torch.arange(31, -1, -1, dtype=scores.dtype, device=scores.device)
    membership = (1 - (31 * scores.sigmoid()[:, None] - centers).abs()).clamp_min(0)
    positive_mass = (membership * (labels >= threshold)[:, None]).sum(dim=0)
    precision = positive_mass.cumsum(0) / membership.sum(dim=0).cumsum(0).clamp_min(1e-7)
    # A batch with no hotspots contributes zero ranking loss.
    return (positive_mass * (1 - precision)).sum() / positive_mass.sum().clamp_min(1)


def spatial_correlation(prediction, label, *, variance_floor):
    axes = (1, 2, 3)
    prediction = prediction - prediction.mean(dim=axes, keepdim=True)
    label = label - label.mean(dim=axes, keepdim=True)
    pred_variance = prediction.square().mean(dim=axes)
    label_variance = label.square().mean(dim=axes)
    covariance = (prediction * label).mean(dim=axes)
    scale = pred_variance.clamp_min(variance_floor).sqrt() * label_variance.clamp_min(variance_floor).sqrt()
    return covariance / scale, pred_variance, label_variance


def correlation_loss(prediction, label):
    correlation, _, label_variance = spatial_correlation(prediction, label, variance_floor=1e-6)
    valid = label_variance >= 1e-16
    error = (1 - correlation).masked_fill(~valid, 0)
    return error.sum() / valid.sum().clamp_min(1)


def structural_similarity(prediction, label):
    """SSIM per layout, with 7x7 valid windows, sample covariance and unit range."""
    moments = torch.cat((prediction, label, prediction.square(), label.square(), prediction * label), dim=1)
    x, y, x2, y2, xy = avg_pool2d(moments, 7, stride=1).chunk(5, dim=1)
    covariance = (xy - x * y) * (49 / 48)
    variance_sum = ((x2 - x.square()) + (y2 - y.square())) * (49 / 48)
    ssim = ((2*x*y + .01**2) * (2*covariance + .03**2)
            / ((x.square() + y.square() + .01**2) * (variance_sum + .03**2)))
    return ssim.mean(dim=(1, 2, 3))


def training_hotspot_threshold(dataset, seed, *, show_progress=False):
    # Sample only positive pixels from training labels, with a per-layout cap.
    rng = np.random.default_rng(seed)
    sampled = []
    with tqdm(dataset, desc="Scan training labels", unit="layout",
              dynamic_ncols=True, disable=not show_progress) as layouts:
        for _, label in layouts:
            values = label.numpy().ravel()
            positive = values[values > 0]
            if len(positive):
                sampled.append(rng.choice(positive, size=min(2048, len(positive)), replace=False))
    if not sampled:
        raise ValueError('Training labels contain no positive congestion')
    return float(np.quantile(np.concatenate(sampled), .9))


def select_device(requested):
    if requested == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this PyTorch environment')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('Metal/MPS is unavailable in this PyTorch environment')
    return requested


@torch.no_grad()
def evaluate(model, loader, device, threshold, *, desc="Evaluate", position=0,
             show_progress=False, include_ap=False, include_top5=False, predictor=None,
             score_threshold=None, calibrate=False, include_map_quality=False, include_layout_ap=False):
    if calibrate and (score_threshold is not None or not include_ap):
        raise ValueError('F1 calibration requires AP evaluation and an unset score threshold')
    model.eval()
    error = hotspot_error = zero_error = 0.0
    count = tp = fp = fn = 0
    hits = inspected = 0
    total_correlation = 0.0
    correlation_layouts = 0
    total_ssim = total_nrms = 0.0
    layouts = nrms_layouts = 0
    total_layout_ap = 0.0
    nonempty_layouts = 0
    score_batches, hotspot_batches = [], []
    use_hotspot_head = isinstance(model, HotspotUNet)
    if score_threshold is None:
        score_threshold = .5 if use_hotspot_head else threshold
    with tqdm(loader, desc=desc, position=position, unit="batch", leave=position == 0,
              dynamic_ncols=True, disable=not show_progress) as batches:
        for x, y in batches:
            x, y = transfer_batch(x, y, device)
            if predictor is not None:
                p, scores = predictor(model, x)
            elif use_hotspot_head:
                p, logits = model(x)
                scores = logits.sigmoid()
            else:
                p = scores = model(x)
            a, b = scores >= score_threshold, y >= threshold
            if include_layout_ap:
                per_layout_ap, nonempty = layout_average_precisions(scores, b)
                total_layout_ap += per_layout_ap.sum().item()
                nonempty_layouts += nonempty.sum().item()
            absolute_error = (p-y).abs()
            if include_map_quality:
                total_ssim += structural_similarity(p, y).sum().item()
                axes = (1, 2, 3)
                span = y.amax(dim=axes) - y.amin(dim=axes)
                valid_range = span > 0
                rmse = absolute_error.square().mean(dim=axes).sqrt()
                total_nrms += (rmse[valid_range] / span[valid_range]).sum().item()
                nrms_layouts += valid_range.sum().item()
                layouts += y.size(0)
            correlation, pred_variance, label_variance = spatial_correlation(p, y, variance_floor=1e-16)
            valid = (pred_variance >= 1e-16) & (label_variance >= 1e-16)
            total_correlation += correlation.masked_fill(~valid, 0).sum().item()
            correlation_layouts += valid.sum().item()
            error += absolute_error.sum().item()
            hotspot_error += absolute_error.masked_fill(~b, 0).sum().item()
            zero_error += y.abs().sum().item()
            count += y.numel()
            tp += (a & b).sum().item()
            fp += (a & ~b).sum().item()
            fn += (~a & b).sum().item()
            if include_ap:
                ap_device = device if device == 'cuda' else 'cpu'
                score_batches.append(scores.flatten().to(ap_device))
                hotspot_batches.append(b.flatten().to(ap_device))
            if include_top5:
                for score, mask in zip(scores.cpu().numpy(), b.cpu().numpy()):
                    layout_hits, layout_inspected = top5_counts(score, mask)
                    hits += layout_hits
                    inspected += layout_inspected
            if show_progress:
                f1_denominator = 2*tp + fp + fn
                f1 = f"{2*tp/f1_denominator:.4f}" if f1_denominator else "n/a"
                batches.set_postfix(mae=f"{error/count:.4f}", f1=f1, refresh=False)
    positives = tp + fn
    if include_ap:
        scores = torch.cat(score_batches)
        hotspots = torch.cat(hotspot_batches)
        score_batches.clear()
        hotspot_batches.clear()
        if device != 'cuda':
            scores, hotspots = scores.numpy(), hotspots.numpy()
        ranked = ranking_metrics(scores, hotspots, calibrate=calibrate)
        if calibrate:
            if ranked['ap'] is None:
                raise ValueError('Cannot calibrate on a validation split without hotspots')
            score_threshold, tp = ranked['score_threshold'], ranked['true_positives']
            fp, fn = ranked['flagged'] - tp, positives - tp
    predicted = tp + fp
    metrics = {'mae': error/count, 'precision': tp/predicted if predicted else None,
               'recall': tp/positives if positives else None,
               'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
               'hotspot_fraction': positives/count, 'predicted_hotspot_fraction': predicted/count,
               'hotspot_mae': hotspot_error/positives if positives else None,
               'zero_map_mae': zero_error/count, 'pixels': count,
               'pearson': total_correlation/correlation_layouts if correlation_layouts else None,
               'pearson_layouts': correlation_layouts,
               'score_threshold': score_threshold,
               'ranking_score': 'hotspot_probability' if use_hotspot_head else 'congestion_map'}
    if include_ap:
        metrics['ap'] = ranked['ap']
    if include_map_quality:
        metrics.update(ssim=total_ssim/layouts, nrms=total_nrms/nrms_layouts if nrms_layouts else None,
                       nrms_layouts=nrms_layouts)
    if include_layout_ap:
        metrics.update(mean_layout_ap=total_layout_ap/nonempty_layouts if nonempty_layouts else None,
                       layout_ap_layouts=nonempty_layouts)
    if include_top5:
        metrics.update(recall_top5=hits/positives if positives else None,
                       precision_top5=hits/inspected, hotspots_in_top5=hits,
                       tiles_inspected=inspected, true_hotspots=positives)
    return metrics


def transfer_batch(features, labels, device):
    on_cuda = device == 'cuda'
    memory_format = torch.channels_last if on_cuda else torch.contiguous_format
    return (features.to(device, non_blocking=on_cuda, memory_format=memory_format),
            labels.to(device, non_blocking=on_cuda))


def make_loader(rows, root, batch_size, device, workers, *, shuffle=False, generator=None):
    return DataLoader(Maps(rows, root), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, persistent_workers=workers > 0,
                      pin_memory=device == 'cuda', generator=generator)


def format_test_summary(run, config, metrics):
    def row(label, key, spec):
        value = metrics.get(key)
        if spec == '%':
            spec = '.4%' if value is not None and 0 < value < .01 else '.2%'
        display = format(value, spec) if value is not None else 'n/a'
        return f'  {label:<24} {display}'

    architecture = config.get('model', 'fcn')
    model_name = {'unet': 'U-Net', 'fcn': 'FCN', 'unet-hotspot': 'U-Net + hotspot head',
                  'unet-context': 'U-Net + placement context'}.get(
        architecture, architecture)
    details = []
    layouts = metrics.get('layouts', config.get('split_counts', {}).get('test'))
    if layouts is not None:
        details.append(f'{layouts:,} test layouts')
    if metrics.get('evaluation_device'):
        details.append(metrics['evaluation_device'].upper())

    lines = ['', f'TEST RESULTS  |  {model_name} on {config["test_design"]}']
    if details:
        lines.append('  ' + '  |  '.join(details))
    lines.extend(['-' * 56, 'Map prediction and ranking',
                  row('MAE (normalized map)', 'mae', '.5f'),
                  row('Hotspot AP', 'ap', '.4f')])
    if 'pearson' in metrics:
        lines.insert(-1, row('Spatial Pearson', 'pearson', '.4f'))
    if 'tiles_inspected' in metrics:
        found, total = metrics['hotspots_in_top5'], metrics['true_hotspots']
        lines.extend(['', 'Top 5% per layout',
                      row('Precision', 'precision_top5', '.2%'),
                      row('Recall', 'recall_top5', '.2%'),
                      f'  {"Hotspots found":<24} {found:,} / {total:,}',
                      row('Tiles inspected', 'tiles_inspected', ',d')])

    threshold = metrics.get('score_threshold', config.get('threshold'))
    heading = 'Fixed score threshold'
    if threshold is not None:
        score_name = 'hotspot probability' if metrics.get('ranking_score') == 'hotspot_probability' else 'score'
        heading += f' ({score_name} >= {threshold:.5f})'
    lines.extend(['', heading, row('Precision', 'precision', '%'),
                  row('Recall', 'recall', '%'), row('F1', 'f1', '%'),
                  row('Tiles flagged', 'predicted_hotspot_fraction', '%'),
                  '-' * 56, f'Full metrics  {run / "test_metrics.json"}'])
    return '\n'.join(lines)


def build_model(architecture, bottleneck_dilation=1):
    if architecture == 'fcn':
        return FCN()
    return MODELS[architecture](bottleneck_dilation=bottleneck_dilation)


def evaluate_test(run, rows, root, config, device, show_progress):
    model = build_model(config.get('model', 'fcn'), config.get('bottleneck_dilation', 1)).to(device)
    if device == 'cuda':
        model.to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(run/'best.pt', map_location=device, weights_only=True))
    loader = make_loader(rows, root, config['batch_size'], device, config.get('workers', 0))
    result = evaluate(model, loader, device, config['threshold'], desc="Test",
                      show_progress=show_progress, include_ap=True, include_top5=True)
    result.update(label_threshold=config['threshold'], layouts=len(rows), evaluation_device=device)
    (run/'test_metrics.json').write_text(json.dumps(result, indent=2))
    print(format_test_summary(run, config, result), flush=True)


def evaluate_saved_run(run, device, show_progress):
    config = json.loads((run/'config.json').read_text())
    history = json.loads((run/'history.json').read_text())
    if len(history) != config['epochs']:
        raise ValueError(f'Training did not finish in {run}')
    root = Path(config['manifest']).resolve().parent
    saved_root = Path(config.get('data_root', root))
    if saved_root.is_dir():
        root = saved_root
    with (run/'manifest.csv').open(newline='') as file:
        rows = [row for row in csv.DictReader(file) if row['design'] == config['test_design']]
    evaluate_test(run, rows, root, config, device, show_progress)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path)
    p.add_argument('--test-design')
    p.add_argument('--val-design')
    p.add_argument('--out', type=Path)
    p.add_argument('--model', choices=MODELS, default='fcn')
    p.add_argument('--alpha', type=float, default=0)
    p.add_argument('--hotspot-loss-weight', type=float, default=0.01,
                   help='Auxiliary classification loss weight for models with a hotspot head')
    p.add_argument('--correlation-loss-weight', type=float, default=0,
                   help='Weight of the per-layout Pearson loss on the regression map')
    p.add_argument('--ranking-loss-weight', type=float, default=0,
                   help='Weight of the hard-negative ranking loss for models with a hotspot head')
    p.add_argument('--ap-loss-weight', type=float, default=0,
                   help='Weight of the minibatch soft-histogram AP loss for a hotspot head')
    p.add_argument('--bottleneck-dilation', type=int, default=1,
                   help='Dilation of the second U-Net bottleneck convolution')
    p.add_argument('--flip-augment', action='store_true',
                   help='Randomly mirror training batches and their labels along each axis')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threshold-seed', type=int,
                   help='Optional independent training-label sampling seed for matched multi-seed comparisons')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                   help='Compute device (default: try CUDA, then Metal/MPS, then CPU)')
    p.add_argument('--select-by', choices=['mae', 'f1', 'ap'], default='mae',
                   help='Validation metric used to select the checkpoint')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--val-only', action='store_true', help='Train without evaluating the test split')
    mode.add_argument('--evaluate-run', type=Path, help='Test the saved checkpoint from a completed run')
    mode.add_argument('--show-results', type=Path, metavar='RUN_DIR',
                      help='Display a run\'s stored test results without evaluating again')
    p.add_argument('--no-progress', action='store_true', help='Hide progress bars for plain logs')
    args = p.parse_args()
    if args.show_results is not None:
        run = args.show_results
        if not (run/'config.json').is_file() or not (run/'test_metrics.json').is_file():
            p.error(f'Expected config.json and test_metrics.json in {run}')
        config = json.loads((run/'config.json').read_text())
        metrics = json.loads((run/'test_metrics.json').read_text())
        print(format_test_summary(run, config, metrics))
        return
    requested_device = args.device
    try:
        args.device = select_device(requested_device)
    except ValueError as error:
        p.error(str(error))
    show_progress = not args.no_progress
    torch.set_num_threads(4)
    if args.evaluate_run is not None:
        if any(value is not None for value in (args.manifest, args.test_design, args.val_design, args.out)):
            p.error('--evaluate-run uses the manifest and split saved in the run directory')
        evaluate_saved_run(args.evaluate_run, args.device, show_progress)
        return
    if any(value is None for value in (args.manifest, args.test_design, args.val_design, args.out)):
        p.error('Training requires --manifest, --test-design, --val-design and --out')
    if not np.isfinite(args.alpha) or args.alpha < 0:
        p.error('--alpha must be finite and nonnegative')
    if not np.isfinite(args.hotspot_loss_weight) or args.hotspot_loss_weight <= 0:
        p.error('--hotspot-loss-weight must be finite and positive')
    if not np.isfinite(args.correlation_loss_weight) or args.correlation_loss_weight < 0:
        p.error('--correlation-loss-weight must be finite and nonnegative')
    if not np.isfinite(args.ranking_loss_weight) or args.ranking_loss_weight < 0:
        p.error('--ranking-loss-weight must be finite and nonnegative')
    use_hotspot_head = issubclass(MODELS[args.model], HotspotUNet)
    if args.ranking_loss_weight and not use_hotspot_head:
        p.error('--ranking-loss-weight requires a model with a hotspot head')
    if not np.isfinite(args.ap_loss_weight) or args.ap_loss_weight < 0:
        p.error('--ap-loss-weight must be finite and nonnegative')
    if args.ap_loss_weight and not use_hotspot_head:
        p.error('--ap-loss-weight requires a model with a hotspot head')
    if args.bottleneck_dilation < 1 or (args.model == 'fcn' and args.bottleneck_dilation != 1):
        p.error('--bottleneck-dilation must be positive and is supported for U-Net only')
    if min(args.epochs, args.batch_size) < 1 or args.val_design == args.test_design:
        p.error('Require positive epochs and batch size and different validation/test designs')
    if args.workers < 0:
        p.error('--workers must be nonnegative')
    if args.threshold_seed is not None and args.threshold_seed < 0:
        p.error('--threshold-seed must be nonnegative')
    if args.out.exists():
        p.error('Output directory already exists; choose a new run directory')
    print(f'Training device: {args.device} (requested: {requested_device})',
          file=sys.stderr, flush=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    with args.manifest.open(newline='') as file:
        rows = list(csv.DictReader(file))
    root = args.manifest.resolve().parent
    for key in ('feature', 'label'):
        paths = [str((root/r[key]).resolve()) for r in rows]
        if len(paths) != len(set(paths)):
            raise ValueError(f'Duplicate {key} paths in manifest')
    test = [r for r in rows if r['design'] == args.test_design]
    val = [r for r in rows if r['design'] == args.val_design]
    train = [r for r in rows if r['design'] not in (args.test_design, args.val_design)]
    if not all([train, val, test]):
        raise ValueError('Need nonempty training, validation and test designs')
    threshold_seed = args.seed if args.threshold_seed is None else args.threshold_seed
    threshold = training_hotspot_threshold(Maps(train, root), threshold_seed, show_progress=show_progress)
    train_loader = make_loader(train, root, args.batch_size, args.device, args.workers,
                               shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    val_loader = make_loader(val, root, args.batch_size, args.device, args.workers)
    positive_weight = None
    if use_hotspot_head:
        positive_weight = training_hotspot_positive_weight(train, root, threshold)
    args.out.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ('evaluate_run', 'show_results')}
    model = build_model(args.model, args.bottleneck_dilation).to(args.device)
    if args.device == 'cuda':
        model.to(memory_format=torch.channels_last)
    config.update(device_requested=requested_device,
                  data_root=str(root), threshold=threshold,
                  hotspot_positive_weight=positive_weight, shuffle_seed=args.seed, threshold_seed=threshold_seed,
                  model_parameters=sum(parameter.numel() for parameter in model.parameters()),
                  split_counts=dict(train=len(train), val=len(val), test=len(test)),
                  train_designs=sorted({r['design'] for r in train}),
                  label_definition='per-map normalized aggregate overflow; not physical overflow units')
    (args.out/'config.json').write_text(json.dumps(config, indent=2))
    (args.out/'manifest.csv').write_text(args.manifest.read_text())
    if use_hotspot_head:
        positive_weight = torch.tensor(positive_weight, device=args.device, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    flip_rng = random.Random(args.seed)
    best_score, history = (float('inf'), float('inf')), []
    best_record = None
    with tqdm(total=args.epochs, desc="Training", position=0, unit="epoch",
              dynamic_ncols=True, disable=not show_progress) as epochs:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            total_loss = 0.0
            with tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} train",
                      position=1, unit="batch", leave=False, dynamic_ncols=True,
                      disable=not show_progress) as batches:
                for x, y in batches:
                    x, y = transfer_batch(x, y, args.device)
                    if args.flip_augment:
                        axes = [axis for axis in (-2, -1) if flip_rng.random() < 0.5]
                        if axes:
                            x, y = x.flip(axes), y.flip(axes)
                    opt.zero_grad()
                    if use_hotspot_head:
                        prediction, logits = model(x)
                        loss = weighted_mse(prediction, y, args.alpha, threshold)
                        loss = loss + args.hotspot_loss_weight * hotspot_loss(logits, y, threshold, positive_weight)
                        if args.ranking_loss_weight:
                            loss = loss + args.ranking_loss_weight * hard_negative_ranking_loss(
                                logits, y, threshold)
                        if args.ap_loss_weight:
                            loss = loss + args.ap_loss_weight * soft_ap_loss(logits, y, threshold)
                    else:
                        prediction = model(x)
                        loss = weighted_mse(prediction, y, args.alpha, threshold)
                    if args.correlation_loss_weight:
                        loss = loss + args.correlation_loss_weight * correlation_loss(prediction, y)
                    loss.backward()
                    opt.step()
                    batch_loss = loss.item()
                    losses.append(batch_loss)
                    total_loss += batch_loss
                    if show_progress:
                        batches.set_postfix(loss=f"{batch_loss:.4f}",
                                            avg=f"{total_loss/len(losses):.4f}", refresh=False)
            metrics = evaluate(model, val_loader, args.device, threshold,
                               desc=f"Epoch {epoch}/{args.epochs} validation",
                               position=1, show_progress=show_progress,
                               include_ap=args.select_by == 'ap', include_top5=args.select_by == 'ap')
            record = dict(epoch=epoch, training_loss=float(np.mean(losses)), validation=metrics)
            history.append(record)
            score = (0, metrics['mae'])
            if args.select_by in ('f1', 'ap') and metrics['hotspot_fraction'] > 0:
                score = (-metrics[args.select_by], metrics['mae'])
            if score < best_score:
                best_score, best_record = score, record
                torch.save(model.state_dict(), args.out/'best.pt')
            history_tmp = args.out/'history.json.tmp'
            history_tmp.write_text(json.dumps(history, indent=2))
            history_tmp.replace(args.out/'history.json')
            with tqdm.external_write_mode():
                print(json.dumps(record), flush=True)
            best_metric = best_record['validation'][args.select_by]
            best_display = f"{best_metric:.4f}" if best_metric is not None else 'n/a'
            epochs.set_postfix(val_mae=f"{metrics['mae']:.4f}",
                               **{f'best_{args.select_by}': best_display}, refresh=False)
            epochs.update()
    (args.out/'validation_metrics.json').write_text(json.dumps(best_record, indent=2))
    if not args.val_only:
        evaluate_test(args.out, test, root, config, args.device, show_progress)


if __name__ == '__main__':
    main()
