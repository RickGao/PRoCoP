"""Small validation-only weight search for the three frozen congestion predictors."""

import torch

from ranking import average_precision
from train import transfer_batch


# Baseline, Pearson and context, in order of increasing distance from the original.
# Within a distance group the listed order breaks exact validation ties.
HOTSPOT_GRID = (
    (.25, .25, .5),
    (.3125, .3125, .375), (.1875, .1875, .625),
    (.125, .375, .5), (.375, .125, .5),
    (.125, .25, .625), (.25, .125, .625),
    (.375, .25, .375), (.25, .375, .375),
)
MAP_CONTEXT_GRID = (.5, .375, .625)


def dual_prediction(model, features):
    congestion, logits = model(features)
    return congestion, logits.sigmoid()


def component_outputs(models, features):
    _, baseline_score = dual_prediction(models['baseline'], features)
    pearson_map, pearson_score = dual_prediction(models['pearson'], features)
    context_map, context_score = dual_prediction(models['context'], features)
    return (baseline_score, pearson_score, context_score), (pearson_map, context_map)


def blend_scores(scores, weights):
    return scores[0] * weights[0] + scores[1] * weights[1] + scores[2] * weights[2]


def blend_map(maps, context_weight):
    return maps[0] * (1 - context_weight) + maps[1] * context_weight


def weighted_prediction(models, features, weights):
    scores, maps = component_outputs(models, features)
    return blend_map(maps, weights['map_context']), blend_scores(scores, weights['hotspot'])


@torch.inference_mode()
def fit_fusion_weights(models, loader, device, label_threshold, *, tune_hotspots=True):
    """Run each checkpoint once per validation batch, then score the small grid."""
    score_batches = [[], [], []]
    mask_batches = []
    map_errors = [0.0] * len(MAP_CONTEXT_GRID)
    pixels = 0
    for features, labels in loader:
        features, labels = transfer_batch(features, labels, device)
        pixels += labels.numel()
        if tune_hotspots:
            scores, maps = component_outputs(models, features)
            for i, score in enumerate(scores):
                score_batches[i].append(score.flatten().cpu())
            mask_batches.append((labels >= label_threshold).flatten().cpu())
        else:
            maps = models['pearson'](features)[0], models['context'](features)[0]
        for i, weight in enumerate(MAP_CONTEXT_GRID):
            map_errors[i] += (blend_map(maps, weight) - labels).abs().sum().item()

    map_trials = [dict(context_weight=w, mae=error / pixels)
                  for w, error in zip(MAP_CONTEXT_GRID, map_errors)]
    best_map = min(map_trials, key=lambda trial: trial['mae'])
    selected = dict(hotspot=list(HOTSPOT_GRID[0]), map_context=best_map['context_weight'])
    sweep = dict(role_order=['baseline', 'pearson', 'context'], pixels=pixels,
                 hotspot_trials=[], map_trials=map_trials)
    if not tune_hotspots:
        return selected, sweep

    # Keep the three original score maps in host memory; sort one mixture at a time.
    flat_scores = [torch.cat(batches) for batches in score_batches]
    flat_mask = torch.cat(mask_batches)
    del mask_batches, score_batches
    truth = flat_mask.to(device) if device == 'cuda' else flat_mask.numpy()
    for weights in HOTSPOT_GRID:
        mixed = blend_scores(flat_scores, weights)
        values = mixed.to(device) if device == 'cuda' else mixed.numpy()
        ap = average_precision(values, truth)
        if ap is None:
            raise ValueError('Hotspot weight selection needs positives in the validation split')
        sweep['hotspot_trials'].append(dict(weights=list(weights), ap=ap))
        del values, mixed
    best_scores = max(sweep['hotspot_trials'], key=lambda trial: trial['ap'])
    selected['hotspot'] = best_scores['weights']
    return selected, sweep
