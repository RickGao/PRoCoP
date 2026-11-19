"""Pixel ranking metrics for normalized congestion maps."""

import numpy as np
import torch


def top5_counts(scores, hotspots):
    """Count hits among ceil(5%) of a layout's tiles, breaking ties in grid order."""
    scores = scores.ravel()
    hotspots = hotspots.ravel()
    budget = (scores.size + 19) // 20
    rank = scores.size - budget
    cutoff = np.partition(scores, rank)[rank]
    above = scores > cutoff
    remaining = budget - np.count_nonzero(above)
    tied = np.flatnonzero(scores == cutoff)[:remaining]
    hits = np.count_nonzero(hotspots & above) + np.count_nonzero(hotspots[tied])
    return int(hits), int(budget)


def ranking_metrics(scores, hotspots, *, calibrate=False):
    """Pooled AP and optional F1 operating point. Tied scores enter together."""
    if isinstance(scores, torch.Tensor):
        ranked, order = torch.sort(scores, descending=True)
        ordered = hotspots[order]
        ends = torch.ones_like(ordered)
        ends[:-1] = ranked[:-1] != ranked[1:]
        ranks = torch.where(ends)[0]
        true_positives = ordered.cumsum(dim=0, dtype=torch.int32)[ranks]
        added = torch.diff(true_positives, prepend=true_positives.new_zeros(1))
        positives = int(true_positives[-1].item())
        if not positives:
            return {'ap': None}
        ap = ((added / positives) * (true_positives / (ranks + 1))).sum()
        result = {'ap': ap.item()}
        if calibrate:
            # Descending groups give the higher threshold when F1 is tied.
            best = (2 * true_positives.double() / (positives + ranks + 1)).argmax()
            result.update(score_threshold=ranked[ranks[best]].item(),
                          true_positives=true_positives[best].item(), flagged=ranks[best].item() + 1)
        return result

    positives = np.count_nonzero(hotspots)
    if not positives:
        return {'ap': None}
    order = np.argsort(scores, kind='stable')[::-1]
    ranked = scores[order]
    ends = np.flatnonzero(ranked[:-1] != ranked[1:])
    ends = np.append(ends, ranked.size - 1)
    true_positives = np.cumsum(hotspots[order], dtype=np.int64)[ends]
    added = np.diff(true_positives, prepend=0)
    result = {'ap': float(np.dot(added / positives, true_positives / (ends + 1)))}
    if calibrate:
        best = np.argmax(2 * true_positives / (positives + ends + 1))
        result.update(score_threshold=float(ranked[ends[best]]),
                      true_positives=int(true_positives[best]), flagged=int(ends[best]) + 1)
    return result


def average_precision(scores, hotspots):
    """Compute pooled, non-interpolated AP with all tied scores grouped."""
    return ranking_metrics(scores, hotspots)['ap']


def layout_average_precisions(scores, hotspots):
    """Exact AP for each layout in a batch and a mask of layouts with positives."""
    ranked, order = torch.sort(scores.flatten(1), dim=1, descending=True)
    positives = hotspots.flatten(1).gather(1, order).cumsum(1, dtype=torch.int32)
    ends = torch.ones_like(ranked, dtype=torch.bool)
    ends[:, :-1] = ranked[:, :-1] != ranked[:, 1:]
    complete = positives.masked_fill(~ends, 0).cummax(dim=1).values
    added = torch.diff(complete, dim=1, prepend=torch.zeros_like(complete[:, :1]))
    k = torch.arange(1, ranked.size(1) + 1, device=scores.device)
    total = positives[:, -1]
    ap = (added * (positives / k)).sum(dim=1) / total.clamp_min(1)
    return ap, total > 0
