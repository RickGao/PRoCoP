"""Measure where the two locked predictors gain or lose on the outer test design."""

import csv
import json
from pathlib import Path

import torch

from compare_hotspot_methods import load_network, method_predictor
from fusion import dual_prediction
from ranking import layout_average_precisions, ranking_metrics
from train import make_loader, transfer_batch


@torch.inference_mode()
def audit_outer(root, report_path, output):
    report = json.loads((root / report_path).read_text())
    if report["split"] != "test" or not report["calibrated"] or len(report["methods"]) != 2:
        raise ValueError("Expected a paired test report with validation-selected thresholds")
    device = report["device"]
    paths = {k: root / p for k, p in report["runs"].items()}
    configs = {k: json.loads((p / "config.json").read_text()) for k, p in paths.items()}
    models = {k: load_network(p, configs[k], device) for k, p in paths.items()}
    anchor = next(iter(paths))
    cfg = configs[anchor]
    with (paths[anchor] / "manifest.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["design"] == report["design"]]
    data_root = Path(cfg["data_root"])
    if not data_root.is_dir():
        data_root = (root / cfg["manifest"]).resolve().parent
    loader = make_loader(rows, data_root, cfg["batch_size"], device, cfg["workers"])
    bins = ("0–7", "8–31", "32–63", "64+")
    by_method, layout_aps = {}, {}
    for name, original in report["methods"].items():
        model, predictor, _ = method_predictor(name, models)
        truth_chunks = [[] for _ in bins]
        score_chunks = [[] for _ in bins]
        aps = []
        totals = torch.zeros(len(bins), 3, dtype=torch.float64, device=device)
        for inputs, labels in loader:
            inputs, labels = transfer_batch(inputs, labels, device)
            predicted, scores = (predictor or dual_prediction)(model, inputs)
            truth = labels >= report["label_threshold"]
            per_layout, nonempty = layout_average_precisions(scores, truth)
            aps.append(torch.where(nonempty, per_layout, torch.nan).cpu())
            h, w = labels.shape[-2:]
            y, x = torch.arange(h, device=device), torch.arange(w, device=device)
            distance = torch.minimum(torch.minimum(y, h - 1 - y)[:, None], torch.minimum(x, w - 1 - x)[None, :])
            zones = (distance < 8, (distance >= 8) & (distance < 32),
                     (distance >= 32) & (distance < 64), distance >= 64)
            for i, zone in enumerate(zones):
                tile_scores, hot = scores[..., zone].flatten(), truth[..., zone].flatten()
                flagged = tile_scores >= original["score_threshold"]
                truth_chunks[i].append(hot)
                score_chunks[i].append(tile_scores)
                totals[i, 0] += (hot & flagged).sum()
                totals[i, 1] += flagged.sum()
                totals[i, 2] += (predicted[..., zone] - labels[..., zone]).abs().sum()
        regions = {}
        for i, area in enumerate(bins):
            truth = torch.cat(truth_chunks[i])
            scores = torch.cat(score_chunks[i])
            positives, count = int(truth.sum()), truth.numel()
            tp, flagged, error = totals[i].tolist()
            regions[area] = dict(pixels=count, true_hotspots=positives, flagged=int(flagged), hits=int(tp),
                                 ap=ranking_metrics(scores, truth)["ap"] if count else None,
                                 mae=error / count if count else None,
                                 recall=tp / positives if positives else None,
                                 false_positive_rate=(flagged - tp) / (count - positives) if count > positives else None)
        if sum(r["true_hotspots"] for r in regions.values()) != original["true_hotspots"]:
            raise ValueError("Region counts disagree with the paired evaluation")
        by_method[name] = regions
        layout_aps[name] = torch.cat(aps)
        mean_ap = original["mean_layout_ap"]
        if mean_ap is not None and abs(float(layout_aps[name].double().nanmean()) - mean_ap) > 1e-6:
            raise ValueError("Layout AP disagrees with the paired evaluation")
    reference, challenger = report["methods"]
    delta = layout_aps[challenger] - layout_aps[reference]
    delta = delta[delta.isfinite()]
    summary = dict(source=str(report_path), design=report["design"], layouts=len(rows),
                   distance_unit="256x256 grid cells from the closest die edge",
                   label_threshold=report["label_threshold"], regions=by_method, challenger=challenger,
                   paired_layouts=delta.numel(), improved=int((delta > 0).sum()), tied=int((delta == 0).sum()),
                   declined=int((delta < 0).sum()), median_layout_ap_delta=float(torch.quantile(delta, .5)) if delta.numel() else None)
    (root / output).write_text(json.dumps(summary, indent=2) + "\n")
