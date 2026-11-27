"""Render the 24 pinned held-out layouts as one 16:9 PNG per layout."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
import torch

from compare_hotspot_methods import load_network, method_predictor
from ranking import average_precision
from train import Maps

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/n28_full"
GALLERY = ROOT / "figures"
INK = "#17212a"
MUTED = "#405463"
RULE = "#98a6ae"
UNIT = Normalize(0, 1)
CONGESTION = Normalize(0, .40)
CONGESTION_CMAP = plt.get_cmap("OrRd").with_extremes(over="#67000d")
CLASSES = ListedColormap(["#eff1f2", "#007b77", "#d9771a", "#7155a4"])
CLASS_NORM = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], CLASSES.N)
CLASS_NAMES = ("Neither", "Hit", "False alarm", "Miss")


@torch.inference_mode()
def predict_layouts(design, rows):
    manifest = DATA / "manifest.csv"
    if any(row["design"] != design for row in rows):
        raise ValueError("Every layout must belong to the held-out design")
    report = json.loads((ROOT / f"runs/spatial_refinement/consensus_{design}_test.json").read_text())
    calibration = json.loads((ROOT / report["thresholds_from"]).read_text())
    assert report["split"] == "test" and report["design"] == design
    assert calibration["split"] == "validation" and calibration["test_design"] == design
    models = {}
    for role, folder in report["runs"].items():
        path = ROOT / folder
        config = json.loads((path / "config.json").read_text())
        digest = hashlib.sha256((path / "best.pt").read_bytes()).hexdigest()
        assert digest == report["checkpoint_sha256"][role] == calibration["checkpoint_sha256"][role]
        assert config["test_design"] == design and design not in config["train_designs"]
        assert config["threshold"] == report["label_threshold"] == calibration["label_threshold"]
        assert (path / "manifest.csv").read_text() == manifest.read_text()
        models[role] = load_network(path, config, "cpu")
    net, predict, _ = method_predictor("spatial_consensus", models)
    cutoff = calibration["methods"]["spatial_consensus"]["score_threshold"]
    sample = Maps(rows, DATA)
    for i, row in enumerate(sample.rows):
        features, label = sample[i]
        congestion, scores = predict(net, features[None])
        x, truth = features.numpy(), label[0].numpy()
        prediction, score = congestion[0, 0].numpy(), scores[0, 0].numpy()
        hotspot = truth >= report["label_threshold"]
        alarm = score >= cutoff
        ap = average_precision(score.ravel(), hotspot.ravel())
        hits = int((alarm & hotspot).sum())
        metrics = dict(design=design, layout=Path(row["feature"]).stem, ap=ap,
                       rudy_ap=average_precision(x[1].ravel(), hotspot.ravel()),
                       hotspots=int(hotspot.sum()), alarms=int(alarm.sum()), hits=hits, tiles=truth.size)
        yield metrics, x, prediction, truth, hotspot, alarm


def text(fig, x, top, value, size, *, bold=False, color=INK, **kw):
    w, h = fig.get_size_inches()
    return fig.text(x / w, 1-top / h, value, va="top", fontsize=size,
                    weight="bold" if bold else "normal", color=color, **kw)


def area(fig, x, top, width, height):
    w, h = fig.get_size_inches()
    return fig.add_axes((x/w, 1-(top+height)/h, width/w, height/h))


def rule(fig, x1, x2, top):
    w, h = fig.get_size_inches()
    fig.add_artist(Line2D([x1/w, x2/w], [1-top/h, 1-top/h], transform=fig.transFigure,
                          linewidth=.55, color=RULE))


def panel(fig, x, top, side, data, cmap, norm):
    ax = area(fig, x, top, side, side)
    ax.imshow(data, interpolation="none", origin="upper", cmap=cmap, norm=norm, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(.35)
        spine.set_color("#89949a")
    return ax


def color_scale(fig, x, top, width, title, cmap, norm, ticks, *, size=7, clipped=False):
    text(fig, x, top, title, size)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                       cax=area(fig, x, top+size/72+.065, width, size/100),
                       orientation="horizontal", ticks=ticks,
                       extend="max" if clipped else "neither", extendfrac=.05)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2, width=.4, pad=2, labelsize=size)


def categories(fig, positions, size):
    for (x, top), color, name in zip(positions, CLASSES.colors, CLASS_NAMES, strict=True):
        side = size/72
        swatch = area(fig, x, top+.01, side, side)
        swatch.set_facecolor(color)
        swatch.set_xticks([])
        swatch.set_yticks([])
        for spine in swatch.spines.values():
            spine.set_edgecolor(RULE)
            spine.set_linewidth(.35)
        text(fig, x+side+.047, top, name, size)


def outcomes(truth, alarm):
    return np.select([truth & alarm, ~truth & alarm, truth & ~alarm], [1, 2, 3], 0)


def slide_figure(example, title, note):
    m, features, prediction, truth, mask, alarm = example
    fig = plt.figure(figsize=(13+1/3, 7.50))
    text(fig, .16, .17, title, 25, bold=True)
    text(fig, .17, .65, f"CircuitNet-N28   /   Held-out design {m['design']}   /   Layout {m['layout'].split('-')[0]}",
         14, color=MUTED)
    rule(fig, .15, 13.18, 1.02)

    # A narrow vertical input rail leaves room for three equally sized output maps.
    input_x, input_side = .16, 1.55
    text(fig, input_x, 1.16, "Inputs", 15, bold=True)
    for i, (feature, name) in enumerate(zip(features, ["Macro region", "RUDY", "Pin RUDY"])):
        label_top = 1.47 + i * 1.86
        text(fig, input_x, label_top, name, 12)
        panel(fig, input_x, label_top + .24, input_side, feature, "Greys" if i == 0 else "Blues", UNIT)

    xs = [1.88, 5.68, 9.48]
    side, top = 3.72, 1.66
    for x, name in zip(xs, ["Ensemble prediction", "Routing ground truth", "Hotspot detection"]):
        text(fig, x, 1.16, name, 17, bold=True)
    for x, arr in zip(xs[:2], [prediction, truth]):
        panel(fig, x, top, side, arr, CONGESTION_CMAP, CONGESTION)
    panel(fig, xs[2], top, side, outcomes(mask, alarm), CLASSES, CLASS_NORM)

    color_scale(fig, xs[0], 5.53, 4.50, "Normalized overflow", CONGESTION_CMAP, CONGESTION,
                [0, .1, .2, .3, .4], size=11, clipped=True)
    categories(fig, [(xs[2], 5.57), (11.13, 5.57), (xs[2], 5.87), (11.13, 5.87)], 11.5)
    rule(fig, xs[0], 13.18, 6.24)
    text(fig, xs[0], 6.43,
         f"Layout AP   {m['rudy_ap']:.3f} (RUDY)   →   {m['ap']:.3f} (ensemble)", 16.5, bold=True)
    precision = f"{m['hits']/m['alarms']:.1%}" if m["alarms"] else "n/a"
    text(fig, 13.16, 6.45, f"Precision   {precision}      Recall   {m['hits']/m['hotspots']:.1%}",
         14, ha="right")
    text(fig, xs[0], 6.99, note, 10.5, color=MUTED)
    return fig


def render_gallery(directory):
    with (directory / "samples.csv").open(newline="") as stream:
        samples = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["design"]].append(sample)
    if len(groups) != 6 or any(len(rows) != 4 for rows in groups.values()):
        raise ValueError("Expected four layouts from each of the six held-out designs")

    for design, rows in groups.items():
        (directory / design).mkdir(exist_ok=True)
        for row, example in zip(rows, predict_layouts(design, rows), strict=True):
            metrics = example[0]
            if metrics["layout"] != row["layout"] or abs(metrics["ap"] - float(row["ap"])) > 1e-5:
                raise ValueError(f"Layout or hotspot AP differs for {row['layout']}")
            note = (f"Within-design sample {row['percentile']}   /   "
                    f"True hotspots: {metrics['hotspots'] / metrics['tiles']:.1%} of tiles")
            fig = slide_figure(example, "Placement-based routing congestion", note)
            fig.savefig(directory / row["image"], format="png", dpi=216)
            plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(3)
    with plt.rc_context({"font.family": "DejaVu Sans", "figure.facecolor": "white"}):
        render_gallery(GALLERY)
