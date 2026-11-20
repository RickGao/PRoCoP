"""Screen spatial and score refinements without using the outer test design, then replay six folds."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from statistics import fmean

from summarize_study import COUNTS, METRICS, collect


ROOT = Path(__file__).resolve().parent
OUT = Path("runs/spatial_refinement")
BASE = "context_without_baseline"
CANDIDATES = ("spatial_pair", "spatial_ap_pair", "layout_centered_pair", "cross_head_pair")
OUTER = "RISCY-FPU-b"
SCREENS = {"selection_b": "RISCY-b", "selection_a": "zero-riscy-a"}
HISTORICAL = {"zero-riscy-b": "RISCY-FPU-b", "RISCY-a": "RISCY-FPU-a",
              "zero-riscy-a": "RISCY-b", "RISCY-b": "zero-riscy-b"}
SECONDARY_FOLDS = ("RISCY-a", "zero-riscy-a", "RISCY-b", "RISCY-FPU-a")
ROLES = {"pearson": ["--model", "unet-hotspot"], "context": ["--model", "unet-context"],
         "spatial": ["--model", "unet-spatial"],
         "spatial_ap": ["--model", "unet-spatial", "--ap-loss-weight", "0.005"]}
PLAIN = ("fcn", "unet")
PLAIN_ARCHIVES = {
    "zero-riscy-b": "runs/full_zero-riscy-b_{}_ap_cuda",
    "RISCY-FPU-a": "runs/full_RISCY-FPU-a_{}_ap_cuda",
    "RISCY-a": "runs/context_study/RISCY-a_{}",
    "zero-riscy-a": "runs/model_comparison/zero-riscy-a_{}",
    "RISCY-b": "runs/model_comparison/RISCY-b_{}",
}


def read(path):
    return json.loads((ROOT / path).read_text())


def save_once(path, data):
    with (ROOT / path).open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def run(args, log, *, output_dir=OUT):
    print(f"Running {log}", flush=True)
    with (ROOT / output_dir / (log + ".log")).open("a") as stream:
        subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def train(test, validation, prefix, role, seed=42, archive=None):
    stem = f"{prefix}_{role}"
    folder = OUT / stem
    options = ["--model", role] if role in PLAIN else ROLES[role]
    corr_weight = 0 if role in PLAIN else 0.005
    if archive and all((ROOT / archive / f).is_file() for f in ("best.pt", "validation_metrics.json")):
        folder = archive
    else:
        # Keep interrupted experiments and place a retry in a fresh directory.
        attempt = 1
        while (ROOT / folder).exists() and not (ROOT / folder / "validation_metrics.json").is_file():
            attempt += 1
            folder = OUT / f"{stem}_attempt{attempt}"
    if not (ROOT / folder).exists():
        run(["train.py", "--manifest", "data/n28_full/manifest.csv", "--test-design", test,
             "--val-design", validation, "--out", folder, *options, "--correlation-loss-weight", corr_weight,
             "--epochs", 15, "--lr", "0.002", "--batch-size", 4, "--workers", 4, "--seed", seed, "--threshold-seed", 42,
             "--select-by", "ap", "--device", "cuda", "--val-only", "--no-progress"], folder.name)
    cfg = read(folder / "config.json")
    expected = dict(test_design=test, val_design=validation, seed=seed, model=options[1],
                    epochs=15, batch_size=4, lr=0.002, alpha=0, select_by="ap")
    if (any(cfg.get(k) != v for k, v in expected.items())
            or cfg.get("correlation_loss_weight", 0) != corr_weight or cfg.get("flip_augment", False)
            or cfg.get("threshold_seed", cfg["seed"]) != 42
            or cfg.get("ap_loss_weight", 0) != (0.005 if role == "spatial_ap" else 0)
            or not (ROOT / folder / "best.pt").is_file()):
        raise ValueError(f"Incompatible training run in {folder}")
    return folder


def evaluate(prefix, split, paths, names, calibration=None, *, output_dir=OUT):
    dest = output_dir / f"{prefix}_{split}.json"
    if not (ROOT / dest).exists():
        args = ["compare_hotspot_methods.py", "--device", "cuda", "--split", split,
                "--out", dest, "--methods", *names]
        for role, path in paths.items():
            args.extend(["--" + role.replace("_", "-"), path])
        args.extend(["--thresholds-from", calibration] if calibration else ["--calibrated"])
        run(args, dest.stem, output_dir=output_dir)
    report = read(dest)
    expected_cutoff = str(calibration) if calibration else None
    if (report["split"] != split or not report["calibrated"] or set(report["methods"]) != set(names)
            or report["thresholds_from"] != expected_cutoff):
        raise ValueError(f"Incompatible evaluation report in {dest}")
    for role, folder in report["runs"].items():
        if (paths.get(role) != Path(folder)
                or report["checkpoint_sha256"][role] != hashlib.sha256((ROOT / folder / "best.pt").read_bytes()).hexdigest()):
            raise ValueError(f"Checkpoint mismatch for {role} in {dest}")
    return dest


def screen():
    reports = []
    for prefix, val in SCREENS.items():
        paths = {role: train(OUTER, val, prefix, role) for role in ROLES}
        reports.append(evaluate(prefix, "validation", paths, (BASE, *CANDIDATES)))
    contents = [read(p) for p in reports]
    for report, val in zip(contents, SCREENS.values()):
        if (report["split"], report["design"], report["test_design"]) != ("validation", val, OUTER):
            raise ValueError("Unexpected selection split")
    metrics = {name: {metric: fmean(r["methods"][name][metric] for r in contents)
                      for metric in ("ap", "mean_layout_ap", "mae", "pearson", "f1")}
               for name in (BASE, *CANDIDATES)}
    best = min(CANDIDATES, key=lambda n: (-metrics[n]["ap"], metrics[n]["mae"], CANDIDATES.index(n)))
    decision = dict(challenger=best, incumbent=BASE, outer_test=OUTER, outer_validation=SCREENS["selection_b"],
                    selection_rule="highest equal-design validation AP among challengers, then lowest MAE and listed order",
                    beats_validation_incumbent=metrics[best]["ap"] > metrics[BASE]["ap"], validation_means=metrics,
                    validation_reports={str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in reports})
    lock = OUT / "selection.json"
    if (ROOT / lock).exists():
        if decision != read(lock):
            raise ValueError("Existing selection disagrees with the two validation reports")
    else:
        save_once(lock, decision)
    print(f"Locked {best} on validation. Mean AP {metrics[best]['ap']:.6f}, incumbent {metrics[BASE]['ap']:.6f}.", flush=True)
    return best


def paired_tests(challenger):
    reports = []
    role = {"spatial_pair": "spatial", "spatial_ap_pair": "spatial_ap"}.get(challenger)
    for test in (OUTER, *HISTORICAL, "RISCY-FPU-a"):
        if test == OUTER:
            prefix = "selection_b"
            paths = {k: train(test, SCREENS[prefix], prefix, k) for k in ("pearson", "context")}
            if role:
                paths[role] = train(test, SCREENS[prefix], prefix, role)
            calibration = OUT / "selection_b_validation.json"
        else:
            val = HISTORICAL.get(test, "RISCY-b")
            archive = Path("runs/fusion_refinement") / f"{test}_pairs_validation.json"
            paths = {}
            if test in HISTORICAL and (ROOT / archive).is_file():
                archived = read(archive)
                if archived["val_design"] != val or archived["test_design"] != test:
                    raise ValueError(f"Wrong historical fold in {archive}")
                for k in ("pearson", "context"):
                    folder = Path(archived["runs"][k])
                    checkpoint = ROOT / folder / "best.pt"
                    if checkpoint.is_file() and (ROOT / folder / "validation_metrics.json").is_file():
                        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != archived["checkpoint_sha256"][k]:
                            raise ValueError(f"Historical {k} checkpoint differs from {archive}")
                        paths[k] = folder
            for k in ("pearson", "context"):
                if k not in paths:
                    paths[k] = train(test, val, test, k)
            if role:
                paths[role] = train(test, val, test, role)
            calibration = evaluate(test, "validation", paths, (BASE, challenger))
        reports.append(evaluate(test, "test", paths, (BASE, challenger), calibration))
    return reports


def export(reports, methods, filename):
    folds = collect([ROOT / p for p in reports], "test")
    groupings = {"historical_four": HISTORICAL, "two_fpu": (OUTER, "RISCY-FPU-a"), "all_six": tuple(folds)}
    if "spatial_consensus" in methods:
        groupings["secondary_registered_four"] = SECONDARY_FOLDS
    fields = ("scope", "design", "validation", "layouts", "method", *METRICS, "label_threshold",
              "score_threshold", *COUNTS, "source")
    output = ROOT / OUT / filename
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for path in reports:
            report = read(path)
            for method, values in report["methods"].items():
                writer.writerow(dict(scope="test", design=report["design"], validation=report["val_design"],
                                     layouts=report["layouts"], method=method, label_threshold=report["label_threshold"],
                                     source=str(path), **{k: values[k] for k in (*METRICS, "score_threshold", *COUNTS)}))
        for name, designs in groupings.items():
            for method in methods:
                values = [folds[d]["methods"][method] for d in designs]
                means = {k: fmean(v[k] for v in values) if all(v[k] is not None for v in values) else None
                         for k in METRICS}
                writer.writerow(dict(scope=name, design="+".join(designs), layouts=sum(folds[d]["layouts"] for d in designs),
                                     method=method, **means,
                                     **{k: sum(v[k] for v in values) for k in COUNTS}))
    temporary.replace(output)
    return output


def single_model_ablation(challenger):
    role = {"spatial_pair": "spatial", "spatial_ap_pair": "spatial_ap"}.get(challenger)
    if not role:
        return  # The two score-only alternatives have the same underlying networks as the incumbent.
    methods = ("placement_context", role + "_context")
    reports = []
    for test in (OUTER, *HISTORICAL, "RISCY-FPU-a"):
        pair = read(OUT / f"{test}_test.json")
        paths = {k: Path(pair["runs"][k]) for k in ("context", role)}
        prefix = f"single_{test}"
        val = evaluate(prefix, "validation", paths, methods)
        reports.append(evaluate(prefix, "test", paths, methods, val))
    export(reports, methods, "single_models.csv")


def replicate(challenger):
    prefix = "outer_seed43"
    roles = ["pearson", "context"]
    role = {"spatial_pair": "spatial", "spatial_ap_pair": "spatial_ap"}.get(challenger)
    if role:
        roles.append(role)
    paths = {r: train(OUTER, SCREENS["selection_b"], prefix, r, seed=43) for r in roles}
    calibration = evaluate(prefix, "validation", paths, (BASE, challenger))
    reference = read(OUT / f"{OUTER}_test.json")
    if read(calibration)["label_threshold"] != reference["label_threshold"]:
        raise ValueError("Seed replication changed the ground-truth threshold")
    evaluate(prefix, "test", paths, (BASE, challenger), calibration)


def secondary_fusion(challenger):
    if challenger != "spatial_ap_pair":
        return
    method = "spatial_consensus"
    reports = []
    # The alternate screening split contributes only a validation score.
    for case in ("selection_a", OUTER, *HISTORICAL, "RISCY-FPU-a", "outer_seed43"):
        val_only = case == "selection_a"
        src = OUT / (case + ("_validation.json" if val_only else "_test.json"))
        data = read(src)
        paths = {role: Path(data["runs"][role]) for role in ("pearson", "context", "spatial_ap")}
        prefix = f"consensus_{case}"
        val = evaluate(prefix, "validation", paths, (method,))
        if not val_only:
            test = evaluate(prefix, "test", paths, (method,), val)
            if case != "outer_seed43":
                reports.extend([src, test])
    export(reports, (BASE, challenger, method), "secondary_fusion.csv")


def compare_with_baselines(challenger):
    methods = ["rudy", *PLAIN, BASE, challenger]
    single = {"spatial_pair": "spatial_context", "spatial_ap_pair": "spatial_ap_context"}.get(challenger)
    if single:
        methods.extend(("placement_context", single))
    if challenger == "spatial_ap_pair":
        methods.append("spatial_consensus")
    reports = []
    for design in (OUTER, *HISTORICAL, "RISCY-FPU-a"):
        pair_path = OUT / f"{design}_test.json"
        val = read(pair_path)["val_design"]
        template = PLAIN_ARCHIVES.get(design)
        paths = {role: train(design, val, design, role,
                             archive=Path(template.format(role)) if template else None) for role in PLAIN}
        prefix = "baselines_" + design
        calibration = evaluate(prefix, "validation", paths, ("rudy", *PLAIN))
        reports.extend((evaluate(prefix, "test", paths, ("rudy", *PLAIN), calibration), pair_path))
        if single:
            reports.append(OUT / f"single_{design}_test.json")
        if challenger == "spatial_ap_pair":
            reports.append(OUT / f"consensus_{design}_test.json")
    return export(reports, methods, "comparison_with_baselines.csv")


def print_summary(path):
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["scope"] == "all_six"]
    names = {"rudy": "RUDY", "fcn": "FCN (MSE)", "unet": "U-Net (MSE)",
             BASE: "Pearson + context", "placement_context": "Context U-Net",
             "spatial_ap_context": "Spatial + AP U-Net", "spatial_ap_pair": "Pearson + spatial AP",
             "spatial_consensus": "Pearson + both contexts"}
    print(f"\nSix held-out designs, {int(rows[0]['layouts']):,} layouts, equal design weights")
    print(f"{'Model':<27}{'Pooled AP':>11}{'Layout AP':>11}{'MAE':>11}{'Pearson':>11}{'F1':>11}")
    for row in sorted(rows, key=lambda r: float(r["ap"]), reverse=True):
        label = names.get(row["method"], row["method"])
        values = [f"{float(row[k]):.5f}" if row[k] else "n/a"
                  for k in ("ap", "mean_layout_ap", "mae", "pearson", "f1")]
        print(f"{label:<27}" + "".join(f"{v:>11}" for v in values))
    print(f"Per-design metrics and source reports in {path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "test", "followup", "all", "report"), default="all")
    args = parser.parse_args()
    if args.stage == "report":
        table = ROOT / OUT / "comparison_with_baselines.csv"
        if not table.is_file():
            parser.error("Run --stage all to generate the six-design comparison table")
        print_summary(table)
        return
    (ROOT / OUT).mkdir(parents=True, exist_ok=True)
    if args.stage in ("test", "followup") and not (ROOT / OUT / "selection.json").is_file():
        parser.error("Complete validation screening before testing a challenger")
    if args.stage == "followup" and not (ROOT / OUT / "comparison.csv").is_file():
        parser.error("Complete the six paired folds before the ablation and independent seed replication")
    best = screen()
    if args.stage in ("all", "test"):
        primary = export(paired_tests(best), (BASE, best), "comparison.csv")
        if args.stage == "test":
            print_summary(primary)
    if args.stage in ("all", "followup"):
        single_model_ablation(best)
        replicate(best)
        secondary_fusion(best)
        from spatial_diagnostics import audit_outer
        audit_outer(ROOT, OUT / f"{OUTER}_test.json", OUT / "outer_regions.json")
        print_summary(compare_with_baselines(best))


if __name__ == "__main__":
    main()
