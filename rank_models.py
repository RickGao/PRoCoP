"""Complete and rank the project's archived congestion predictors on common design splits."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from functools import cache
from pathlib import Path
from statistics import fmean

from run_spatial_study import HISTORICAL, ROOT, evaluate, read, run


OUT = Path("runs/rankings")
FOUR = Path("runs/model_comparison/comparison.csv")
SIX = Path("runs/spatial_refinement/comparison_with_baselines.csv")
FPU_DESIGNS = ("RISCY-FPU-b", "RISCY-FPU-a")
DESIGNS = (*HISTORICAL, *FPU_DESIGNS)
ROLE_REFERENCES = {
    "baseline": "runs/fusion_refinement/RISCY-b_baseline",
    "ranking": "runs/model_comparison/RISCY-b_ranking",
    "flip": "runs/model_comparison/RISCY-b_flip",
}
ARCHIVES = {
    "terminal_unet": ("早期终端单头 U-Net", "runs/full_zerob_unet_terminal", "unet"),
    "rerun_unet": ("配对重跑单头 U-Net", "runs/zerob_unet_head_compare_unet_seed42", "unet"),
    "unet_alpha0": ("单头 U-Net 权重扫描 α=0", "runs/zerob_unet_weight_sweep_alpha0", "unet"),
    "unet_alpha1": ("单头 U-Net 加权 MSE α=1", "runs/zerob_unet_weight_sweep_alpha1", "unet"),
    "unet_alpha3": ("单头 U-Net 加权 MSE α=3", "runs/zerob_unet_weight_sweep_alpha3", "unet"),
    "dilation3": ("双头 U-Net，瓶颈空洞率 3", "runs/correlation_iterations/dev_dilation3_seed42", "baseline"),
}
PILOTS = ("fold_zerob_baseline", "fold_zerob_weighted", "pilot_zerob_mae_alpha0",
          "pilot_zerob_f1_alpha0", "pilot_zerob_f1_alpha1", "pilot_zerob_f1_alpha3", "pilot_zerob_f1_alpha5")
SCREEN_ONLY = ("spatial_pair", "spatial_context", "layout_centered_pair", "cross_head_pair")
NAMES = {
    "spatial_consensus": "新三模型（Pearson + 原上下文 + 空间 AP）",
    "context_without_baseline": "原双模型（Pearson + 原上下文）",
    "incumbent_context_ensemble": "旧三模型（固定权重）",
    "validation_map_ensemble": "旧三模型（仅调回归权重）",
    "validation_weighted_ensemble": "旧三模型（验证集调热点和回归权重）",
    "spatial_ap_pair": "Pearson 双头 + 空间 AP 上下文",
    "spatial_ap_context": "空间 AP 上下文双头",
    "spatial_pair": "Pearson 双头 + 空间上下文（无 AP 损失）",
    "spatial_context": "空间上下文单网（无 AP 损失）",
    "layout_centered_pair": "原双模型 + 布局分数居中",
    "cross_head_pair": "原双模型 + 回归与热点分数融合",
}
METRICS = ("ap", "mean_layout_ap", "mae", "pearson", "f1", "ssim", "nrms",
           "precision", "recall", "precision_top5", "recall_top5")
RANKED = ("ap", "mean_layout_ap", "mae", "pearson", "f1", "ssim", "nrms", "pr_auc", "spearman")
COUNTS = ("pixels", "true_hotspots", "layout_ap_layouts")
SCOPES = {
    "six_designs": ("六个测试电路，10,242 份布局", "ap"),
    "four_designs": ("原四个测试电路，7,025 份布局", "ap"),
    "zero_b_joint": ("zero-riscy-b，本地与论文，1,122 份布局", "pearson"),
    "fpu_b_including_screen": ("RISCY-FPU-b，含开发阶段方案，1,248 份布局", "ap"),
    "fpu_b_seed43": ("RISCY-FPU-b，第二个训练种子", "ap"),
    "pilot32": ("早期 FCN，32 份测试布局", "ap"),
    "mac96": ("Mac 试验，每个设计 96 份布局，用户交接", "ap"),
    "ibunet_table1": ("ibUNet 论文表 1", "nrms"),
    "axialnet_unseen": ("AxialNet 官方未见电路测试", "nrms"),
    "hgrp_table1": ("HGRP 论文表 1，拥塞热点", "pr_auc"),
    "verihgn_table2": ("VeriHGN 论文表 2，N28 网格", "pearson"),
    "lhnn_table2": ("LHNN 论文表 2，水平方向拥塞", "f1"),
}


def read_csv(path):
    with (ROOT / path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def complete_training(design, role):
    reference = read(Path(ROLE_REFERENCES[role]) / "config.json")
    keys = ("model", "alpha", "epochs", "batch_size", "lr", "seed", "workers", "select_by",
            "correlation_loss_weight", "ranking_loss_weight", "hotspot_loss_weight", "bottleneck_dilation")
    stem = f"{design}_{role}"
    folder = OUT / stem
    attempt = 1
    while (ROOT / folder).exists() and not (ROOT / folder / "validation_metrics.json").is_file():
        attempt += 1
        folder = OUT / f"{stem}_attempt{attempt}"
    if not (ROOT / folder).exists():
        args = ["train.py", "--manifest", "data/n28_full/manifest.csv", "--test-design", design,
                "--val-design", "RISCY-b", "--device", "cuda", "--threshold-seed", 42,
                "--out", folder, "--val-only", "--no-progress"]
        for key in keys:
            args.extend(["--" + key.replace("_", "-"), reference[key]])
        if reference["flip_augment"]:
            args.append("--flip-augment")
        run(args, folder.name, output_dir=OUT)
    config = read(folder / "config.json")
    if (any(config[key] != reference[key] for key in (*keys, "flip_augment"))
            or config["test_design"] != design or config["val_design"] != "RISCY-b"
            or config["threshold_seed"] != 42 or config["ap_loss_weight"] != 0
            or not (ROOT / folder / "best.pt").is_file()):
        raise ValueError(f"Incompatible checkpoint in {folder}")
    return folder


def complete():
    historical = {r["method"] for r in read_csv(FOUR) if r["scope"] == "local_design_mean"}
    existing = {r["method"] for r in read_csv(SIX) if r["scope"] == "all_six"}
    methods = sorted(historical - existing)
    (ROOT / OUT).mkdir(parents=True, exist_ok=True)
    for design in FPU_DESIGNS:
        paired = read(Path("runs/spatial_refinement") / f"{design}_test.json")
        paths = {k: Path(paired["runs"][k]) for k in ("pearson", "context")}
        for role, folder in paths.items():
            if hashlib.sha256((ROOT / folder / "best.pt").read_bytes()).hexdigest() != paired["checkpoint_sha256"][role]:
                raise ValueError(f"Changed checkpoint for {design} {role}")
        paths.update({role: complete_training(design, role) for role in ROLE_REFERENCES})
        val = evaluate(design, "validation", paths, methods, output_dir=OUT)
        evaluate(design, "test", paths, methods, val, output_dir=OUT)


def replay_archives():
    (ROOT / OUT).mkdir(parents=True, exist_ok=True)
    for key, (_, folder, role) in ARCHIVES.items():
        paths = {role: Path(folder)}
        methods = ["dual_head" if role == "baseline" else "unet"]
        prefix = "archive_" + key
        val = evaluate(prefix, "validation", paths, methods, output_dir=OUT)
        evaluate(prefix, "test", paths, methods, val, output_dir=OUT)
    source = read(Path("runs/spatial_refinement/selection_b_validation.json"))
    paths = {r: Path(source["runs"][r]) for r in ("pearson", "context", "spatial")}
    val = evaluate("screen_FPU-b", "validation", paths, SCREEN_ONLY, output_dir=OUT)
    evaluate("screen_FPU-b", "test", paths, SCREEN_ONLY, val, output_dir=OUT)


def replay_pilots():
    import torch
    from train import build_model, evaluate as compute_metrics, make_loader

    torch.set_num_threads(4)
    results = []
    for name in PILOTS:
        folder = Path("runs") / name
        cfg = read(folder / "config.json")
        if len(read(folder / "history.json")) != cfg["epochs"]:
            raise ValueError(f"Incomplete pilot {name}")
        manifest = ROOT / folder / "manifest.csv"
        rows = [r for r in read_csv(manifest) if r["design"] == cfg["test_design"]]
        root = (ROOT / cfg["manifest"]).parent
        loader = make_loader(rows, root, cfg["batch_size"], "cpu", 0)
        weights = ROOT / folder / "best.pt"
        model = build_model(cfg.get("model", "fcn"), cfg.get("bottleneck_dilation", 1))
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        metrics = compute_metrics(model, loader, "cpu", cfg["threshold"], include_ap=True,
                                  include_layout_ap=True, include_map_quality=True)
        previous = ROOT / folder / "test_metrics.json"
        if previous.exists():
            original = read(previous)
            if abs(metrics["mae"] - original["mae"]) > 1e-6 or abs(metrics["f1"] - original["f1"]) > 1e-5:
                raise ValueError(f"Replayed metrics differ in {folder}")
        results.append(dict(run=str(folder), design=cfg["test_design"], layouts=len(rows),
                            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                            checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                            select_by=cfg.get("select_by", "mae"), alpha=cfg["alpha"], seed=cfg["seed"],
                            training_layouts=cfg["split_counts"]["train"], label_threshold=cfg["threshold"],
                            **{k: metrics[k] for k in ("ap", "mae", "f1", "pearson", "mean_layout_ap",
                                                       "score_threshold", "pixels", "hotspot_fraction")}))
    identity = {(r["manifest_sha256"], r["label_threshold"], r["score_threshold"], r["pixels"]) for r in results}
    if len(identity) != 1:
        raise ValueError("Pilot test populations or thresholds differ")
    (ROOT / OUT).mkdir(parents=True, exist_ok=True)
    tmp = ROOT / OUT / "pilot_results.json.tmp"
    tmp.write_text(json.dumps(results, indent=2) + "\n")
    tmp.replace(ROOT / OUT / "pilot_results.json")


@cache
def manifest_info():
    path = ROOT / "data/n28_full/manifest.csv"
    return hashlib.sha256(path.read_text().encode()).hexdigest(), Counter(r["design"] for r in read_csv(path))


@cache
def checked_test(path, seed=42):
    report = read(path)
    val = read(report["thresholds_from"])
    digest, counts = manifest_info()
    design, validation = report["design"], report["val_design"]
    expected_val = HISTORICAL.get(design, "RISCY-b")
    if ((report["split"], report["test_design"], validation, report["layouts"], report["calibrated"])
            != ("test", design, expected_val, counts[design], True)
            or (val["split"], val["design"], val["test_design"], val["layouts"], val["calibrated"])
            != ("validation", validation, design, counts[validation], True)
            or val["manifest_sha256"] != digest or report["manifest_sha256"] != digest
            or val["label_threshold"] != report["label_threshold"]):
        raise ValueError(f"Mismatched evaluation data in {path}")
    for role, folder in report["runs"].items():
        folder = Path(folder)
        cfg = read(folder / "config.json")
        sha = hashlib.sha256((ROOT / folder / "best.pt").read_bytes()).hexdigest()
        if (sha != report["checkpoint_sha256"][role] or val["checkpoint_sha256"].get(role) != sha
                or cfg["seed"] != seed or cfg.get("threshold_seed", cfg["seed"]) != 42
                or cfg["threshold"] != report["label_threshold"]
                or (cfg["test_design"], cfg["val_design"]) != (design, validation)
                or hashlib.sha256((ROOT / folder / "manifest.csv").read_bytes()).hexdigest() != digest
                or set(cfg["train_designs"]) != set(counts) - {design, validation}):
            raise ValueError(f"Wrong weights or split for {role} in {path}")
    for method, m in report["methods"].items():
        if m["pixels"] != counts[design] * 256**2 or m["score_threshold"] != val["methods"][method]["score_threshold"]:
            raise ValueError(f"Wrong pixel count or calibration for {method} in {path}")
    return report


def result_row(path, method, name, *, key=None, origin="本地", seed=42):
    r = checked_test(path, seed)
    m = r["methods"][method]
    return dict(method=key or method, name=name, origin=origin, design=r["design"], validation=r["val_design"],
                layouts=r["layouts"], label_threshold=r["label_threshold"], score_threshold=m["score_threshold"],
                forward_passes=m["forward_passes"], source=str(path), **{k: m[k] for k in (*METRICS, *COUNTS)})


def formal_results(old):
    names = {r["method"]: r["name"] for r in old if r["scope"] == "local_design_mean"}
    names.update(NAMES)
    rows = [(r["method"], Path(r["source"])) for r in old if r["scope"] == "local_fold"]
    rows += [(r["method"], Path(r["source"])) for r in read_csv(SIX) if r["scope"] == "test"]
    for design in FPU_DESIGNS:
        path = OUT / f"{design}_test.json"
        rows.extend((m, path) for m in checked_test(path)["methods"])
    folds = {d: {} for d in DESIGNS}
    for method, path in rows:
        row = result_row(path, method, names[method])
        methods = folds[row["design"]]
        if method in methods:
            old_row = methods[method]
            original = checked_test(Path(old_row["source"]))["checkpoint_sha256"]
            current = checked_test(path)["checkpoint_sha256"]
            shared = original.keys() & current.keys()
            same_checkpoint = method == "rudy" or (shared and all(original[k] == current[k] for k in shared))
            # Archived CUDA replays can differ by one or two tiles near a cutoff.
            def agrees(k):
                a, b = row[k], old_row[k]
                return a == b or (a is not None and b is not None and abs(a - b) <= 1e-6)
            if (not same_checkpoint or any(row[k] != old_row[k] for k in (*COUNTS, "label_threshold"))
                    or not all(agrees(k) for k in (*METRICS, "score_threshold"))):
                raise ValueError(f"Conflicting {method} results at {path}")
        # Prefer the six-design source when the older table repeats the same weights.
        methods[method] = row
    expected = {r["method"] for r in old if r["scope"] == "local_design_mean"} | {
        "spatial_ap_pair", "spatial_ap_context", "spatial_consensus"}
    for design, methods in folds.items():
        identities = {(r["label_threshold"], *(r[k] for k in COUNTS)) for r in methods.values()}
        if set(methods) != expected or len(identities) != 1:
            raise ValueError(f"Incomplete or inconsistent tests on {design}")
    return folds


def design_means(folds, designs):
    rows = []
    for method in folds[designs[0]]:
        group = [folds[d][method] for d in designs]
        means = {k: fmean(r[k] for r in group) if all(r[k] is not None for r in group) else None for k in METRICS}
        rows.append(dict(method=method, name=group[0]["name"], origin="本地", design="+".join(designs),
                         layouts=sum(r["layouts"] for r in group), forward_passes=group[0]["forward_passes"],
                         source=" | ".join(r["source"] for r in group), **means))
    return rows


def competition_ranks(rows):
    for k in RANKED:
        scores = {r["method"]: Decimal(str(r[k])).quantize(Decimal(".00001"), rounding=ROUND_HALF_UP)
                  for r in rows if r.get(k) is not None}
        if len(scores) < 2:
            continue
        values = sorted(scores.values(), reverse=k not in ("mae", "nrms"))
        for r in rows:
            if r["method"] in scores:
                r["rank_" + k] = values.index(scores[r["method"]]) + 1
    return rows


def csv_output(filename, fields, rows):
    dst = ROOT / OUT / filename
    tmp = dst.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(dst)


def paper_row(r):
    return dict(method=r["method"], name=r["name"], origin="论文", design=r["design"], layouts=int(r["layouts"]),
                source=r["source"], **{k: float(r[k]) for k in METRICS if r[k]})


def published_groups(old):
    others = [paper_row(r) for r in old if r["scope"] == "paper_other_protocol"]
    groups = {"ibunet_table1": [r for r in others if r["method"] in ("RouteNet", "ibUNet")],
              "axialnet_unseen": [r for r in others if r["method"] == "AxialNet"]}
    protocols = (
        ("hgrp_table1", "四个 RISCY 电路训练，两个 zero-riscy 电路验证与测试",
         "https://arxiv.org/pdf/2610.20209#page=5", ("pr_auc", "f1", "nrms", "ssim"),
         (("HGRP", .5428, .5447, .0338, .8090), ("LHNN", .4731, .4589, .0350, .7750),
          ("NetlistGNN", .4118, .3992, .0406, .7511), ("GPDL", .4055, .3558, .0495, .7789))),
        ("verihgn_table2", "作者 N28 网格级实验", "https://arxiv.org/html/2603.11075v1#S4.SS2",
         ("pearson", "mae", "spearman"),
         (("VeriHGN", .568, .034, .401), ("MIHC", .551, .034, .374), ("GAT", .414, .044, .316),
          ("GCN", .408, .043, .308), ("CircuitGNN", .387, .048, .236))),
        ("lhnn_table2", "ISPD 2011 与 DAC 2012，水平方向", "https://arxiv.org/pdf/2203.12831#page=5",
         ("f1",), (("LHNN", .4089), ("4-layer MLP", .3258), ("Pix2Pix", .3020), ("U-Net", .2975))),
    )
    for scope, design, url, keys, data in protocols:
        groups[scope] = [dict(method=name, name=name, design=design, origin="论文", source=url,
                              **dict(zip(keys, values))) for name, *values in data]
    return groups


def export_rankings():
    old = read_csv(FOUR)
    folds = formal_results(old)
    zero = [dict(r) for r in folds["zero-riscy-b"].values()]
    for key, (label, folder, role) in ARCHIVES.items():
        path = OUT / f"archive_{key}_test.json"
        r = result_row(path, "dual_head" if role == "baseline" else "unet", label, key=key, origin="本地旧权重")
        reference = zero[0]
        if any(r[k] != reference[k] for k in ("design", "label_threshold", *COUNTS)):
            raise ValueError(f"Historical evaluation uses a different population in {path}")
        original = Path(folder) / "test_metrics.json"
        if (ROOT / original).exists():
            previous = read(original)
            # Calibrating the alarm cutoff may change F1, while AP and MAE must agree.
            if any(abs(r[k] - previous[k]) > 1e-6 for k in ("ap", "mae")):
                raise ValueError(f"Replayed predictions differ from the original test in {folder}")
        zero.append(r)
    zero.extend(paper_row(r) for r in old if r["scope"] == "paper_matched_split")
    outer = [dict(r) for r in folds["RISCY-FPU-b"].values()]
    outer.extend(result_row(OUT / "screen_FPU-b_test.json", m, NAMES[m]) for m in SCREEN_ONLY)
    repeated = []
    for filename in ("outer_seed43_test.json", "consensus_outer_seed43_test.json"):
        path = SIX.parent / filename
        for method in checked_test(path, 43)["methods"]:
            repeated.append(result_row(path, method, NAMES[method], seed=43))
    population = {(r["label_threshold"], *(r[k] for k in COUNTS)) for r in (*outer, *repeated)}
    if len(population) != 1:
        raise ValueError("Screening or seed repeats use a different outer test population")
    pilots = []
    for p in read(OUT / "pilot_results.json"):
        label = f"FCN α={p['alpha']:g}，按 {p['select_by'].upper()} 选（{Path(p['run']).name}）"
        pilots.append({**p, "method": p["run"], "name": label, "origin": "本地早期",
                       "source": str(OUT / "pilot_results.json")})
    groups = {"six_designs": design_means(folds, DESIGNS), "four_designs": design_means(folds, tuple(HISTORICAL)),
              "zero_b_joint": zero, "fpu_b_including_screen": outer, "fpu_b_seed43": repeated, "pilot32": pilots,
              "mac96": [dict(method=n, name=n, origin="用户交接", design="RISCY-FPU-a", layouts=96,
                             ap=v, source="用户提供的 architecture_pilot96 交接记录")
                        for n, v in (("RUDY", .3502), ("FCN", .4124), ("U-Net", .4157))],
              **published_groups(old)}
    common = ("method", "name", "origin", "design", "validation", "layouts", "forward_passes")
    fields = (*common, *METRICS, "pr_auc", "spearman", "source")
    ranked = []
    for scope, group in groups.items():
        ranked.extend(dict(scope=scope, **r) for r in competition_ranks(group))
    rank_fields = tuple("rank_" + k for k in RANKED)
    csv_output("rankings.csv", ("scope", *fields, *rank_fields), ranked)
    by_design = []
    for design in DESIGNS:
        group = competition_ranks(list(folds[design].values()))
        by_design.extend(sorted(group, key=lambda r: (r["rank_ap"], r["name"])))
    csv_output("local_per_design.csv", (*fields, "label_threshold", "score_threshold", *COUNTS, *rank_fields), by_design)


def show_report(scope):
    table = OUT / "rankings.csv"
    if not (ROOT / table).exists():
        raise SystemExit("Run rank_models.py --stage all to generate the saved rankings.")
    rows = read_csv(table)
    for section in SCOPES if scope == "all" else (scope,):
        title, key = SCOPES[section]
        selected = [r for r in rows if r["scope"] == section]
        metric = {"pearson": "Pearson", "pr_auc": "PR-AUC"}.get(key, key.upper())
        direction = "↓" if key == "nrms" else "↑"
        print(f"\n{title}（{len(selected)} 项）\n名次  {metric} {direction}   模型")
        for row in sorted(selected, key=lambda r: (int(r["rank_" + key] or 1), r["name"])):
            print(f"{row['rank_' + key] or '—':>2}  {float(row[key]):.5f}  {row['name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stages = {"complete": complete, "pilots": replay_pilots, "archive": replay_archives, "export": export_rankings}
    parser.add_argument("--stage", choices=(*stages, "report", "all"), default="report")
    parser.add_argument("--scope", choices=(*SCOPES, "all"), default="six_designs",
                        help="Saved ranking to display with --stage report")
    args = parser.parse_args()
    if args.stage == "all":
        for stage in stages.values():
            stage()
        show_report(args.scope)
    elif args.stage == "report":
        show_report(args.scope)
    else:
        stages[args.stage]()
