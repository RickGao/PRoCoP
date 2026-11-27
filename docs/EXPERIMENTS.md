# Experiment protocols

Use these commands from the repository root to reproduce the pilot, ablations and six-fold evaluation. For environment setup and a first run, see the [project README](../README.md).

The reference GPDL FCN maps three placement channels to one congestion map
(`--model fcn`, 119,169 parameters). The lightweight U-Net (`--model unet`, 118,725 parameters) uses
skip connections at full, half, and quarter resolution. The experiments use
design-held-out splits and report MAE and hotspot metrics.
[CircuitNet reference code](https://github.com/circuitnet/CircuitNet)

The final comparison covers 21 local methods across six held-out designs. The earlier four-design experiments and matched-fold comparisons with published methods are also recorded. [Model comparison](MODEL_COMPARISON.md)

## Data contract

Use CircuitNet-N28 routability data. `prepare_n28.py` reads the four required
archives directly, including copies with two layers of gzip compression.
Feature arrays are H x W x 3 (macro_region, RUDY, RUDY_pin); labels are H x W x 1.
Following the upstream script, preprocessing resizes to 256 x 256 with cubic
interpolation, adds horizontal and vertical overflow,
and min-max normalizes each label map. The resulting labels describe relative
spatial congestion within each layout. Constant maps are normalized to zero.

Preparation writes a CSV with columns `design,feature,label` and paths relative
to the CSV. It pairs all five maps by sample name and uses the six original design
IDs from [CircuitNet's naming convention](https://circuitnet.github.io/feature/properties.html).

## Run the pilot fold

This workspace already has `data/n28_pilot/manifest.csv` (32 layouts per design)
and `data/n28_full/manifest.csv` (10,242 layouts). Use `conda activate procop`
on this machine. For a fresh environment, run `python3 -m venv .venv`, activate
it with `source .venv/bin/activate`, and run `pip install -r requirements.txt`.

```sh
python smoke_check.py
for alpha in 0 3; do
  python train.py --manifest data/n28_pilot/manifest.csv \
    --test-design zero-riscy-b --val-design RISCY-FPU-b \
    --alpha "$alpha" --epochs 15 --lr 0.002 --out "runs/pilot_alpha${alpha}"
done
```

If you need to prepare the arrays again, use a fresh directory.

```sh
python prepare_n28.py --source CircuitNet-N28/routability_features \
  --out data/n28_pilot_fresh --per-design 32
```

Omit `--per-design` to prepare all layouts. The script selects CUDA, Metal/MPS,
or CPU in that order. Use `--device cuda` to require CUDA. Choose a fresh output
directory for each run. The default selects the checkpoint with the lowest
validation MAE. Each run saves `config.json`, `history.json`, `best.pt`,
`validation_metrics.json`, and, after testing, `test_metrics.json`.
The terminal summarizes the test metrics. To display a saved run, use
`python train.py --show-results RUN_DIR`.

## Tune for hotspot detection

Train each candidate with `--select-by f1 --val-only`. Compare alpha values
`0`, `1`, `3`, and `5` using `validation_metrics.json`. The script uses lower MAE
to break F1 ties and to rank checkpoints if the validation split has zero true
hotspots. Use the same checkpoint metric for the baseline and the weighted model.
Freeze the configuration before testing the chosen checkpoints with
`python train.py --evaluate-run RUN_DIR`.

Check hotspot precision, recall, and F1 alongside MAE. The reported
`predicted_hotspot_fraction`, `hotspot_mae`, and `zero_map_mae` quantify detection
coverage, regression error within true hotspots, and an all-zero reference.

## Run all six folds

Use `data/n28_full/manifest.csv` for the final comparison and fix the validation
design for each held-out design before training. The example uses a fixed
weighted loss with `alpha=3`, so each pair can go straight to testing. If tuning
alpha, use the validation-only workflow within each fold first.

```sh
while read -r test_design val_design; do
  for alpha in 0 3; do
    python train.py --manifest data/n28_full/manifest.csv --device cuda \
      --test-design "$test_design" --val-design "$val_design" \
      --alpha "$alpha" --select-by f1 --epochs 15 --batch-size 16 --lr 0.0002 \
      --out "runs/full_${test_design}_alpha${alpha}_seed42" || exit 1
  done
done <<'FOLDS'
RISCY-a RISCY-FPU-a
RISCY-b zero-riscy-b
RISCY-FPU-a zero-riscy-a
RISCY-FPU-b RISCY-b
zero-riscy-a RISCY-a
zero-riscy-b RISCY-FPU-b
FOLDS
```

## Evaluate ranking and a fixed tile budget

Use `--select-by ap` to select either model's checkpoint by pooled validation
hotspot average precision, breaking ties with lower MAE. Saved checkpoint tests
report AP and Top 5% as well as fixed-threshold detection metrics. The test JSON
records the label and score thresholds separately. FCN and the single-head
U-Net use the training label threshold as their score cutoff.

`evaluate_ranking.py` scores RUDY and any supplied prediction maps on the same
test layouts. It computes non-interpolated AP over all tiles in the fold and
MAE on the normalized maps. Each layout's Top 5% budget is rounded up to an
integer number of tiles. Tied scores at the cutoff are ordered by tile index.
Precision divides the total true positives in the budget by all inspected
tiles. Recall divides the same numerator by all true hotspots in the fold.

```sh
python evaluate_ranking.py --manifest data/n28_full/manifest.csv \
  --test-design RISCY-FPU-a --val-design RISCY-b \
  --out runs/top5_rudy_audit/RISCY-FPU-a.json
```

Add `--prediction FCN=PATH_TO_SCORES --prediction UNet=PATH_TO_SCORES` to compare
model predictions. Each directory should contain one `.npy` score array per
test layout, using the file name in the `label` column of the manifest. To match
a saved run, pass its training-derived label threshold using `--threshold`.
The Top 5% comparison uses the rank within each layout and requires no
prediction threshold calibration.

## Compare architectures on the two full folds

Train both models with ordinary MSE and select each checkpoint by validation
AP. Use fresh run directories. The two fold assignments below match the local
CUDA comparison.

```sh
while read -r test_design val_design; do
  for model in fcn unet; do
    python train.py --manifest data/n28_full/manifest.csv \
      --test-design "$test_design" --val-design "$val_design" \
      --model "$model" --select-by ap --alpha 0 --epochs 15 --lr 0.002 \
      --batch-size 4 --seed 42 --workers 4 --device cuda \
      --out "runs/full_${test_design}_${model}_ap_cuda" || exit 1
  done
done <<'FOLDS'
RISCY-FPU-a RISCY-b
zero-riscy-b RISCY-FPU-b
FOLDS
```

## Train U-Net with a hotspot head

Use `--model unet-hotspot --alpha 0 --hotspot-loss-weight 0.01 --select-by ap`
to train the same U-Net backbone with a second output head. Both heads use the
shared decoder. The congestion head uses MSE, and the hotspot head uses weighted
binary cross-entropy. The hotspot labels and positive class weight come from
the training split. The positive weight is the ratio of negative tiles to
positive tiles. The loss is normalized by the sum of the tile weights before
applying `--hotspot-loss-weight`.

AP and Top 5% use the hotspot scores. MAE still measures the congestion map.
Fixed-threshold metrics use a hotspot probability cutoff of 0.5. Training uses
a shuffle generator seeded independently of model initialization so that the
single-head and dual-head models see the same batch sequence for a given seed.

## Placement context and validation-calibrated evaluation

`--model unet-context` adds local, regional, row, column and layout-wide demand
summaries to both output heads. `--flip-augment` mirrors inputs and labels
together during training and can also be used with the original U-Net.
The four-design study combines an independently flip-trained dual-head U-Net with
the context model by averaging both congestion maps and both hotspot scores.
The three-model comparison averages the context prediction with the previous
ensemble. Its hotspot score weights are 50% context, 25% original dual-head,
and 25% Pearson-trained dual-head.

For the three-model weight ablation, pass `--methods incumbent_context_ensemble
validation_weighted_ensemble validation_map_ensemble` with all three checkpoints.
The second method selects hotspot weights by validation AP and map weights by
validation MAE. The third uses the original hotspot weights and selects only
the map weights. The validation JSON records the proportions and the test
command reuses them. [Fusion weight experiments](ITERATION_LOG.md#三模型融合权重微调)

The measured two-checkpoint alternative uses `--pearson`, `--context` and
`--methods context_without_baseline`. It weights hotspot scores one-third
Pearson and two-thirds context and averages the two regression maps.
[Four-design ablation and replay command](ITERATION_LOG.md#每个模型对融合的贡献)

### Spatial features and hotspot ranking

The spatial U-Net (`--model unet-spatial`) adds horizontal and vertical edge
distances and three local standard deviations to the context branch. It takes
the same three placement maps. Add `--ap-loss-weight 0.005` to train with
a soft-histogram approximation of tile AP alongside the original losses.
All validation and test reports use the exact ranking metric.

Run the validation screen, six paired test folds, and the outer seed replication
from the project root in the CUDA environment.

```bash
python run_spatial_study.py --stage screen
python run_spatial_study.py --stage test
python run_spatial_study.py --stage followup
```

The screen holds `RISCY-FPU-b` out and compares four alternatives using two
validation designs. The test stage loads the selected recipe and applies each
fold's own validation cutoff. It reuses the archived reference checkpoints
when available and trains them from the full manifest on a fresh checkout.
The follow-up compares the original and selected context checkpoints individually
when the selected recipe changes the network. The replication changes the training seed to 43
while keeping the training-label sampling seed at 42.
For the selected spatial AP recipe, the follow-up also evaluates a fixed mean of
the earlier and new ensembles, sharing their Pearson checkpoint across three network passes.
It benchmarks RUDY, ordinary MSE FCN and single-head MSE U-Net on the same six folds.
The resulting per-design scores and equal-design means are in
`runs/spatial_refinement/comparison_with_baselines.csv`.
Run `python run_spatial_study.py --stage all` for the entire experiment or
`python run_spatial_study.py --stage report` to print the saved table immediately on CPU.
[Study protocol and results](ITERATION_LOG.md#2026-11-19--空间特征与热点排序优化)

To run the three-model predictor on the 1,248 held-out `RISCY-FPU-b` layouts
using the trained checkpoints and their saved validation cutoff, use a fresh
output filename.

```bash
python compare_hotspot_methods.py --device cuda --split test \
  --pearson runs/spatial_refinement/selection_b_pearson \
  --context runs/spatial_refinement/selection_b_context \
  --spatial-ap runs/spatial_refinement/selection_b_spatial_ap \
  --methods spatial_consensus \
  --thresholds-from runs/spatial_refinement/consensus_RISCY-FPU-b_validation.json \
  --out runs/replay_fpu_b_spatial_consensus.json
```

The terminal tables show hotspot ranking, map quality and thresholded detection.
To view a saved comparison in the same format, run
`python compare_hotspot_methods.py --show-results runs/spatial_refinement/consensus_RISCY-FPU-b_test.json`.
Use `--raw` for one method name and its metrics JSON per line.

`compare_hotspot_methods.py --split validation --calibrated` selects a score
threshold per method using validation F1. Supply that report as
`--split test --thresholds-from VALIDATION_JSON` to transfer the cutoffs.
The evaluator also reports pooled hotspot AP, mean AP within positive layouts,
MAE, spatial Pearson and SSIM. The iteration log contains the training and
evaluation commands, checkpoint paths and design-level comparison.
[Iteration log](ITERATION_LOG.md#文献复查与新一轮全量实验) ·
[Results by design](../runs/context_study/comparison.csv) ·
[Held-out comparison figure](../runs/context_study/heldout_metrics.png)

### All-method rankings

From the project root, print the 21-method six-design hotspot leaderboard with
`python rank_models.py`. Run `python rank_models.py --scope zero_b_joint` for the
matched-fold Pearson ranking of local checkpoints and published results, or
`python rank_models.py --scope all` for every ranked group. These commands read the saved
`runs/rankings/rankings.csv` with the Python standard library.

`python rank_models.py --stage all` uses the four-design archive and the spatial
study as inputs, completes their missing two-fold comparisons on CUDA, replays
the early checkpoints and exports the rankings. To rebuild only the CSVs from
finished evaluations, use `python rank_models.py --stage export`.
[Model and literature comparison](MODEL_COMPARISON.md)

## Experiment protocol

1. Inspect real arrays: shapes, ranges, constant/empty labels, pairing and design identities.
2. Overfit 8-16 training examples to check learning.
3. Pilot: one test design, a distinct validation design, and a fixed sample cap
   per design. Compare alpha=0 and alpha=3 with identical seed,
   architecture, samples, optimization budget and validation selection rule.
4. Tune alpha in {1,3,5} using validation results only.
5. Final: six held-out design folds if all six are present, fixed validation
   assignment per fold, seeds 42/43/44 where budget permits. Related design
   families should be documented as a limitation of generalization claims.
6. Report per-design and equal-design-weight mean MAE, precision, recall and F1,
   seed variability, hotspot prevalence, and representative prediction/error maps.
   Include the zero-map baseline for context on MAE. Conclusions apply to map
   prediction and hotspot detection.

Weighted MSE uses w=1+alpha*I(y>=t), normalized by sum(w). Threshold t is the
90th percentile of bounded samples of positive TRAINING-label pixels, then frozen
for validation and testing. Zero-denominator precision/recall is recorded as null.
No augmentation initially: rotations can complicate directional-label semantics.

## Status

`smoke_check.py` uses small synthetic arrays to verify the training mechanics.
The commands above use the downloaded CircuitNet-N28 data.
