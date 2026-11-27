# PRoCoP 模型运行手册

上一张比较表中的 21 种方案都可用同一个入口单独运行。只想看已有数字用第 1 节；想用已训练的权重重新读取布局、计算预测和指标用第 2 节；想从头训练用第 3 节。以下命令在 **Bash** 中从项目根目录运行。

~~~bash
cd ~/Documents/PRoCoP
conda activate procop
~~~

预处理数据在 `data/n28_full/manifest.csv`。回放还需要本仓库 `runs/` 下的排行榜、各折运行记录、验证报告和 `best.pt`。设备默认自动选择 CUDA → MPS → CPU；想指定 GPU，在回放命令末尾加 `--device cuda`。原六折榜使用 CUDA。

## 1. 最快看到结果

~~~bash
# 打印六个测试电路的平均排名
python rank_models.py

# 打印三模型在 RISCY-FPU-b 上已经保存的单折成绩
python compare_hotspot_methods.py --show-results runs/spatial_refinement/consensus_RISCY-FPU-b_test.json

# 真正用三个归档权重在这一折的 1,248 份测试布局上重新推理
python replay_model.py spatial_consensus
~~~

最后一条会在终端打印指标表，并将新 JSON 的路径显示在表尾。默认输出为 `runs/replays/电路_方法_时间戳.json`。核对结果时，这一折三模型的历史 **AP 为 0.09997、MAE 为 0.03003**；六折等权平均的 AP 才是 **0.28353**。不同设备的末位可能略有差别。

## 2. 21 种方案的单独运行命令

每行都可以直接复制。默认测试 `RISCY-FPU-b`、验证 `RISCY-b`。入口按电路和方法查找当时的权重与验证阈值，交给 `compare_hotspot_methods.py` 在测试集重新评价。融合方案会自动加载对应成员。最后一列也是从头训练时需要准备的分支名。

| 六折 AP 名次 | 模型 | 终端命令 | 权重分支 |
|---:|---|---|---|
| 1 | 最终三模型 | `python replay_model.py spatial_consensus` | pearson, context, spatial_ap |
| 2 | Pearson + 空间 AP | `python replay_model.py spatial_ap_pair` | pearson, spatial_ap |
| 3 | Pearson + 上下文 | `python replay_model.py context_without_baseline` | pearson, context |
| 4 | 上下文双头 U-Net | `python replay_model.py placement_context` | context |
| =5 | 旧三模型，只调回归权重 | `python replay_model.py validation_map_ensemble` | baseline, pearson, context |
| =5 | 旧三模型，固定权重 | `python replay_model.py incumbent_context_ensemble` | baseline, pearson, context |
| 7 | 空间 AP 双头 U-Net | `python replay_model.py spatial_ap_context` | spatial_ap |
| 8 | 旧三模型，同时调热点和回归权重 | `python replay_model.py validation_weighted_ensemble` | baseline, pearson, context |
| 9 | 普通双头 + 上下文 | `python replay_model.py context_without_pearson` | baseline, context |
| 10 | 上下文双头，四视图推理 | `python replay_model.py context_tta` | context |
| 11 | 翻转训练的 Pearson 双头 + 上下文 | `python replay_model.py flip_context_ensemble` | flip, context |
| 12 | 普通双头 + Pearson | `python replay_model.py checkpoint_ensemble` | baseline, pearson |
| 13 | Pearson 双头 U-Net | `python replay_model.py pearson_loss` | pearson |
| 14 | Pearson 双头，四视图推理 | `python replay_model.py flip_tta` | pearson |
| 15 | Pearson 双头 + RUDY 分数 | `python replay_model.py rudy_blend` | pearson |
| 16 | 翻转训练的 Pearson 双头 | `python replay_model.py flip_training` | flip |
| 17 | 单头 U-Net | `python replay_model.py unet` | unet |
| 18 | 普通双头 U-Net | `python replay_model.py dual_head` | baseline |
| 19 | 基础 FCN | `python replay_model.py fcn` | fcn |
| 20 | Pearson 双头 + 难负例排序 | `python replay_model.py hard_negative_ranking` | ranking |
| 21 | RUDY 直接排序 | `python replay_model.py rudy` | 无需训练，评估器借用同折归档读取划分 |

**换电路、查看底层指令、指定输出文件**

~~~bash
python replay_model.py spatial_consensus --design zero-riscy-b
python replay_model.py spatial_consensus --design zero-riscy-b --print-command
python replay_model.py fcn --design RISCY-a --out runs/replays/my_fcn_RISCY-a.json
~~~

`--print-command` 只打印完整的评估命令，包括权重目录和 `--thresholds-from` 的验证 JSON。指定 `--out` 时使用新文件名。所有六折回放都使用各自的训练标签阈值与验证报警阈值；两个调权方案同时沿用对应验证集选出的融合权重。

**把同一个模型重跑六折**，把第一行换成上表任一方法名即可。

~~~bash
method=spatial_consensus
for design in zero-riscy-b RISCY-a zero-riscy-a RISCY-b RISCY-FPU-a RISCY-FPU-b; do
  python replay_model.py "$method" --design "$design" || break
done
~~~

**在一个电路上依次重跑全部 21 项**，每项分别生成 JSON。这个循环从正式六折榜读取方法名。

~~~bash
while read -r method; do
  python replay_model.py "$method" --design RISCY-FPU-b || break
done < <(python -c 'import csv; print("\n".join(r["method"] for r in csv.DictReader(open("runs/rankings/rankings.csv")) if r["scope"] == "six_designs"))')
~~~

## 3. 从头训练，并评估自己新得到的权重

21 项包含 8 个需要训练的分支。每个分支在同一折训练一次，就能在多个融合或四视图方案里复用。RUDY 直接读取输入的第 2 个通道。下面的例子复用正式实验设置：15 轮、Adam、学习率 0.002、batch size 4、seed 42、按验证 AP 选权重。

**第一步，在同一终端粘贴公共设置。** 新实验目录名由 `mktemp` 生成，记下终端打印的位置。要换折，按第 4 节同时改测试和验证电路。

~~~bash
TEST_DESIGN=RISCY-FPU-b
VAL_DESIGN=RISCY-b
DEVICE=auto
RUN_DIR=$(mktemp -d "runs/retrain_${TEST_DESIGN}_XXXXXX")
echo "$RUN_DIR"

train_branch() {
  local role=$1
  shift
  python train.py --manifest data/n28_full/manifest.csv \
    --test-design "$TEST_DESIGN" --val-design "$VAL_DESIGN" \
    --device "$DEVICE" --epochs 15 --lr 0.002 --batch-size 4 --workers 4 \
    --seed 42 --threshold-seed 42 --select-by ap --alpha 0 \
    --hotspot-loss-weight 0.01 --val-only --out "$RUN_DIR/$role" "$@"
}
~~~

**第二步，按第 2 节表格最后一列选择训练行。** 要重训最高 AP 的三模型，只运行第一段三行。要覆盖全部 21 项，再运行第二段五行。

~~~bash
train_branch pearson    --model unet-hotspot --correlation-loss-weight 0.005
train_branch context    --model unet-context --correlation-loss-weight 0.005
train_branch spatial_ap --model unet-spatial --correlation-loss-weight 0.005 --ap-loss-weight 0.005
~~~

~~~bash
train_branch fcn      --model fcn
train_branch unet     --model unet
train_branch baseline --model unet-hotspot
train_branch ranking  --model unet-hotspot --correlation-loss-weight 0.005 --ranking-loss-weight 0.002
train_branch flip     --model unet-hotspot --correlation-loss-weight 0.005 --flip-augment
~~~

每个分支会在 `$RUN_DIR/分支名/` 下写入 `config.json`、`history.json`、`best.pt` 和 `validation_metrics.json`。`--val-only` 让这一阶段在验证集选好 checkpoint 后结束。

**第三步，用新权重先校准，再测试。** 粘贴下面的 Bash 函数。它的第一个参数是第 2 节命令里的方法名，其余参数是最后一列中的分支名，用空格分开。函数先在验证电路选择报警阈值，再用该份 JSON 评估测试电路。对于调权方案，融合权重也只在验证阶段选择。

~~~bash
eval_new() {
  local method=$1 role
  shift
  local -a weights=()
  for role in "$@"; do
    weights+=("--${role//_/-}" "$RUN_DIR/$role")
  done
  local val="$RUN_DIR/${method}_validation.json"
  local test="$RUN_DIR/${method}_test.json"
  python compare_hotspot_methods.py --device "$DEVICE" --methods "$method" \
    "${weights[@]}" --split validation --calibrated --out "$val" &&
  python compare_hotspot_methods.py --device "$DEVICE" --methods "$method" \
    "${weights[@]}" --split test --thresholds-from "$val" --out "$test"
}
~~~

例如，前两行只需要上面第一段的三个分支。后三行分别演示单网络、验证集调权和 RUDY。RUDY 示例传入同折的 fcn 目录，以便评估器读取划分与标签阈值；RUDY 的分数直接来自输入。

~~~bash
eval_new spatial_consensus pearson context spatial_ap
eval_new placement_context context
eval_new fcn fcn
eval_new validation_weighted_ensemble baseline pearson context
eval_new rudy fcn
~~~

之后可随时用下列命令查看新三模型的终端结果。

~~~bash
python compare_hotspot_methods.py --show-results "$RUN_DIR/spatial_consensus_test.json"
~~~

## 4. 正式六折划分

`--design` 指测试电路。回放入口自动找到同折验证报告。从头训练时，另外四个电路用于训练，同一电路的所有布局放在同一分区。

| 测试电路 | 验证电路 |
|---|---|
| zero-riscy-b | RISCY-FPU-b |
| RISCY-a | RISCY-FPU-a |
| zero-riscy-a | RISCY-b |
| RISCY-b | zero-riscy-b |
| RISCY-FPU-a | RISCY-b |
| RISCY-FPU-b | RISCY-b |

单折回放写入自己的新报告，`python rank_models.py` 始终读取已归档排行榜。原始逐折指标及结果路径见 [local_per_design.csv](../runs/rankings/local_per_design.csv)，架构、消融与成绩见 [模型对照](MODEL_COMPARISON.md)。早期加权 MSE、单折筛选实验及其独立运行方式见 [历史实验说明](EXPERIMENTS.md) 和 [迭代日志](ITERATION_LOG.md)。

## 5. 查看 24 份布局的预测结果

[样本图片目录](../figures/README.md)按 6 个测试电路排列，每个电路 4 份布局。点击布局编号即可打开一张 2880 × 1620 PNG。所有图的排版、拥塞色标和报警图例一致，可直接用于幻灯片。

每张图依次展示三通道输入（Macro Region、RUDY、Pin RUDY）、三模型融合预测、全局布线真实拥塞和热点命中情况。底部列出这份布局的 RUDY AP、模型 AP、Precision 和 Recall。P20、P40、P60、P80 是该电路剩余 44 份候选中按模型布局内 AP 排序的抽样位置。文件名和具体指标见[固定样本清单](../figures/samples.csv)。

在项目根目录运行以下命令可以用正式权重重画同一组图。

```bash
conda activate procop
python plot_samples.py
```
