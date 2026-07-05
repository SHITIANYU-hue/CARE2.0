# MoleculeNet Value-Prior Transfer Budget Sweep

日期：2026-07-05

这组实验专门用来检验一个更容易产生明确 transfer gain 的场景：source 和 target 共享同一套公开 descriptor vocabulary。这里 source 是 `real_moleculenet_freesolv`，target 是 `real_moleculenet_lipophilicity`。两者都被转成 finite-pool molecular property search，候选的 public features 是 SMILES-derived descriptor bins，例如 `smiles_length_bin`、`hetero_atom_bin`、`aromatic_bin`、`ring_token_bin` 等。

和 BH/Suzuki reaction transfer 不同，这里 value-level transfer 更合理：`long_smiles` 或 `high_aromatic` 这类 descriptor bin 在两个 MoleculeNet 任务里是同一个 public vocabulary，不是数据集内部的局部标签。因此我们允许 source value prior 作为 transfer skill 直接参与候选打分，并继续通过 gate 审计。

## 实验设置

共同设置：

- Source：`real_moleculenet_freesolv`
- Target：`real_moleculenet_lipophilicity`
- Seeds：100
- Initial observations：5
- Source observations：192
- Budgets：3、5、10 reveal rounds
- Modes：`incumbent`、`transfer_gate_v1`、`transfer_value_prior_gate_v1`、`transfer_value_prior_strict_gate_v1`

命令示例：

```bash
python3 experiments/care_replay/scripts/run_transfer_ablation.py \
  --source-dataset real_moleculenet_freesolv \
  --target-dataset real_moleculenet_lipophilicity \
  --seeds 100 \
  --rounds 5 \
  --initial 5 \
  --source-observations 192 \
  --modes incumbent,transfer_gate_v1,transfer_value_prior_gate_v1,transfer_value_prior_strict_gate_v1 \
  --output-tag molprop_value_prior_budget5_100seed
```

## 主要结果

### 3-round budget

| mode | final_best | Δ final | AUC | Δ AUC | top10 | Δ top10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent | 84.9338 | +0.0000 | 84.1358 | +0.0000 | 0.0700 | +0.0000 |
| transfer_gate_v1 | 84.9338 | +0.0000 | 84.1358 | +0.0000 | 0.0700 | +0.0000 |
| transfer_value_prior_gate_v1 | 86.1000 | +1.1662 | 84.5542 | +0.4184 | 0.1300 | +0.0600 |
| transfer_value_prior_strict_gate_v1 | 84.3175 | -0.6163 | 83.4867 | -0.6491 | 0.0600 | -0.0100 |

### 5-round budget

| mode | final_best | Δ final | AUC | Δ AUC | top10 | Δ top10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent | 86.1287 | +0.0000 | 84.8325 | +0.0000 | 0.1100 | +0.0000 |
| transfer_gate_v1 | 86.4137 | +0.2850 | 84.8683 | +0.0358 | 0.1300 | +0.0200 |
| transfer_value_prior_gate_v1 | 87.1012 | +0.9725 | 85.5032 | +0.6707 | 0.1800 | +0.0700 |
| transfer_value_prior_strict_gate_v1 | 85.6538 | -0.4749 | 84.1600 | -0.6725 | 0.1300 | +0.0200 |

### 10-round budget

| mode | final_best | Δ final | AUC | Δ AUC | top10 | Δ top10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent | 87.9513 | +0.0000 | 86.0901 | +0.0000 | 0.1700 | +0.0000 |
| transfer_gate_v1 | 88.2013 | +0.2500 | 86.1850 | +0.0949 | 0.1800 | +0.0100 |
| transfer_value_prior_gate_v1 | 88.8063 | +0.8550 | 86.7910 | +0.7009 | 0.2300 | +0.0600 |
| transfer_value_prior_strict_gate_v1 | 87.4000 | -0.5513 | 85.4852 | -0.6049 | 0.1300 | -0.0400 |

## Paired Seed Check

相对 incumbent 的 paired seed 结果：

| budget | mode | mean final delta | wins | ties | mean AUC delta | seeds with top10 improvement |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 3 | transfer_value_prior_gate_v1 | +1.1663 | 35/100 | 41/100 | +0.4183 | 10 |
| 5 | transfer_value_prior_gate_v1 | +0.9725 | 40/100 | 29/100 | +0.6707 | 14 |
| 10 | transfer_value_prior_gate_v1 | +0.8550 | 39/100 | 25/100 | +0.7009 | 15 |

这个 paired 结果很重要：提升不是来自单个 outlier seed，而是在大量 seed 上 either 持平或变好，且 top10 命中率持续提高。

## 判断

这是目前最适合作为 CARE 2.0 transfer 正向效果展示的一组结果。原因有三点：

1. **transfer boundary 干净**：FreeSolv 和 Lipophilicity 共享 descriptor vocabulary，value prior 不依赖 dataset-local label。
2. **低预算也有效**：3-round budget 下 final best 已经 +1.1662，top10 hit 从 0.07 到 0.13。
3. **多预算一致**：3、5、10 round 都是同一个 mode 正向，AUC 和 top10 都提高。

也要诚实说明：strict value prior 不是更好，反而变差。这说明当前 shared descriptor value prior 需要允许一定探索，不应该过早只保留 strict-positive signals。后续可以把 strict threshold 和 prior cap 交给 calibration split 或 LLM rule-level proposer 来选。

## 和 Reaction Transfer 的关系

这组结果不是要替代 Suzuki -> BH reaction transfer，而是提供一个更清楚的“跨任务共享 descriptor transfer”正例。Reaction transfer 目前的主结论仍然是：

- role-level Suzuki -> BH 在 10-round/100-seed 下有稳定正向 gain；
- raw reaction descriptor value prior 风险高；
- target-calibrated descriptor 能修复部分负迁移，但还不是最强。

因此 CARE 2.0 的叙事可以分成两条：

- 分子性质任务：共享 descriptor value prior 是当前最强、最干净的 transfer win；
- 反应 HTE 任务：role-level transfer 有正向信号，descriptor transfer 需要 target calibration。

## 输出文件

| path | 内容 |
| --- | --- |
| `transfer_summary.csv` | 三个 budget、四种 mode 的 aggregate 指标 |
| `tables/*_metrics.csv` | seed-level metrics |
| `runs/*_summary.json` | 每个 budget 的 aggregate JSON |
| `runs/*_audit_*.jsonl` | 每个 seed/mode 的 audit log |
| `runs/*_knowledge_*.json` | 每个 seed/mode 的 hypothesis/knowledge snapshot |
| `runs/*_transfer_card_seed*.json` | per-seed transfer card |
