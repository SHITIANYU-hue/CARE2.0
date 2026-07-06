# Calibrated Transfer-Weighted Kernel Follow-up

日期：2026-07-05

这轮实验是为了回应一个关键质疑：之前 `transfer_weighted_gp_ucb_scale_1p5` 看起来比 GP-UCB 好，但 scale 是不是看完结果后手工挑出来的？

这里把问题改成 calibration / held-out 设置：

1. 跑一个固定 scale grid：`0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 6`。
2. `scale=0` 就是原始 `mixed_kernel_gp_ucb`。
3. 用 seeds `0-49` 做 calibration，按预先写好的 selector 选 policy。
4. 只在 seeds `50-99` 上汇报最终 held-out delta。

这样就不是看完整 100 seeds 后挑一个好看的 scale，而是先选参数，再看没参与选择的 seeds。

## 新增脚本

新增：

```bash
python3 experiments/care_replay/scripts/build_transfer_weighted_calibration_summary.py \
  --metrics <metrics.csv> \
  --out-dir experiments/care_replay/results/2026-07-05-calibrated-transfer-weighted-kernel \
  --label <label> \
  --calibration-seed-count 50
```

这个脚本只读 metrics，不重新跑 replay。它输出：

- 每个 mode 的 calibration/evaluation 指标；
- 相对 baseline 的 held-out delta；
- 三个 selector 的选择结果：`balanced`、`final_priority`、`auc_priority`。

## 1. Reaction HTE：Suzuki-Miyaura -> Buchwald-Hartwig

设置：

- Source：`real_suzuki_miyaura`
- Target：`real_buchwald_hartwig`
- Base acquisition：`mixed_kernel_gp_ucb`
- Transfer mechanism：source transfer-card role confidence -> GP categorical kernel weights
- Seeds：100
- Initial observations：5
- Reveal budget：10
- Calibration/evaluation split：50/50

命令：

```bash
python3 experiments/care_replay/scripts/run_transfer_weighted_kernel.py \
  --source-dataset real_suzuki_miyaura \
  --target-dataset real_buchwald_hartwig \
  --seeds 100 \
  --rounds 10 \
  --initial 5 \
  --source-observations 96 \
  --scales 0,0.25,0.5,0.75,1,1.25,1.5,2,3,4,6 \
  --output-tag calibrated_grid_100seed
```

Calibration 选择结果：

| Selector | Selected policy | Calibration delta final | Calibration delta AUC | Held-out delta final | Held-out delta AUC | Held-out delta top10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| balanced | `transfer_weighted_gp_ucb_scale_1p5` | +0.3001 | +0.3162 | +0.3966 | +0.3162 | +0.0400 |
| auc_priority | `transfer_weighted_gp_ucb_scale_1p5` | +0.3001 | +0.3162 | +0.3966 | +0.3162 | +0.0400 |
| final_priority | `transfer_weighted_gp_ucb_scale_4` | +0.3379 | -0.8680 | +0.0135 | +0.1079 | -0.0200 |

判断：

这是目前 reaction HTE 方向更扎实的一条结果。`scale=1.5` 不是在 100 seeds 上事后挑的；它在 calibration seeds 上被 balanced / AUC selector 选中，并且在 held-out seeds 上仍然超过 GP-UCB。提升幅度不大，但方向一致：final best、AUC、top10 都是正的。

这说明 CARE transfer skill 进入 acquisition geometry 之后，可以在强 target-only GP-UCB baseline 上产生可复查的正向收益。

## 2. Molecular Property GP-kernel：FreeSolv -> Lipophilicity

设置：

- Source：`real_moleculenet_freesolv`
- Target：`real_moleculenet_lipophilicity`
- Base acquisition：`mixed_kernel_gp_ucb`
- Transfer mechanism：source transfer-card role confidence -> GP categorical kernel weights
- Seeds：100
- Initial observations：5
- Reveal budget：10
- Source observations：192
- Calibration/evaluation split：50/50

命令：

```bash
python3 experiments/care_replay/scripts/run_transfer_weighted_kernel.py \
  --source-dataset real_moleculenet_freesolv \
  --target-dataset real_moleculenet_lipophilicity \
  --seeds 100 \
  --rounds 10 \
  --initial 5 \
  --source-observations 192 \
  --scales 0,0.25,0.5,0.75,1,1.25,1.5,2,3,4,6 \
  --output-tag calibrated_grid_100seed
```

Calibration 选择结果：

| Selector | Selected policy | Calibration delta final | Calibration delta AUC | Held-out delta final | Held-out delta AUC | Held-out delta top10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| balanced | `transfer_weighted_gp_ucb_scale_3` | +0.3450 | +0.0217 | -0.1925 | -0.1654 | -0.0200 |
| final_priority | `transfer_weighted_gp_ucb_scale_3` | +0.3450 | +0.0217 | -0.1925 | -0.1654 | -0.0200 |
| auc_priority | `transfer_weighted_gp_ucb_scale_1p5` | +0.0875 | +0.2230 | -0.3600 | -0.0824 | -0.0200 |

判断：

这条不能报告成正向结果。FreeSolv -> Lipophilicity 的 GP-kernel reweighting 在 calibration seeds 上有小幅正信号，但 held-out seeds 上没有保住。这里的结论是：分子方向不能简单复用 reaction 的 transfer-weighted kernel 方案。

这也帮助我们收窄下一步方向：分子性质任务的主结果应该继续放在 shared descriptor value-prior，而不是 GP-kernel field reweighting。

## 3. Molecular Property Value-prior Held-out Check

为了确认分子方向的主结果不是 100-seed aggregate 偶然好看，我们对 `2026-07-05-molprop-value-prior-budget-sweep` 的 3/5/10 budget 结果也做了同样的 50/50 split。baseline 是 target-only `incumbent`，selector 在 calibration seeds 上选择 mode。

| Budget | Selected policy | Held-out delta final | Held-out delta AUC | Held-out delta top10 |
| ---: | --- | ---: | ---: | ---: |
| 3 | `transfer_value_prior_gate_v1` | +0.3425 | +0.1900 | +0.0400 |
| 5 | `transfer_value_prior_gate_v1` | +0.2975 | +0.2605 | +0.0400 |
| 10 | `transfer_value_prior_gate_v1` | +0.3750 | +0.2168 | -0.0200 |

判断：

这说明 FreeSolv -> Lipophilicity 的 shared descriptor value-prior result 仍然是当前最稳的分子 transfer 正例。它在 full 100-seed aggregate 上提升更明显；在 held-out split 上提升幅度变小，但 final best 和 AUC 仍然稳定为正。

## 总结

这轮推进后，CARE 2.0 的 transfer 证据可以分得更清楚：

1. **Reaction HTE acquisition-level transfer 有更扎实的正向证据。** Suzuki -> BH 上，用 calibration seeds 选出的 `scale=1.5` 在 held-out seeds 上仍然赢 GP-UCB。
2. **Molecular shared descriptor value-prior 仍然是最强主结果。** FreeSolv -> Lipophilicity 在 3/5/10 budget 的 held-out split 上 final 和 AUC 都保持正向。
3. **Molecular GP-kernel transfer 不是当前正确方向。** 它在 held-out 上没赢，说明不是所有 transfer mechanism 都能通用。

下一步如果还要扩大提升，应该把 LLM 或自动搜索放在 policy-selection / rule-evolution 层：根据 source-target pair 决定用 role transfer、kernel transfer、value-prior transfer、target-calibrated descriptor，还是保持 incumbent，而不是固定一种 transfer 机制到处套。

## 输出文件

| path | 内容 |
| --- | --- |
| `runs/*calibrated_grid_100seed*_summary.json` | 两个 100-seed GP-kernel grid 的 aggregate summary |
| `runs/*calibrated_grid_100seed*_audit_*.jsonl` | 每个 seed/mode 的 audit log |
| `runs/*calibrated_grid_100seed*_transfer_card_seed*.json` | 每个 seed 的 transfer card |
| `tables/*calibrated_grid_100seed*_metrics.csv` | 两个 GP-kernel grid 的 seed-level metrics |
| `*_policy_calibration_summary.csv` | calibration/evaluation split table |
| `*_selected_policies.json` | 三个 selector 的选择结果和 held-out delta |
