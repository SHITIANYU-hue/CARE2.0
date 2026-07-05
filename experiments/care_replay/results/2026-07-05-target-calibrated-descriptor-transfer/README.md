# Target-Calibrated Descriptor Transfer

日期：2026-07-05

这次实验是对 `2026-07-04-reaction-transfer-descriptors` 的直接跟进。前一版已经证明 descriptor-level transfer 通路能跑通，但直接迁移 source descriptor value prior 会带来明显风险：尤其是 Suzuki -> BH，raw descriptor prior 会把搜索方向带偏，bad interventions 很高。

本次新增一个更保守的策略：`target_calibrated_descriptor_prior`。核心变化是：

- source descriptor 不再直接决定正负方向；
- source 只提供“哪些 descriptor value 值得关注”的候选白名单；
- 正负方向和主要强度由 target 已观测样本估计；
- strict 版本只允许正向 target signal，并继续经过 gate。

换句话说，这版不是把 Suzuki 里某个 descriptor 的好坏照搬到 BH，而是用 source 帮忙缩小注意力范围，再让 target early observations 决定是否真的采用。

## 新增 mode

| mode | 说明 |
| --- | --- |
| `transfer_descriptor_target_calibrated_no_gate` | source descriptor whitelist + target-calibrated sign，不经过 gate |
| `transfer_descriptor_target_calibrated_gate_v1` | 同上，经过 `gate_v1` |
| `transfer_descriptor_target_calibrated_strict_gate_v1` | 只接受 positive target signal 的 strict 版本，经过 `gate_v1` |

## 实验命令

BH -> Suzuki, 30 seeds：

```bash
python3 experiments/care_replay/scripts/run_transfer_ablation.py \
  --source-dataset real_buchwald_hartwig \
  --target-dataset real_suzuki_miyaura \
  --seeds 30 \
  --rounds 10 \
  --initial 5 \
  --source-observations 48 \
  --modes incumbent,transfer_gate_v1,transfer_strict_gate_v1,transfer_descriptor_value_prior_gate_v1,transfer_descriptor_value_prior_strict_gate_v1,transfer_descriptor_target_calibrated_no_gate,transfer_descriptor_target_calibrated_gate_v1,transfer_descriptor_target_calibrated_strict_gate_v1 \
  --output-tag target_calibrated_30seed
```

Suzuki -> BH, 50 seeds：

```bash
python3 experiments/care_replay/scripts/run_transfer_ablation.py \
  --source-dataset real_suzuki_miyaura \
  --target-dataset real_buchwald_hartwig \
  --seeds 50 \
  --rounds 10 \
  --initial 5 \
  --source-observations 96 \
  --modes incumbent,transfer_gate_v1,transfer_strict_gate_v1,transfer_descriptor_value_prior_gate_v1,transfer_descriptor_value_prior_strict_gate_v1,transfer_descriptor_target_calibrated_no_gate,transfer_descriptor_target_calibrated_gate_v1,transfer_descriptor_target_calibrated_strict_gate_v1 \
  --output-tag target_calibrated_50seed
```

## 主要结果

### BH -> Suzuki, 30 seeds

| mode | final_best | Δ final | AUC | Δ AUC | top10 | interventions | bad interventions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent | 92.6085 | +0.0000 | 87.5533 | +0.0000 | 0.1000 | 0.0000 | 0.0000 |
| transfer_gate_v1 | 91.3879 | -1.2206 | 86.4187 | -1.1346 | 0.1667 | 3.2000 | 1.6333 |
| transfer_strict_gate_v1 | 92.8770 | +0.2685 | 87.7002 | +0.1469 | 0.1333 | 1.4333 | 0.8333 |
| transfer_descriptor_value_prior_gate_v1 | 92.5510 | -0.0575 | 88.1918 | +0.6385 | 0.0667 | 7.0000 | 4.1667 |
| transfer_descriptor_value_prior_strict_gate_v1 | 92.0322 | -0.5763 | 88.3821 | +0.8288 | 0.1000 | 7.4333 | 4.1000 |
| transfer_descriptor_target_calibrated_no_gate | 92.0185 | -0.5900 | 86.7690 | -0.7843 | 0.1667 | 5.5333 | 2.2000 |
| transfer_descriptor_target_calibrated_gate_v1 | 91.6215 | -0.9870 | 86.7785 | -0.7748 | 0.2000 | 5.0000 | 1.9333 |
| transfer_descriptor_target_calibrated_strict_gate_v1 | 92.0578 | -0.5507 | 87.8745 | +0.3212 | 0.2000 | 3.7000 | 1.8667 |

BH -> Suzuki 方向，target-calibrated strict 没有超过 incumbent 或 role-level strict transfer；它的价值主要是：相比 raw descriptor prior，bad interventions 从约 4.1 降到 1.87，同时 AUC 和 top10 有一定改善。这里不能报告成“明显胜出”，只能说 target calibration 降低了 descriptor prior 的风险。

### Suzuki -> BH, 50 seeds

| mode | final_best | Δ final | AUC | Δ AUC | top10 | interventions | bad interventions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent | 86.6477 | +0.0000 | 79.9713 | +0.0000 | 0.1600 | 0.0000 | 0.0000 |
| transfer_gate_v1 | 89.0866 | +2.4389 | 81.0779 | +1.1066 | 0.2400 | 2.0800 | 0.8600 |
| transfer_strict_gate_v1 | 87.6158 | +0.9681 | 80.2528 | +0.2815 | 0.1600 | 0.6200 | 0.1800 |
| transfer_descriptor_value_prior_gate_v1 | 75.8008 | -10.8469 | 72.3470 | -7.6243 | 0.1000 | 6.5000 | 4.8400 |
| transfer_descriptor_value_prior_strict_gate_v1 | 73.3883 | -13.2594 | 70.5371 | -9.4342 | 0.1000 | 6.1200 | 4.8000 |
| transfer_descriptor_target_calibrated_no_gate | 83.9891 | -2.6586 | 78.2160 | -1.7553 | 0.1600 | 5.0400 | 2.2200 |
| transfer_descriptor_target_calibrated_gate_v1 | 84.2290 | -2.4187 | 78.5778 | -1.3935 | 0.1600 | 4.8200 | 1.9800 |
| transfer_descriptor_target_calibrated_strict_gate_v1 | 88.0155 | +1.3678 | 81.0597 | +1.0884 | 0.1400 | 3.1600 | 0.9400 |

Suzuki -> BH 是更关键的方向，因为 raw descriptor prior 在这里负迁移最严重。target-calibrated strict 把 raw descriptor strict 的 `-13.2594` final delta 修正为 `+1.3678`，AUC 从 `-9.4342` 修正为 `+1.0884`。这说明 descriptor transfer 不是不能用，问题在于不能直接照搬 source value direction。

不过它仍然没有超过最强的 role-level `transfer_gate_v1`：后者 final delta 是 `+2.4389`，top10 也更高。因此这版更准确的结论是：**target calibration 修复了 descriptor prior 的负迁移问题，并在 Suzuki -> BH 上超过 incumbent，但还不是当前最强策略。**

## Paired Seed Check

相对 incumbent 的 paired seed 结果：

| direction | mode | mean final delta | wins | ties | mean AUC delta |
| --- | --- | ---: | ---: | ---: | ---: |
| BH -> Suzuki | transfer_descriptor_target_calibrated_strict_gate_v1 | -0.5507 | 7/30 | 19/30 | +0.3212 |
| Suzuki -> BH | transfer_descriptor_target_calibrated_strict_gate_v1 | +1.3677 | 13/50 | 28/50 | +1.0884 |
| Suzuki -> BH | transfer_descriptor_value_prior_strict_gate_v1 | -13.2594 | 6/50 | 16/50 | -9.4342 |
| Suzuki -> BH | transfer_gate_v1 | +2.4389 | 13/50 | 32/50 | +1.1067 |

这个 paired 结果也说明：target-calibrated strict 的提升不是每个 seed 都赢，更多是“少数 seed 明显救回来 + 很多 seed 持平”。这和有限预算下的 gate 行为是一致的。

## 判断

这次推进的主要价值不是找到最终最强策略，而是把 descriptor transfer 的失败原因拆开了：

1. descriptor infrastructure 是有用的；
2. raw source descriptor value direction 直接迁移会导致负迁移；
3. 用 target observations 校准方向后，Suzuki -> BH 的大幅负迁移可以被修复；
4. role-level transfer 目前仍是最强 baseline；
5. 下一步要让 LLM 参与的是“选择/优化这些策略”，而不是只在固定 schema 里给 factor adjustment。

## 下一步

建议下一轮不要继续只让 LLM 输出局部 adjustment，而是做一个 policy selector / rule-evolver：

- 输入：当前 source/target pair、descriptor card、target observation summary、历史 seed calibration 结果；
- 输出：选择 `role transfer`、`strict role transfer`、`target-calibrated descriptor strict` 或继续 incumbent；
- 在 calibration seeds 上选 policy，在 held-out seeds 上报告；
- LLM 只能看公开 observation 和 summary，不能看未揭示 target outcome；
- report 里同时给 policy 选择理由和 held-out performance。

这样更符合 CARE 2.0 “skill evolution”的叙事：LLM 不只是执行一条手写规则，而是在可验证的 replay 环境里提出和筛选可迁移策略。

## 输出文件

| path | 内容 |
| --- | --- |
| `transfer_summary.csv` | 两个方向、所有 mode 的 aggregate 指标 |
| `tables/*_metrics.csv` | seed-level metrics |
| `runs/*_summary.json` | 每个方向的 aggregate JSON |
| `runs/*_audit_*.jsonl` | 每个 seed/mode 的 audit log |
| `runs/*_knowledge_*.json` | 每个 seed/mode 的 hypothesis/knowledge snapshot |
| `runs/*_transfer_card_seed*.json` | role-level transfer card |
| `runs/*_descriptor_transfer_card_seed*.json` | descriptor-level transfer card |
