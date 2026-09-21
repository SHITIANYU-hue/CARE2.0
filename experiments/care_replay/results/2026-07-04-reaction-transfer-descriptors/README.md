# Reaction Transfer Descriptors

日期：2026-07-04

本次工作基于 `carry2.0` 分支，在 `feature/reaction-transfer-descriptors` 中完成。目标是把 BH/Suzuki 的 ligand、reagent/base、reactant、solvent/additive 做成可迁移的化学 descriptor，并接入 transfer gate 做重跑评估。

## 做了什么

1. 新增 RDKit reaction component descriptor 表：
   - 脚本：`experiments/care_replay/scripts/build_reaction_descriptors.py`
   - 输出：`experiments/care_replay/data/descriptors/reaction_component_descriptors.csv`
   - 覆盖 82 个唯一 component，其中 77 个 RDKit 可解析，5 个保留 semantic descriptor。

2. descriptor schema 包含：
   - canonical SMILES、`rdkit_parse_ok`
   - RDKit descriptors：MW、LogP、TPSA、HBD/HBA、rotatable bonds、aromatic rings、hetero atoms
   - descriptor bins：`mw_bin`、`logp_bin`、`tpsa_bin`、`rotatable_bin`、`aromatic_ring_bin`
   - functional flags：`has_phosphorus`、`has_phosphine`、`has_boron`、`has_aryl_halide`、`has_heteroaromatic`
   - semantic families：`halide_type`、`boron_species`、`ligand_family`、`reagent_base_family`、`solvent_family`、`solvent_is_protic`、`functional_class`
   - Morgan 128-bit fingerprint 和 MACCS bits

3. replay adapter 现在会给候选反应带上 role-prefixed descriptor metadata，例如：
   - `ligand_has_phosphine`
   - `reactant_1_halide_type`
   - `reactant_1_has_boron`
   - `reagent_reagent_base_family`
   - `solvent_solvent_family`

4. 新增 descriptor-level transfer modes：
   - `transfer_descriptor_value_prior_gate_v1`
   - `transfer_descriptor_value_prior_strict_gate_v1`

旧的 role-level transfer modes 保持不变，作为 baseline。descriptor modes 只迁移共享化学 descriptor value prior，不迁移 dataset-local label，例如不会把 `L00`、`L01` 当成跨数据集信息。

## Matched-Cap 修正

第一次实现中，descriptor cap 被设置得过弱：

| mode | old cap |
| --- | ---: |
| descriptor signed cap | 0.02 |
| descriptor strict positive cap | 0.015 |

但 `gate_v1` 要求 `gate_margin >= 0.025`。因为 incumbent 本来就是 base score 最高的候选，descriptor 最多只能加 `0.02` 时，descriptor-only challenger 基本不可能通过 gate。因此 weak-cap 版本虽然参与 scoring，但实际 intervention 为 0，结果等同 incumbent。

本次 corrected rerun 已把 descriptor cap 对齐到现有 value prior mode：

| mode | corrected cap |
| --- | ---: |
| descriptor signed cap | 0.08 |
| descriptor strict positive cap | 0.06 |

这次 matched-cap 后 descriptor gate 确实产生了 approved interventions，因此下面结果反映的是 descriptor prior 真正介入后的效果。

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
  --modes incumbent,transfer_gate_v1,transfer_strict_gate_v1,transfer_descriptor_value_prior_gate_v1,transfer_descriptor_value_prior_strict_gate_v1 \
  --output-tag reaction_descriptor_30seed
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
  --modes incumbent,transfer_gate_v1,transfer_strict_gate_v1,transfer_descriptor_value_prior_gate_v1,transfer_descriptor_value_prior_strict_gate_v1 \
  --output-tag reaction_descriptor_50seed
```

## 输出文件

| file | contents |
| --- | --- |
| `transfer_summary.csv` | 两方向、五种 mode 的核心 aggregate metrics |
| `tables/transfer_real_buchwald_hartwig_to_real_suzuki_miyaura_reaction_descriptor_30seed_metrics.csv` | BH -> Suzuki seed-level metrics |
| `tables/transfer_real_suzuki_miyaura_to_real_buchwald_hartwig_reaction_descriptor_50seed_metrics.csv` | Suzuki -> BH seed-level metrics |
| `runs/*reaction_descriptor_30seed_summary.json` | BH -> Suzuki summary JSON |
| `runs/*reaction_descriptor_50seed_summary.json` | Suzuki -> BH summary JSON |
| `runs/*transfer_card_seed*.json` | per-seed role-level transfer cards |
| `runs/*descriptor_transfer_card_seed*.json` | per-seed descriptor transfer cards |

完整 audit logs 和 outputs mirror 在：

```text
experiments/care_replay/outputs/runs/*reaction_descriptor_30seed*
experiments/care_replay/outputs/runs/*reaction_descriptor_50seed*
```

## 主要结果

### BH -> Suzuki, 30 seeds

| mode | final_best | best_so_far_auc | top10_hit | interventions | bad_interventions |
| --- | ---: | ---: | ---: | ---: | ---: |
| incumbent | 92.6085 | 87.5533 | 0.1000 | 0.0000 | 0.0000 |
| transfer_gate_v1 | 91.3879 | 86.4187 | 0.1667 | 3.2000 | 1.6333 |
| transfer_strict_gate_v1 | 92.8770 | 87.7002 | 0.1333 | 1.4333 | 0.8333 |
| transfer_descriptor_value_prior_gate_v1 | 92.5510 | 88.1918 | 0.0667 | 7.0000 | 4.1667 |
| transfer_descriptor_value_prior_strict_gate_v1 | 92.0322 | 88.3821 | 0.1000 | 7.4333 | 4.1000 |

matched-cap descriptor transfer 不再等同 incumbent：普通 descriptor gate 在 300 轮中批准 210 次 intervention，strict descriptor gate 批准 223 次。但 final_best 没有超过 incumbent，bad interventions 明显升高。AUC 有小幅上升，说明它有时提前探索到较好区域，但最终选择稳定性不足。

### Suzuki -> BH, 50 seeds

| mode | final_best | best_so_far_auc | top10_hit | interventions | bad_interventions |
| --- | ---: | ---: | ---: | ---: | ---: |
| incumbent | 86.6477 | 79.9713 | 0.1600 | 0.0000 | 0.0000 |
| transfer_gate_v1 | 89.0866 | 81.0779 | 0.2400 | 2.0800 | 0.8600 |
| transfer_strict_gate_v1 | 87.6158 | 80.2528 | 0.1600 | 0.6200 | 0.1800 |
| transfer_descriptor_value_prior_gate_v1 | 75.8008 | 72.3470 | 0.1000 | 6.5000 | 4.8400 |
| transfer_descriptor_value_prior_strict_gate_v1 | 73.3883 | 70.5371 | 0.1000 | 6.1200 | 4.8000 |

Suzuki -> BH 方向，旧 role-level `transfer_gate_v1` 仍然是有效的正向 transfer baseline。matched-cap descriptor transfer 明显变差，说明直接把 source descriptor value prior 作为强方向性先验迁移，会把 BH 搜索带偏。

## Gate Audit

matched-cap 后，descriptor policy 确实参与并改变选择：

| direction | mode | rounds | authorized | challenger_matches_incumbent | rejected_by_gate_bounds |
| --- | --- | ---: | ---: | ---: | ---: |
| BH -> Suzuki | descriptor gate | 300 | 210 | 64 | 26 |
| BH -> Suzuki | descriptor strict gate | 300 | 223 | 56 | 21 |
| Suzuki -> BH | descriptor gate | 500 | 325 | 97 | 78 |
| Suzuki -> BH | descriptor strict gate | 500 | 306 | 129 | 65 |

这说明 corrected cap 解决了“descriptor 完全不介入”的问题，但暴露出另一个问题：当前 descriptor prior 的方向和强度还没有校准好。

## 判断

leader 要求的工程项已经完成：
   - 有可复用 reaction component descriptor 表；
   - BH/Suzuki adapter 已带 descriptor metadata；
   - transfer ablation 已支持 descriptor-level transfer modes；
   - 30/50 seed 对比实验已跑完；
   - metrics、summary、audit logs 已输出。

matched-cap rerun 的实验结论是：descriptor transfer 通路已经真正生效，但直接迁移 descriptor value direction 目前不是一个可靠提升策略。它在 BH -> Suzuki 上没有超过 incumbent，在 Suzuki -> BH 上明显劣化，并带来较多 bad interventions。

因此，这版最准确的定位是：**descriptor infrastructure + diagnostic rerun 已完成；raw descriptor value prior 不应直接作为强迁移策略上线。**

## 下一步建议

1. 不建议直接使用 matched-cap descriptor value prior 作为默认 transfer gate。
2. 先把 source descriptor prior 改成“候选 descriptor whitelist”，方向由 target early observations 决定。
3. 优先保留化学意义明确的字段：
   - ligand: `ligand_family`, `has_phosphine`, `has_phosphorus`
   - reagent/base: `reagent_base_family`
   - reactant: `halide_type`, `has_boron`, `boron_species`, `has_heteroaromatic`
   - solvent: `solvent_family`, `solvent_is_protic`
4. Suzuki -> BH 不建议把 solvent -> additive 作为强方向性先验；两个 role 化学含义不完全等价。
5. Morgan/MACCS fingerprint 后续更适合做 similarity bucket 或 nearest-neighbor grouping，而不是 raw bit 直接迁移。
