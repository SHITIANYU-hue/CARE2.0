# Reaction Transfer Descriptors

日期：2026-07-04

本次工作在新的本地 clone `work/CARE2.0-transfer-descriptor` 中完成，基于 `carry2.0` 分支新建工作分支 `feature/reaction-transfer-descriptors`。没有修改用户 `Downloads` 里的论文目录。

## 做了什么

1. 新建并验证 RDKit 环境 `care2-rdkit`。
   - Python 3.11
   - RDKit `2026.03.3`
   - 验证命令：`conda run -n care2-rdkit python -c "from rdkit import Chem"`

2. 新增 reaction component descriptor 生成脚本：
   - `experiments/care_replay/scripts/build_reaction_descriptors.py`
   - 输出：`experiments/care_replay/data/descriptors/reaction_component_descriptors.csv`

3. descriptor 表覆盖 BH/Suzuki 的可迁移化学 component：

| dataset | role | unique components |
| --- | ---: | ---: |
| real_buchwald_hartwig | additive | 22 |
| real_buchwald_hartwig | aryl_halide | 15 |
| real_buchwald_hartwig | base | 3 |
| real_buchwald_hartwig | ligand | 4 |
| real_suzuki_miyaura | catalyst | 1 |
| real_suzuki_miyaura | ligand | 12 |
| real_suzuki_miyaura | reactant_1 | 7 |
| real_suzuki_miyaura | reactant_2 | 4 |
| real_suzuki_miyaura | reagent | 8 |
| real_suzuki_miyaura | solvent | 6 |

总计 82 个 component，其中 77 个 RDKit 可解析，5 个保留 semantic descriptor 且 `rdkit_parse_ok=no`。

4. descriptor schema 包含：
   - raw name、canonical SMILES、`rdkit_parse_ok`
   - RDKit descriptors：分子量、LogP、TPSA、HBD/HBA、rotatable bonds、aromatic rings、hetero atoms
   - descriptor bins：`mw_bin`、`logp_bin`、`tpsa_bin`、`rotatable_bin`、`aromatic_ring_bin`
   - functional flags：`has_phosphorus`、`has_phosphine`、`has_boron`、`has_aryl_halide`、`has_heteroaromatic`
   - semantic families：`halide_type`、`boron_species`、`ligand_family`、`reagent_base_family`、`solvent_family`、`solvent_is_protic`、`functional_class`
   - fingerprints：Morgan 128-bit、MACCS bits

5. 修改 replay adapter，让候选 metadata 带 role-prefixed descriptor 字段，例如：
   - `ligand_has_phosphine`
   - `ligand_mw_bin`
   - `reactant_1_halide_type`
   - `reactant_1_has_boron`
   - `reagent_reagent_base_family`
   - `solvent_solvent_family`
   - `solvent_solvent_is_protic`

6. 新增 descriptor transfer modes：
   - `transfer_descriptor_value_prior_no_gate`
   - `transfer_descriptor_value_prior_gate_v1`
   - `transfer_descriptor_value_prior_strict_gate_v1`

旧的 role-level transfer modes 保持不变，作为 baseline。descriptor modes 只迁移共享化学 descriptor value prior，不迁移 dataset-local label，例如不会把 `L00`、`L01` 当作跨数据集可迁移信息。

## Transfer 边界

descriptor value prior 只在 source/target 使用同一公开 descriptor vocabulary 时启用，例如：

| source | target |
| --- | --- |
| `ligand_has_phosphine` | `ligand_has_phosphine` |
| `base_reagent_base_family` | `reagent_reagent_base_family` |
| `aryl_halide_halide_type` | `reactant_1_halide_type` |
| `reactant_1_has_boron` | `aryl_halide_has_boron` |
| `solvent_functional_class` | `additive_functional_class` |

当前 v1 还加了三个保护：
   - source 或 target 字段少于两个 descriptor value 时不迁移；
   - source 或 target 覆盖率超过 85% 的 value 不迁移；
   - reaction descriptor prior 是弱先验：普通 cap `0.02`，strict positive cap `0.015`。

## 实验命令

生成 descriptor：

```bash
conda run -n care2-rdkit python experiments/care_replay/scripts/build_reaction_descriptors.py
```

BH -> Suzuki，30 seeds：

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

Suzuki -> BH，50 seeds：

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

本目录中的正式汇总文件遵循现有 `carry2.0` 结果目录习惯：

| file | contents |
| --- | --- |
| `transfer_summary.csv` | 两方向、五种 mode 的核心 aggregate metrics |
| `tables/transfer_real_buchwald_hartwig_to_real_suzuki_miyaura_reaction_descriptor_30seed_metrics.csv` | BH -> Suzuki seed-level metrics |
| `tables/transfer_real_suzuki_miyaura_to_real_buchwald_hartwig_reaction_descriptor_50seed_metrics.csv` | Suzuki -> BH seed-level metrics |
| `runs/*reaction_descriptor_30seed_summary.json` | BH -> Suzuki summary JSON |
| `runs/*reaction_descriptor_50seed_summary.json` | Suzuki -> BH summary JSON |
| `runs/*transfer_card_seed*.json` | per-seed role-level and descriptor-level transfer cards |

完整 audit logs、knowledge snapshots 和 outputs mirror 在：

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
| transfer_descriptor_value_prior_gate_v1 | 92.6085 | 87.5533 | 0.1000 | 0.0000 | 0.0000 |
| transfer_descriptor_value_prior_strict_gate_v1 | 92.6085 | 87.5533 | 0.1000 | 0.0000 | 0.0000 |

结论：旧 role-level transfer 在 BH -> Suzuki 上仍不稳定，strict 版本略好于 incumbent，但差距很小。descriptor value prior 在当前弱先验配置下被 gate 评估，但没有批准实际 intervention，因此结果与 incumbent 持平。

### Suzuki -> BH, 50 seeds

| mode | final_best | best_so_far_auc | top10_hit | interventions | bad_interventions |
| --- | ---: | ---: | ---: | ---: | ---: |
| incumbent | 86.6477 | 79.9713 | 0.1600 | 0.0000 | 0.0000 |
| transfer_gate_v1 | 89.0866 | 81.0779 | 0.2400 | 2.0800 | 0.8600 |
| transfer_strict_gate_v1 | 87.6158 | 80.2528 | 0.1600 | 0.6200 | 0.1800 |
| transfer_descriptor_value_prior_gate_v1 | 86.6477 | 79.9713 | 0.1600 | 0.0000 | 0.0000 |
| transfer_descriptor_value_prior_strict_gate_v1 | 86.6477 | 79.9713 | 0.1600 | 0.0000 | 0.0000 |

结论：Suzuki -> BH 方向旧 role-level `transfer_gate_v1` 仍有正向收益。descriptor value prior 不破坏这个结论，但当前 v1 也没有带来额外收益。

## Bad Interventions

旧 role-level transfer 会产生 bad interventions：
   - BH -> Suzuki `transfer_gate_v1`: 平均 1.6333
   - BH -> Suzuki `transfer_strict_gate_v1`: 平均 0.8333
   - Suzuki -> BH `transfer_gate_v1`: 平均 0.8600
   - Suzuki -> BH `transfer_strict_gate_v1`: 平均 0.1800

descriptor value prior 当前没有 bad interventions，原因是弱先验版本没有被 gate 批准实际 candidate replacement。也就是说它是安全的，但不是有效增益。

## 判断

leader 要求的工程项已经完成：
   - 有可复用 reaction component descriptor 表；
   - BH/Suzuki adapter 已带 descriptor metadata；
   - transfer ablation 已支持 descriptor-level transfer modes；
   - 30/50 seed 对比实验已跑完；
   - metrics、summary、audit logs 已输出。

但实验结果说明，单纯把 descriptor value prior 弱迁移进 gate 还不能解决 BH -> Suzuki 效果不好的问题。当前版本更像一个安全的 v1 infrastructure：它把 descriptor 数据通路打通了，能做 audit，但还需要更好的 descriptor selection 和 calibration 才可能变成有效 transfer 策略。

## 下一步建议

1. 做更窄的 descriptor whitelist，先只保留化学意义最明确的字段：
   - ligand: `ligand_family`, `has_phosphine`, `has_phosphorus`
   - reagent/base: `reagent_base_family`
   - reactant: `halide_type`, `has_boron`, `boron_species`, `has_heteroaromatic`
   - solvent: `solvent_family`, `solvent_is_protic`

2. 把 descriptor prior 从“直接 source effect direction”改成“source 只决定哪些 descriptor 值值得看，target early observations 决定方向”。

3. 单独分析被 gate 拒绝的 descriptor challengers，确认是 cap 太低、base acquisition margin 太高，还是 descriptor signal 本身方向不对。

4. Suzuki -> BH 不建议用 solvent -> BH additive 的直接映射作为强先验；这两个 role 化学含义不完全等价。

5. 后续可把 Morgan/MACCS fingerprint 做成 similarity bucket，而不是把 raw bit 全部迁移，避免高维 bit 带来不可解释干预。
