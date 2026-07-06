# CARE 2.0 实验进展汇总

版本日期：2026-07-06

用途：团队内部同步；可直接复制到飞书文档继续编辑。

## 0. 当前结论

这轮工作的重点不是复现 CARE 1.0 论文里的最终数字，而是验证 CARE 2.0 能否从单一反应优化扩展成一个跨任务、跨数据集的 replay + transfer 平台。现在可以比较明确地说：平台已经跑通，transfer 也已经有真实数据正例，但不同 transfer 机制的适用边界很明显。

目前最值得讲的结果有三条。

| 方向 | 对比 baseline | 最强结果 | 结论 |
| --- | --- | --- | --- |
| FreeSolv -> Lipophilicity | target-only incumbent | 100 seeds，3/5/10 budget 下 final best 分别 +1.1662、+0.9725、+0.8550 | 共享分子 descriptor 空间下，value-prior transfer 是当前最清楚的正例 |
| Suzuki-Miyaura -> Buchwald-Hartwig | target-only incumbent | final best +2.4389，AUC +1.1066 | 反应 HTE 里，role-level transfer 可以带来正向收益 |
| Suzuki-Miyaura -> Buchwald-Hartwig | mixed-kernel GP-UCB | calibration 选出的 `scale=1.5` 在 held-out seeds 上 final best +0.3966，AUC +0.3162 | transfer 进入 acquisition geometry 后，也能在强 baseline 上保留小幅正信号 |

同时也有几个需要明确说明的边界：

1. 分子方向的 GP-kernel reweighting 没有在 held-out seeds 上赢 GP-UCB，不能当正结果讲。
2. BH -> Suzuki 方向不稳定，说明 transfer 有明显方向性。
3. raw descriptor value prior 容易负迁移；必须用 target observations 校准方向。
4. LLM 已经真实接入，但目前还没有稳定超过 deterministic transfer rule。下一步更适合让 LLM 做 rule-level proposer / policy selector，而不是直接决定实验点。

## 1. 实验框架

所有任务都被统一成 offline finite-pool replay。数据集中已有候选点和真实结果，但 replay 过程中系统不能提前看未选择候选的目标值。每一轮只能基于已经 reveal 的 observations 和 public features 选择下一个候选点，然后再 reveal 真实结果。

核心组件如下：

| 组件 | 作用 |
| --- | --- |
| `no_care_random` | 不使用 CARE，随机选择未观测候选 |
| `incumbent` | target-only baseline，只用目标任务已观测数据和 public features |
| `challenger / skill` | 根据 factor evidence、transfer card、descriptor prior 或 LLM proposal 调整候选排序 |
| `gate` | 审查 challenger 是否可以覆盖 incumbent，控制 negative transfer 风险 |
| `audit log` | 记录每轮选择、gate certificate、hypothesis snapshot、transfer card 和 LLM 响应 |

主要指标：

| 指标 | 含义 |
| --- | --- |
| `final_best` | replay 结束时找到的最好结果 |
| `best_so_far_auc` | best-so-far 曲线面积；越高说明越早找到好点 |
| `simple_regret` | 距离全局最优还有多远 |
| `top10_hit` | 是否命中过全局 top 10 候选 |
| `intervention_count` | gate 授权 challenger 覆盖 incumbent 的次数 |
| `bad_intervention_count` | intervention 后结果比 incumbent 差的次数 |

## 2. 已接入数据集

目前已经接入的数据集/任务如下：

| 数据集 | 类型 | 候选数 | 作用 |
| --- | ---: | ---: | --- |
| `synthetic_suzuki_i` | synthetic reaction replay | 448 | Suzuki-like smoke test |
| `synthetic_chemlex_i` | synthetic ChemLex-style replay | 1728 | 验证 acid-amine/ChemLex 任务形态 |
| `synthetic_materials_i` | synthetic materials replay | 336 | 验证材料配方/工艺任务形态 |
| `real_buchwald_hartwig` | public real HTE | 3955 | Dreher-Doyle Buchwald-Hartwig yield replay |
| `real_suzuki_miyaura` | public real HTE | 5760 | Perera Suzuki-Miyaura yield replay |
| `real_chemlex_acidamine` | real wetlab ChemLex | 11669 | ChemLex Acid-Amine wetlab conversion replay |
| `real_moleculenet_esol` | molecular property | 1128 | ESOL solubility replay |
| `real_moleculenet_freesolv` | molecular property | 642 | FreeSolv hydration free energy replay |
| `real_moleculenet_lipophilicity` | molecular property | 4200 | Lipophilicity replay |
| `real_matbench_expt_gap` | materials property | 4604 | Matbench experimental band gap replay |

这里需要区分两类结果：真实 HTE、MoleculeNet、ChemLex、Matbench 是真实公开数据；synthetic Suzuki/ChemLex/materials 主要用于接口验证，不应该作为对外科学性能结果。

## 3. 目前完成的主要工作

### 3.1 多数据集 replay harness

已经完成统一 replay harness，支持不同数据集共用一套流程：

`TaskSpec -> SkillCard -> HypothesisEntry -> GateCertificate -> AuditLog -> Metrics`

这部分的意义是工程层面的：不同领域任务可以进入同一个实验评估框架。后续无论是接 ChemLex、材料性质、分子性质，还是 reaction HTE，都不需要重新写一套评估逻辑。

### 3.2 baseline 体系

已经补了多种 baseline，不只是和 random 比：

| Baseline | 作用 |
| --- | --- |
| `no_care_random` | 最低基线 |
| `incumbent` | public target-only baseline |
| incumbent ablations | 检查 incumbent 是否过强或过弱 |
| GP-UCB / GP-EI / kNN-UCB | 更接近常规 Bayesian optimization 的 target-only surrogate baseline |
| no-gate / gate / strict-gate transfer | 分析 transfer 收益和风险 |

这个部分很关键，因为如果只和 random 或弱 incumbent 比，transfer 的说服力不够。现在我们至少能区分：哪些结果只赢了 public incumbent，哪些能在 GP-UCB 这种更强 baseline 上保留收益。

### 3.3 transfer card 和 role-level transfer

我们实现了 source-to-target transfer card。它不直接读取 target hidden outcome，也不直接指定候选点，而是把 source domain 中哪些 role 更有证据、confidence 多高、是否可迁移，整理成可审计的结构。

例如 Suzuki -> Buchwald-Hartwig 的 role map 是：

| Source field | Target field |
| --- | --- |
| ligand | ligand |
| reagent | base |
| reactant_1 | aryl_halide |
| solvent | additive |

这条路线已经在 reaction HTE 上产生正向结果。

## 4. 主结果一：FreeSolv -> Lipophilicity

这是目前最清楚的分子方向 transfer 正例。

FreeSolv 和 Lipophilicity 都来自 MoleculeNet，输入都可以表示成 SMILES-derived descriptors，例如：

- `smiles_length_bin`
- `hetero_atom_bin`
- `halogen_bin`
- `aromatic_bin`
- `ring_token_bin`
- `branch_bin`
- `double_bond_bin`

因为 source 和 target 共享同一套 descriptor vocabulary，所以 value-level prior 的迁移是合理的。这里不是把某个数据集内部编号硬搬到另一个数据集，而是在共享 public descriptor 空间里迁移经验。

### 4.1 100-seed budget sweep

设置：

- Source：`real_moleculenet_freesolv`
- Target：`real_moleculenet_lipophilicity`
- Seeds：100
- Initial observations：5
- Budgets：3、5、10 reveal rounds
- Main mode：`transfer_value_prior_gate_v1`

结果：

| Budget | Incumbent final | Transfer final | Delta final | Delta AUC | Top10 hit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 84.9338 | 86.1000 | +1.1662 | +0.4184 | 0.07 -> 0.13 |
| 5 | 86.1287 | 87.1012 | +0.9725 | +0.6707 | 0.11 -> 0.18 |
| 10 | 87.9513 | 88.8063 | +0.8550 | +0.7009 | 0.17 -> 0.23 |

读法：低预算下提升最明显，说明 transfer prior 对 early discovery 有帮助。

### 4.2 Held-out check

为了避免只看 100-seed aggregate，我们又做了 50/50 split：seeds `0-49` 用于 calibration，seeds `50-99` 用于 held-out evaluation。

| Budget | Selected policy | Held-out delta final | Held-out delta AUC | Held-out delta top10 |
| ---: | --- | ---: | ---: | ---: |
| 3 | `transfer_value_prior_gate_v1` | +0.3425 | +0.1900 | +0.0400 |
| 5 | `transfer_value_prior_gate_v1` | +0.2975 | +0.2605 | +0.0400 |
| 10 | `transfer_value_prior_gate_v1` | +0.3750 | +0.2168 | -0.0200 |

读法：held-out 后提升幅度变小，但 final best 和 AUC 仍然稳定为正。这说明这条分子 transfer 不是单纯靠几个 seed 拉高平均值。

## 5. 主结果二：Suzuki -> Buchwald-Hartwig

这是目前 reaction HTE 方向最重要的 transfer 正例。

### 5.1 相比 public incumbent 的 role-level transfer

设置：

- Source：`real_suzuki_miyaura`
- Target：`real_buchwald_hartwig`
- Seeds：50
- Initial observations：5
- Reveal budget：10
- Source observations：96
- Main mode：`transfer_gate_v1`

结果：

| Mode | Final best | Delta final | AUC | Delta AUC | Top10 hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `no_care_random` | 82.1728 | -4.4749 | 77.4347 | -2.5366 | 0.0600 |
| `incumbent` | 86.6477 | 0.0000 | 79.9713 | 0.0000 | 0.1600 |
| `transfer_gate_v1` | 89.0866 | +2.4389 | 81.0779 | +1.1066 | 0.2400 |
| `transfer_strict_gate_v1` | 87.6158 | +0.9681 | 80.2528 | +0.2815 | 0.1600 |

读法：role-level transfer 明确优于 target-only incumbent，但普通 gate 的 bad interventions 比 strict gate 多。下一步核心是保留收益，同时降低风险。

### 5.2 相比 GP-UCB 的 acquisition-level transfer

为了避免只和 public incumbent 比，我们进一步把 CARE transfer skill 放进 GP-UCB acquisition geometry：source transfer-card role confidence 不再只是给候选加 post-hoc bonus，而是重加权 GP categorical kernel。

设置：

- Source：`real_suzuki_miyaura`
- Target：`real_buchwald_hartwig`
- Base acquisition：`mixed_kernel_gp_ucb`
- Transfer mechanism：role confidence -> categorical kernel weights
- Scale grid：`0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 6`
- Seeds：100
- Calibration/evaluation split：50/50

Calibration 结果：

| Selector | Selected policy | Calibration delta final | Calibration delta AUC | Held-out delta final | Held-out delta AUC | Held-out delta top10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| balanced | `transfer_weighted_gp_ucb_scale_1p5` | +0.3001 | +0.3162 | +0.3966 | +0.3162 | +0.0400 |
| auc_priority | `transfer_weighted_gp_ucb_scale_1p5` | +0.3001 | +0.3162 | +0.3966 | +0.3162 | +0.0400 |
| final_priority | `transfer_weighted_gp_ucb_scale_4` | +0.3379 | -0.8680 | +0.0135 | +0.1079 | -0.0200 |

读法：`scale=1.5` 不是看完 100 seeds 后手工挑的，而是由 calibration seeds 选出来，并且在 held-out seeds 上继续赢 GP-UCB。这个提升不大，但比单纯赢 incumbent 更有说服力。

## 6. descriptor transfer 的诊断结果

我们试过把 reaction component descriptor 纳入 transfer，但 raw descriptor value prior 在 Suzuki -> BH 上会严重负迁移。

原因是：reaction 里的很多 label 是 dataset-local 的，例如不同数据集里的 ligand/base 编号不一定同义。直接迁移 source value direction 风险很高。

因此我们补了一版 `target_calibrated_descriptor_prior`：

- source descriptor 不再直接决定正负方向；
- source 只提供“哪些 descriptor value 值得关注”的白名单；
- 方向和主要强度由 target 已观测样本决定；
- strict 版本只接受 target-side positive signal。

Suzuki -> BH 上的结果：

| Mode | Delta final | Delta AUC | 读法 |
| --- | ---: | ---: | --- |
| raw descriptor strict | -13.2594 | -9.4342 | 严重负迁移 |
| target-calibrated descriptor strict | +1.3678 | +1.0884 | 修复负迁移，并超过 incumbent |
| role-level transfer gate | +2.4389 | +1.1066 | 当前仍是最强 reaction transfer |

结论：descriptor transfer 不是不能用，但不能照搬 source value direction。更合理的方式是让 source 决定 attention，target observation 决定方向。

## 7. LLM 实验状态

LLM 路径已经真实接入，使用 OpenAI-compatible endpoint，audit log 中会记录 raw response、parsed policy、token usage 和 parse error。

目前结论：

1. 直接让 LLM 生成 factor adjustment 不稳定。
2. 在 reaction HTE transfer 上，LLM proposer 还没有超过 deterministic transfer card。
3. LLM auditor 能降低风险，但偏保守，容易把有用 transfer 也挡掉。
4. 在共享 descriptor 的分子任务上，LLM proposer 有小幅正信号，但还不是主结果。

因此下一步 LLM 的位置应该上移：不要让它直接选实验点，而是让它做 rule-level proposer / policy selector，例如：

- 判断 source-target pair 是否适合 transfer；
- 选择 role transfer、value-prior transfer、target-calibrated descriptor transfer，还是保持 incumbent；
- 调 gate threshold、transfer weight、support threshold；
- 根据 audit log 总结 negative transfer 原因；
- 产出可写入知识库的 skill artifact。

## 8. 材料和 ChemLex 方向

### 8.1 Matbench experimental band gap

真实 Matbench 数据已经接入，候选数 4604。当前只用 composition bin 作为 public features。

代表结果：

| Mode | Final best | AUC | Top10 hit |
| --- | ---: | ---: | ---: |
| `no_care_random` | 48.7417 | 43.7104 | 0.0000 |
| `incumbent` | 44.8000 | 41.3633 | 0.0000 |
| `no_gate` | 45.1250 | 41.4283 | 0.0000 |

读法：真实材料数据能跑，但当前 observation model 明显不够强。材料方向不应该现在主讲性能提升，应该先补 matminer-style descriptors、composition embeddings 或 pretrained materials surrogate。

### 8.2 ChemLex Acid-Amine wetlab

真实 ChemLex Acid-Amine wetlab 数据已经接入，候选数 11669。

当前 replay 能跑，但 policy 不强，random baseline 在部分设置里很高。这说明 ChemLex 需要单独分析数据 split、候选分布和 objective 分布。不能把 synthetic ChemLex 上的强提升直接迁移成真实 wetlab 结论。

## 9. CARE 2.0 知识库

除了 replay 实验，我们也搭了 CARE 2.0 知识库原型。这个知识库不是会议纪要，而是系统内部的 structured memory，用来沉淀 task、dataset、mechanism、skill、hypothesis。

已经完成：

| 模块 | 作用 |
| --- | --- |
| `seed_cards.json` | 初始 task/dataset/skill/mechanism cards |
| SQLite + FTS | 支持关键词检索 |
| `build_embeddings.py` | 支持本地 hashed embedding，也预留 OpenAI-compatible embedding endpoint |
| `query_embeddings.py` | 支持语义检索 |
| `awesome_resources.md` | 整理 AI4Chem、LLM4EDA、materials-aware LLM、molecular discovery 资源 |

我们也把 skill transfer 拆成六层：

1. representation transfer：source/target 字段映射；
2. mechanism transfer：可复用科学假说；
3. model transfer：kernel、embedding、feature transform；
4. acquisition transfer：候选排序和探索策略；
5. gate/risk transfer：识别 negative transfer；
6. workflow transfer：实验预算、审计和数据边界。

现在的 transfer-weighted kernel 属于 model/acquisition binding；value-prior transfer 属于 representation + acquisition binding；target-calibrated descriptor 属于 representation + gate/risk binding。

## 10. 当前总体判断

可以给团队这样判断：

1. CARE 2.0 的多领域 replay platform 已经成立。
2. Transfer 已经不是概念验证：FreeSolv -> Lipophilicity 和 Suzuki -> BH 都有真实数据正例。
3. 分子方向的主线是 shared descriptor value-prior；reaction 方向的主线是 role-level transfer 和 acquisition-level kernel reweighting。
4. GP-UCB 对比后，reaction transfer 仍有小幅 held-out 正信号；这比只赢 public incumbent 更有价值。
5. 不是所有机制都通用：分子 GP-kernel transfer 没保住 held-out gain，BH -> Suzuki 也不稳定。
6. LLM 已接入，但要从局部 adjustment 升级为 rule-level proposer / policy selector。
7. 材料和 ChemLex 已经接入，但还不是性能主结果，需要更强 feature/model。

## 11. 下一步建议

### 11.1 做 policy selector

现在已经有多种 transfer policy：

- role-level transfer；
- strict role transfer；
- shared descriptor value-prior；
- target-calibrated descriptor prior；
- transfer-weighted GP kernel；
- incumbent / GP-UCB fallback。

下一步应该做一个 selector：给定 source-target pair、早期 target observations、source transfer card 和历史 calibration 结果，自动选择 policy，而不是手动决定用哪种 transfer。

### 11.2 让 LLM 参与 rule evolution

LLM 不应该直接选实验点。更合理的是让它输出结构化 policy proposal：

- 哪些字段能迁移；
- 哪些 descriptor 只能做 whitelist，不能迁移方向；
- gate threshold 应该更宽还是更严；
- 是否应该 fallback 到 GP-UCB；
- 失败原因是什么，如何写入 skill library。

然后用 replay + held-out seeds 验证 LLM 提案。

### 11.3 补材料 descriptors

Matbench 目前缺的是 representation，不是 replay 框架。下一步需要补：

- composition descriptors；
- element/property embeddings；
- pretrained materials surrogate；
- 或 Matbench 里更适合 finite-pool replay 的任务。

### 11.4 把实验结果沉淀进知识库

建议把以下内容结构化进入 CARE 2.0 knowledge base：

- 每个 dataset 的 task card；
- 每次 transfer 的 source-target role map；
- 成功 transfer cases；
- negative transfer cases；
- gate rejected / approved 的典型 audit examples；
- 可复用 skill/prior 及其适用边界。

## 12. GitHub 中对应文件

当前分支：

`codex/target-calibrated-transfer`

最新关键 commit：

`1aeef9f Add calibrated transfer-weighted kernel follow-up`

主要文件位置：

| 内容 | 路径 |
| --- | --- |
| 总览 | `overview.md` |
| replay 主脚本 | `experiments/care_replay/scripts/run_synthetic_suzuki.py` |
| transfer 主脚本 | `experiments/care_replay/scripts/run_transfer_ablation.py` |
| transfer-weighted kernel | `experiments/care_replay/scripts/run_transfer_weighted_kernel.py` |
| calibration summary 脚本 | `experiments/care_replay/scripts/build_transfer_weighted_calibration_summary.py` |
| MoleculeNet value-prior sweep | `experiments/care_replay/results/2026-07-05-molprop-value-prior-budget-sweep/` |
| calibrated transfer-weighted kernel | `experiments/care_replay/results/2026-07-05-calibrated-transfer-weighted-kernel/` |
| target-calibrated descriptor transfer | `experiments/care_replay/results/2026-07-05-target-calibrated-descriptor-transfer/` |
| transfer advantage sweep | `experiments/care_replay/results/2026-07-03-transfer-advantage-sweep/` |
| real ChemLex | `experiments/care_replay/results/2026-06-30-real-chemlex/` |
| materials baseline | `experiments/care_replay/results/2026-06-29-materials-baselines/` |
| knowledge base | `knowledge_base/` |

## 13. 给团队同步时可以用的一段话

我们现在完成的不是 CARE 1.0 原数字复现，而是 CARE 2.0 的跨任务 replay 和 transfer 框架。所有任务都被统一成 finite-pool search，并且有同一套 incumbent、challenger、gate、audit、metrics。当前最清楚的正例是 FreeSolv -> Lipophilicity：共享分子 descriptor value-prior 在 100 seeds、3/5/10 个 budget 下都提升 final best 和 AUC，held-out split 后仍然保持正向。反应方向上，Suzuki -> Buchwald-Hartwig 的 role-level transfer 相比 target-only incumbent 提升明显；进一步和 GP-UCB 比，calibration 选出的 transfer-weighted kernel 在 held-out seeds 上仍然有小幅正收益。边界也很清楚：分子 GP-kernel transfer 没保住 held-out gain，BH -> Suzuki 不稳定，LLM 目前还没稳定超过 deterministic rule。下一步应该重点做 policy selector 和 LLM rule evolution，而不是继续手动给每个 pair 调一条规则。
