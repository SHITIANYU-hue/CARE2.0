# Overview

## 1. 这轮实验想验证什么

这轮实验不是在复现 CARE 1.0 论文的最终数字，也不是宣称 CARE 2.0 已经在所有任务上有显著提升。目标更基础一点：先把 CARE 2.0 的 replay harness 跑通，看同一套 `incumbent -> challenger/skill -> gate -> audit -> metrics` 流程能不能接不同类型的数据集。

具体来说，我们想验证三件事：

1. CARE 的决策流程能否从单一 synthetic task 扩展到真实 HTE 数据。
2. BH -> Suzuki 这种近邻化学任务迁移，能不能用同一套 gate/evidence 逻辑表达。
3. 分子性质任务能不能也转成 finite-pool search，从而纳入 CARE 2.0 平台。

这和 Tianyu 发的 CARE 2.0 PPT 和 AI4Science 跨领域任务分析是一致的。PPT 里提到 CARE 1.0 的 Buchwald-Hartwig、ChemLex、BH -> Suzuki；AI4Science 文档里强调不同领域都有共同结构：在很大的候选空间里找高价值点，把语义知识转成可执行操作，再通过验证闭环确认结果。

## 2. 实验框架怎么做的

我们把每个任务都统一成一个 offline finite-pool replay。也就是说，数据集中已经有一批候选实验点和对应结果，但 replay 时系统不能提前看目标值。每一轮只能根据已经 reveal 的 observation 和 public features 选择下一个点，然后再把这个点的真实结果 reveal 出来。

每轮都有三个角色：

1. `incumbent`：稳健 baseline，只根据已公开观测做选择。
2. `challenger/skill`：CARE 2.0 侧的候选调整逻辑，用已观察到的 factor evidence 或 skill prior 给候选加分/减分。
3. `gate`：审计层，不让 challenger 直接接管实验选择，而是检查它是否有足够 public evidence，是否风险过高，是否偏离太大。

我们保留的指标包括：

- `final_best`：最终找到的最好结果。
- `AUC`：整个实验过程里的 best-so-far 曲线面积，越高说明越早找到好点。
- `regret`：距离全局最好点还有多远。
- `top-10 hit`：是否命中全局 top 10% 区域。
- `interventions`：gate 授权 challenger 覆盖 incumbent 的次数。
- `bad interventions`：授权后结果变差的次数。

这个设置的好处是，最后不只看分数，还能看 gate 到底有没有在控制风险。

## 3. 第一阶段：Synthetic Suzuki smoke test

这一步是最早的 smoke test，目的不是讲真实化学结论，而是确认 replay 接口、audit log、skill 和 hypothesis update 都能跑。

数据集：

- 名称：`synthetic_suzuki_i`
- 类型：合成 Suzuki-like finite pool
- 候选数：448
- 决策变量：`ligand_identity`, `residence_time`, `temperature`, `catalyst_loading`
- 隐藏目标：`yield_value`
- 设置：30 seeds，5 个 initial observations，10 轮 reveal

最小 skill：

- `suzuki_ligand_prior`
- `suzuki_ligand_risk_penalty`
- `suzuki_ligand_diversity_explorer`

结果：

| 方法 | Final Best | AUC | Regret | Top-10 Hit | Interventions | Bad Interventions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent | 81.8802 | 79.0852 | 5.5186 | 0.6000 | 0.0000 | 0.0000 |
| gate_v1 | 86.4365 | 85.1926 | 0.9622 | 0.9333 | 2.6333 | 0.5000 |
| gate_v2 | 86.4365 | 85.1926 | 0.9622 | 0.9333 | 2.6333 | 0.5000 |

怎么解读：

这里 gate 版本比纯 incumbent 好，尤其 AUC、regret 和 top-10 hit 有明显改善。这个结果主要说明接口是有效的：skill adjustment 能改变排序，gate 能授权部分 intervention，audit/metrics 能记录下来。它不能说明真实 HTE 上一定会提升，因为这个数据是 synthetic。

## 4. 第二阶段：真实 HTE 数据集

这一阶段把代码从 synthetic 扩到公开真实 HTE 数据。

### 4.1 Buchwald-Hartwig

数据集：

- 名称：`real_buchwald_hartwig`
- 类型：真实 HTE reaction yield 数据
- 来源：`rxn4chemistry/rxn_yields` 里的 Dreher-Doyle Buchwald-Hartwig workbook
- 候选数：3955
- 任务形式：给定已观察反应结果，在有限实验预算下继续选择下一批反应条件，目标是找到更高 yield。

当前保守 gate 结果：

| 方法 | Final Best | AUC | Regret | Top-10 Hit | Interventions | Bad Interventions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent | 86.8818 | 80.8275 | 13.1182 | 0.1667 | 0.0000 | 0.0000 |
| gate_v1 | 86.8818 | 80.8275 | 13.1182 | 0.1667 | 0.3000 | 0.1000 |
| gate_v2 | 86.8818 | 80.8275 | 13.1182 | 0.1667 | 0.3000 | 0.1000 |

怎么讲：

Buchwald-Hartwig 上，当前版本基本持平，没有提升 final best。这说明现在的 public evidence model 是偏保守的，它不会乱授权太多 challenger，但也还没有把真实 HTE 的 gain 打出来。这个结果比较适合内部判断下一步怎么改 gate，而不是对外包装成提升。

补充说明：

早一点我们试过更 aggressive 的 factor evidence gate，在 Buchwald-Hartwig 上有过提升，final best 大概从 86.88 到 88.97；但同一套设置在 Suzuki 上不稳定，所以后来改成了现在这个更保守的版本。这是一个很重要的实验教训：gate 不能只在一个数据集上调好看，必须跨数据集稳定。

### 4.2 Suzuki-Miyaura

数据集：

- 名称：`real_suzuki_miyaura`
- 类型：真实 HTE reaction yield 数据
- 来源：`rxn4chemistry/rxn_yields` 里的 Perera Suzuki-Miyaura workbook
- 候选数：5760
- 与 CARE 2.0 路线的关系：对应 BH -> Suzuki 近邻域迁移。

结果：

| 方法 | Final Best | AUC | Regret | Top-10 Hit | Interventions | Bad Interventions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent | 92.6085 | 87.5533 | 7.3915 | 0.1000 | 0.0000 | 0.0000 |
| gate_v1 | 92.6085 | 87.4647 | 7.3915 | 0.1000 | 0.1333 | 0.0667 |
| gate_v2 | 92.6085 | 87.4647 | 7.3915 | 0.1000 | 0.1333 | 0.0667 |

怎么讲：

Suzuki 上 final best 持平，AUC 略低。这说明 BH -> Suzuki 的迁移还没有形成真正有用的 transfer skill。目前只是同一套 replay 接口能跑到 Suzuki，不代表跨反应迁移已经解决。下一步需要更明确地设计迁移机制，比如把 BH 中学到的 factor evidence 降权加载到 Suzuki，而不是简单共用一套启发式。

## 5. 第三阶段：分子性质任务

这一步是为了回应 AI4Science 文档里的“分子发现”方向。我们没有一上来做复杂生成式分子设计，而是先选了一个真实分子性质数据，把它转成有限候选池搜索。

数据集：

- 名称：`real_moleculenet_esol`
- 类型：真实分子性质数据
- 来源：MoleculeNet ESOL / Delaney
- 候选数：1128
- 任务形式：把每个 molecule 当成一个候选，目标是寻找更优的 property score。
- 注意：这不是 reaction optimization，也不是完整 LogP/QED/SA 多目标优化，只是 molecular property finite-pool replay 的第一步。

结果：

| 方法 | Final Best | AUC | Regret | Top-10 Hit | Interventions | Bad Interventions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent | 88.6412 | 86.1189 | 8.3588 | 0.2333 | 0.0000 | 0.0000 |
| gate_v1 | 89.0674 | 86.1829 | 7.9326 | 0.3000 | 0.7333 | 0.1000 |
| gate_v2 | 89.0674 | 86.1758 | 7.9326 | 0.3000 | 0.7667 | 0.1000 |

怎么讲：

ESOL 上有一个小幅正向信号。final best 从 88.6412 到 89.0674，top-10 hit 从 0.2333 到 0.3000，AUC 也略有提升。这个结果不能说很强，但它说明 CARE replay 可以从 HTE 扩到分子性质候选池，而且 gate 可以在非反应任务上产生可记录的 intervention。

比较稳妥的结论是：ESOL 证明“平台接口可迁移”，还没有证明“科学能力已经跨领域成熟”。

## 6. 第四阶段：ChemLex 代理数据

PPT 里明确提到 CARE 1.0 有 ChemLex 双数据集验证，但我们现在没有拿到真实 ChemLex candidate table，所以先用了 `synthetic_chemlex_i` 做接口测试。

数据集：

- 名称：`synthetic_chemlex_i`
- 类型：合成 ChemLex-style acid-amine optimization
- 候选数：1728
- 作用：验证 ChemLex 类任务形状下，skill/gate/hypothesis 是否能正常工作。
- 限制：不是实验真实数据，不能作为对外 ChemLex 结果。

结果：

| 方法 | Final Best | AUC | Regret | Top-10 Hit | Interventions | Bad Interventions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent | 92.7274 | 88.8560 | 7.2726 | 0.2667 | 0.0000 | 0.0000 |
| gate_v1 | 99.6302 | 98.5616 | 0.3698 | 0.9000 | 6.8667 | 1.8000 |
| gate_v2 | 99.6302 | 98.5616 | 0.3698 | 0.9000 | 6.8667 | 1.8000 |

怎么讲：

ChemLex 代理数据上提升很明显，但这个结果要很小心地讲。它的意义不是“我们已经在 ChemLex 上显著提升”，而是“如果任务里存在可学习的结构化规律，CARE 的 skill + gate 机制能把这个规律转成实验选择收益”。真实 ChemLex 还需要真实数据表确认。

## 7. 补充阶段：Synthetic materials proxy

为了覆盖材料发现方向，我们也跑了一个 materials-shaped synthetic adapter。它不是材料真实数据，只是把材料配方/工艺优化抽象成 finite-pool replay，用来验证 CARE 的接口能不能表达材料类候选搜索。

数据集：

- 名称：`synthetic_materials_i`
- 类型：合成 materials formulation replay
- 候选数：336
- 决策变量：`dopant`, `dopant_ratio`, `anneal_temperature`, `dwell_time`
- 隐藏目标：`stability_score`
- 作用：验证材料方向的数据形状和 CARE replay 接口是否兼容。

结果：

| 方法 | Final Best | AUC | Regret | Top-10 Hit | Interventions | Bad Interventions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent | 93.3509 | 90.4718 | 3.1230 | 0.7000 | 0.0000 | 0.0000 |
| gate_v1 | 95.8318 | 94.0513 | 0.6421 | 0.9333 | 3.3667 | 0.7000 |
| gate_v2 | 95.8318 | 94.0513 | 0.6421 | 0.9333 | 3.3667 | 0.7000 |

怎么讲：

这个结果说明材料类任务形态可以接入 CARE replay，但它仍然是 synthetic proxy。下一步真正有价值的是接公开材料 property dataset，例如 Matbench、QM9 或 Materials Project 中能转成 finite-pool replay 的任务。

## 8. 知识库和 embedding 做了什么

我们不是只做了 replay 脚本，也建了一个 CARE 2.0 的轻量知识库原型。这个知识库不是会议纪要库，而是 CARE 系统用来管理 task、dataset、mechanism、skill、hypothesis 的结构化 memory。

已经做的东西：

1. `seed_cards.json`：放了初始 task/dataset/skill/mechanism card。
2. SQLite + FTS：支持按关键词查 task、dataset、skill。
3. `build_embeddings.py`：支持本地 hashed embedding，也预留 OpenAI-compatible embedding endpoint。
4. `query_embeddings.py`：支持后续用 embedding 做语义检索。
5. `awesome_resources.md`：整理了 AI4Chem、LLM4EDA、materials-aware LLM、molecular discovery 几个 awesome repo 里对我们有用的数据和方向。

这部分和实验的关系是：现在 replay 里的 skill 还是手写的轻量规则；后面如果要做 CARE 2.0 的 self-improving skill library，就需要把每次实验中的假说、证据、失败 case、可复用规则沉淀进知识库，再由 gate 控制哪些能被重新使用。

## 9. 现在能得出的结论

第一，代码和实验框架已经从单一 synthetic task 扩到了多个数据集，包括真实 HTE 和真实分子性质数据。这说明 CARE 2.0 的 platform interface 是可行的。

第二，结果目前不是全面提升。Synthetic Suzuki、synthetic ChemLex 和 synthetic materials 上提升明显，ESOL 有小幅提升，真实 Buchwald-Hartwig 和 Suzuki 当前保守 gate 下基本持平。这是正常的，因为我们现在用的是轻量 public evidence model，还不是完整 CARE 1.0 或更强的 LLM/BO challenger。

第三，真正下一步不是继续堆 synthetic，而是补真实数据和更明确的迁移机制。尤其是 ChemLex、Pfizer 零膨胀数据、材料方向 Matbench/QM9/Materials Project，以及更清楚的 BH -> Suzuki confidence-discount transfer。

## 10. 下一步建议

接下来建议按三个优先级推进。

第一，补数据。真实 ChemLex 和 Pfizer 数据最重要，因为它们直接出现在 PPT/CARE 1.0 叙事里。没有这两个表，我们只能说机制跑通，不能说复现或扩展了 CARE 原始结果。

第二，补更强的 challenger。现在的 challenger 主要是规则化 factor evidence，后面可以接 LLM/API，让 LLM 生成 structured proposal、rationale、skill artifact，但最终仍然由 gate 审查，不让 LLM 直接决定实验。

第三，补跨域任务。分子方向可以从 ESOL 扩到 LogP/QED/SA 多目标；材料方向可以接 Matbench 或 Materials Project 中能转成 finite-pool replay 的 property task。这样就能更贴近“化学、材料、药物多个领域的新物质发现平台”的 CARE 2.0 目标。

