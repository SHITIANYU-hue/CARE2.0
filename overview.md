# Overview

## 1. 这轮实验想验证什么

这轮实验不是在复现 CARE 1.0 论文的最终数字。我们做的是 CARE 2.0 的 replay harness 和跨领域 transfer 验证：先把不同领域的数据统一成有限候选池搜索，再看源领域沉淀下来的 skill / prior 能不能在目标领域带来真实收益。

目前结论比最初更清楚：平台接口已经跑通，而且 transfer 不是只有概念验证。最新 50-seed server sweep 里，分子性质任务 `FreeSolv -> Lipophilicity` 和反应 HTE 任务 `Suzuki-Miyaura -> Buchwald-Hartwig` 都出现了稳定正向 transfer gain。随后补做的真实 LLM follow-up 说明，LLM proposer 在共享 descriptor 的分子性质方向能给出小幅正收益；但在反应 HTE transfer 上，当前 LLM proposer 还不如确定性 transfer card，LLM auditor 也偏保守。最新的强模型 follow-up 进一步说明，换成 `openai/gpt-5.5` 会改善 LLM proposer 的 final best 和坏干预率，但仍没有超过 deterministic transfer rule；`deepseek/deepseek-v3.2` 在当前长上下文 tool-call 接口下反而不稳。

最新补充后，最适合作为“明显正向 transfer”展示的是 `FreeSolv -> Lipophilicity` 的 shared descriptor value-prior sweep。100 seeds 下，`transfer_value_prior_gate_v1` 在 3/5/10 个 reveal budget 上都稳定超过 incumbent：final best 分别提升 +1.1662、+0.9725、+0.8550，AUC 分别提升 +0.4184、+0.6707、+0.7009，top-10 hit 分别从 0.07/0.11/0.17 提到 0.13/0.18/0.23。这条结果比反应 descriptor transfer 更干净，因为 source 和 target 共享同一套 SMILES-derived descriptor vocabulary，不是在迁移数据集内部编号。

具体来说，我们想验证三件事：

1. CARE 的决策流程能否从单一 synthetic task 扩展到真实 HTE 数据。
2. 近邻化学任务之间，例如 Suzuki -> Buchwald-Hartwig，能不能用同一套 gate/evidence 逻辑表达 transfer。
3. 分子性质任务能不能也转成 finite-pool search，并在共享 descriptor 空间里形成更明显的 transfer advantage。

这和 Tianyu 发的 CARE 2.0 PPT 和 AI4Science 跨领域任务分析是一致的。PPT 里提到 CARE 1.0 的 Buchwald-Hartwig、ChemLex、BH -> Suzuki；AI4Science 文档里强调不同领域都有共同结构：在很大的候选空间里找高价值点，把语义知识转成可执行操作，再通过验证闭环确认结果。

## 2. 实验框架怎么做的

我们把每个任务都统一成一个 offline finite-pool replay。也就是说，数据集中已经有一批候选实验点和对应结果，但 replay 时系统不能提前看目标值。每一轮只能根据已经 reveal 的 observation 和 public features 选择下一个点，然后再把这个点的真实结果 reveal 出来。

每轮都有三个角色：

1. `incumbent`：稳健 baseline，只根据已公开观测做选择。
2. `challenger/skill`：CARE 2.0 侧的候选调整逻辑，用已观察到的 factor evidence 或 skill prior 给候选加分/减分。
3. `gate`：审计层，不让 challenger 直接接管实验选择，而是检查它是否有足够 public evidence，是否风险过高，是否偏离太大。

这轮新增了一个 transfer 模式：

- `transfer_value_prior_*`：在源任务和目标任务共享同一套 public descriptor vocabulary 时，允许把源任务里学到的 value-level prior 降权迁移到目标任务。例如 MoleculeNet 里的 `smiles_length_bin`、`hetero_atom_bin`、`aromatic_bin` 这类 SMILES-derived bins。这个模式不会用于反应 HTE 里的 `L00/R00` 等局部标签，因为这些标签只是各数据集内部编号，不能假设跨数据集同义。

我们保留的指标包括：

- `final_best`：最终找到的最好结果。
- `AUC`：整个实验过程里的 best-so-far 曲线面积，越高说明越早找到好点。
- `regret`：距离全局最好点还有多远。
- `top-10 hit`：是否在 replay 过程中命中过全局 top 10 个候选点。
- `interventions`：gate 授权 challenger 覆盖 incumbent 的次数。
- `bad interventions`：授权后结果变差的次数。

这个设置的好处是，最后不只看分数，还能看 gate 到底有没有在控制风险。

## 2.5 规则来源和 incumbent 强度

这里需要单独说清楚：当前 replay 里的 incumbent 不是 CARE 1.0 原版，也不是学术界公认的某个标准 baseline。它是我们为了 CARE 2.0 replay 写的一个透明 target-only control，只用已经 reveal 的目标域 observation 和 public candidate feature，不看 unrevealed 的隐藏结果。

具体规则是：对每个未 reveal 的候选点，综合同组候选的 smoothed mean、各个 decision factor 的 smoothed mean、一个类似 UCB 的 uncertainty bonus，以及 `x1/x2/x3` 这几个 public feature 的轻量 prior。这个 baseline 比 random 强很多，所以 transfer 如果能赢它是有意义的；但它也不是 unbeatable oracle。

我们补了一组 50-seed incumbent ablation 来检查这件事。Buchwald-Hartwig 上，完整 incumbent final best 是 86.6477，random 是 82.1728，factor_ucb 是 86.7960，说明 incumbent 是 strong baseline，但和其他公开证据规则在同一档。Lipophilicity 上，完整 incumbent final best 是 87.3075，factor_only 是 87.5225，factor_ucb 是 87.4550，两个更简单的 factor 规则甚至略高一点。这说明“LLM 没打过 incumbent”不能简单解释为 incumbent 被我们手搓得太强。

当前 deterministic transfer rule 也要按工程规则来理解。它不是 CARE 1.0 直接搬来的规则，而是为了验证 CARE 2.0 跨域迁移写的 role-level transfer card：source domain 只提供哪些 role 更可信、权重多大；大多数方向仍然由 target domain 已 reveal 的 evidence 决定。只有 MoleculeNet 这种共享 descriptor vocabulary 的任务，才允许 source value prior 直接迁移。

LLM 目前确实有真实调用，但角色还比较窄。`llm_transfer_gate_v1` 只是让模型在已有 transfer card 和 target evidence 里提 bounded factor adjustment；`llm_audit_transfer_gate_v1` 只是审计 challenger。它还没有在“进化规则”，比如重写 role map、调 support threshold、选哪些 descriptor 可以迁移，或者产出新的 skill artifact。因此 fixed rule 目前赢 LLM 不算特别反常，下一步更应该把 LLM 往 rule-level proposer 推，而不是继续让它只在一个手写 schema 里微调分数。

我们按“LLM 换强模型可能会提升质量”的方向又补了一组同 seeds 的模型替换实验。Suzuki -> Buchwald-Hartwig 上，`openai/gpt-5.5` 把 `llm_transfer_gate_v1` 的 first-five-seed final best 从 `gpt-4o-mini` 的 86.0649 提到 87.1667，bad interventions 从 2.0000 降到 1.0000，说明模型质量确实有影响。但它仍然低于同 seeds 的 deterministic `transfer_gate_v1`，后者 final best 是 91.5394。`deepseek/deepseek-v3.2` 在小测试里能返回 tool-call JSON，但进入真实长 prompt 后约一半调用没有可用 tool args，最后基本退回 incumbent。这说明下一步不能只换模型，还要改 LLM interface：让模型提出 rule artifact，再由 replay 验证。

我们又补了一组更接近外部优化方法的 target-only surrogate baseline，包括 mixed-kernel GP-UCB、GP-EI 和 kNN-UCB。这组 baseline 只用 public feature 和目标域已 reveal 的 observation，不用 source transfer。结果把结论进一步分开了：`FreeSolv -> Lipophilicity` 上，CARE 的 `transfer_value_prior_gate_v1` final best 是 90.0625，仍然高于 GP-UCB 的 88.4775；但 `Suzuki -> Buchwald-Hartwig` 上，GP-UCB final best 是 91.1145，高于当前 transfer gate 的 89.0866。也就是说，分子性质方向可以说 transfer 赢过了更强 surrogate baseline；反应 HTE 方向目前只能说 transfer 赢 incumbent，但还没有赢 GP-UCB。

这个结果对下一步很有帮助：反应方向不应该继续只和手写 incumbent 比，而应该把 GP-UCB 这类 surrogate 当成更强 incumbent，然后让 CARE transfer 去改 acquisition，或者让 LLM 在 role map、threshold、discount 这些规则层面参与进来。

基于这个判断，我们又跑了一版 hybrid surrogate transfer：直接把 mixed-kernel GP-UCB 当作 incumbent acquisition，再让 CARE transfer card 做 bounded adjustment。结果更接近下一阶段真实目标。`FreeSolv -> Lipophilicity` 上，shared value-prior hybrid 把 GP-UCB final best 从 88.4775 提到 88.8775，AUC 从 86.4578 提到 86.7270，top-10 hit 从 0.08 到 0.16，说明 transfer 叠到更强 optimizer 上仍有小幅正信号。`Suzuki -> Buchwald-Hartwig` 上，hybrid 把 AUC 从 82.9700 提到 83.1863，但 final best 从 91.1145 降到 90.7886，所以反应方向现在还不是 headline win，更像是 acquisition calibration 的诊断结果。

顺着天宇提的“能不能优化 skill，而不是只和弱 baseline 比”的方向，我们又做了一版更直接的 acquisition-level skill optimization：不再让 transfer card 只给候选加 bounded bonus，而是用 source-to-target role confidence 去重加权 GP-UCB 的 categorical kernel。直观说，source domain 只告诉系统哪些 target 字段更值得在相似性判断里重视，例如 Suzuki -> Buchwald-Hartwig 里 ligand、base、aryl_halide 这些 role 的权重会被提高；候选选择仍然由 GP-UCB 完成，不直接读取隐藏结果，也不直接指定 candidate id。

这版结果是一个小幅但更干净的正信号。`Suzuki-Miyaura -> Buchwald-Hartwig` 上，原 GP-UCB final best 是 91.1145、AUC 是 82.9700；`transfer_weighted_gp_ucb_scale_1p5` 提到 final best 91.4146、AUC 83.2862。`scale=4` 的 final best 更高一点，91.4524，但 AUC 降到 82.1020，所以默认更适合讲 `scale=1.5`，因为它同时提升 final best 和搜索过程。分子 sanity check 里，`FreeSolv -> Lipophilicity` 的 `scale=1.5` 也从 GP-UCB 的 final best 88.4775 / AUC 86.4578 小幅提高到 88.5650 / 86.6808。

这个结论要保守讲：现在不是“CARE2.0 已经大幅碾压 GP-UCB”，而是证明了一个关键方向可行：当 reusable skill 进入 acquisition geometry，而不是只做后处理加分时，确实可以在真实 replay 上小幅超过更强 target-only surrogate baseline。下一步应该把 `scale`、kernel field weights、gate threshold 这些东西交给 calibration split 或 LLM rule-level proposer 来选，而不是人工固定。

最新一步已经把这件事推进成 calibration/held-out 实验。我们固定了 11 个 scale 的 grid，用前 50 个 seeds 做 calibration，再只在后 50 个 held-out seeds 上评估。`Suzuki-Miyaura -> Buchwald-Hartwig` 上，balanced/AUC selector 都在 calibration 阶段选中 `transfer_weighted_gp_ucb_scale_1p5`；它在 held-out seeds 上仍然超过 GP-UCB，final best +0.3966，AUC +0.3162，top-10 hit +0.04。这个结果比“看完整结果后挑 scale”更可靠，说明 reaction HTE 的 acquisition-level transfer 确实有一个可校准的正向信号。

同样的 GP-kernel calibration 在 `FreeSolv -> Lipophilicity` 上没有保住 held-out gain：calibration 选出的 scale 在 held-out 上 final 和 AUC 都低于 GP-UCB。因此分子方向现在不应该主讲 kernel reweighting，而应该主讲 shared descriptor value-prior。我们对 value-prior 的 3/5/10 budget 结果也做了 50/50 split，calibration 都选中 `transfer_value_prior_gate_v1`，held-out final delta 分别是 +0.3425、+0.2975、+0.3750，held-out AUC delta 分别是 +0.1900、+0.2605、+0.2168。也就是说，分子方向的正向 transfer 仍然成立，但有效机制不是 GP-kernel field reweighting，而是共享 descriptor value prior。

## 2.6 最新补充：descriptor transfer 要做 target calibration

群里最新讨论后，我们又补了一版 reaction descriptor transfer 的诊断实验。前一版 raw descriptor value prior 的问题很明显：它直接把 source 里某个 descriptor value 的正负方向迁移到 target，Suzuki -> Buchwald-Hartwig 上会造成严重负迁移。新实验改成 `target_calibrated_descriptor_prior`：source descriptor 只负责告诉系统哪些 descriptor value 值得关注，方向和强度由 target 已 reveal 的 observation 决定。

正式结果是：BH -> Suzuki 上，新 strict calibrated 版本没有赢 incumbent，final best 是 92.0578，对 incumbent 是 -0.5507，但 AUC 是 +0.3212，top-10 hit 从 0.10 到 0.20，bad interventions 比 raw descriptor prior 少很多。Suzuki -> BH 上，strict calibrated 版本把 raw descriptor strict 的 final delta 从 -13.2594 修到 +1.3678，AUC delta 从 -9.4342 修到 +1.0884，并超过 incumbent。它还没有超过最强的 role-level `transfer_gate_v1`，后者 Suzuki -> BH 的 final delta 是 +2.4389；所以这不是新的 headline win，但它解释了 descriptor transfer 失败在哪里，也证明通过 target calibration 可以修复负迁移。

这件事对 CARE 2.0 很关键：跨领域迁移不能只问“source 里什么好”，而要问“source 让 target 先看哪里，target 自己的早期证据是否支持这个方向”。下一步 LLM 更适合做 policy selector / rule evolver：在 role transfer、strict transfer、target-calibrated descriptor transfer、incumbent 之间选择，或者调 threshold 和 descriptor whitelist；而不是只在固定 schema 里给候选加一点 bounded adjustment。

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

Suzuki 单任务 replay 上 final best 持平，AUC 略低。这说明只在单个 target 内做保守 factor evidence，不足以形成明显收益。真正有用的信号来自跨任务 transfer，尤其是下面这组 Suzuki -> Buchwald-Hartwig 结果。

### 4.3 最新反应 transfer：Suzuki-Miyaura -> Buchwald-Hartwig

这是目前反应 HTE 方向最重要的正例。我们用 Suzuki-Miyaura 作为 source domain，Buchwald-Hartwig 作为 target domain，只迁移 role-level evidence，不直接迁移 dataset-local 的具体 ligand/base label。

设置：

- Source：`real_suzuki_miyaura`
- Target：`real_buchwald_hartwig`
- Seeds：50
- Initial observations：5
- Reveal budget：10
- Source observations：96
- Transfer 方式：role-level transfer card + gate

结果：

| 方法 | Final Best | Delta vs Incumbent | AUC | Delta AUC | Top-10 Hit | Bad Interventions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_care_random | 82.1728 | -4.4749 | 77.4347 | -2.5366 | 0.0600 | 0.0000 |
| incumbent | 86.6477 | 0.0000 | 79.9713 | 0.0000 | 0.1600 | 0.0000 |
| transfer_gate_v1 | 89.0866 | +2.4389 | 81.0779 | +1.1066 | 0.2400 | 0.8600 |
| transfer_strict_gate_v1 | 87.6158 | +0.9681 | 80.2528 | +0.2815 | 0.1600 | 0.1800 |

怎么讲：

这组结果可以作为 CARE 2.0 反应迁移的主结果。它比单任务 BH replay 更有说服力：random baseline 明显差于 incumbent，而 `transfer_gate_v1` 在 50 seeds 下同时提高 final best 和 AUC。也就是说，Suzuki 中学到的 role-level evidence 确实帮助 Buchwald-Hartwig target replay 更早、更稳定地找到好实验区域。

同时，这里也能看到 gate 的 tradeoff：普通 transfer gate 收益更大，但有一定 bad interventions；strict gate 更安全，但收益变小。下一步应该做的是 gate calibration，而不是否定 transfer 本身。

补做的真实 LLM follow-up 结果更像一个边界检查。10 seeds 下，确定性 `transfer_gate_v1` 仍然最强，final best 从 incumbent 的 86.6260 提到 90.0980；但 `llm_transfer_gate_v1` 只有 85.6227，低于 incumbent，`llm_audit_transfer_gate_v1` 基本回到 incumbent。这里的结论不是 LLM 接不进来，事实上 70 次 LLM proposer 调用全部 parse 成功；问题是当前 prompt/约束下，LLM 还没有学会比确定性 role-level transfer 更好地使用反应 HTE evidence。

最新强模型 follow-up 用同样的 seeds 0-4 做了更公平的小对照。`gpt-5.5` 相比 `gpt-4o-mini` 有改善：`llm_transfer_gate_v1` final best 从 86.0649 到 87.1667，bad interventions 从 2.0000 到 1.0000。但 `transfer_gate_v1` 仍然是 91.5394，所以这不是“换模型就赢了”，而是“更强模型能改善 proposer 质量，但当前 adjustment schema 仍然不够”。`deepseek-v3.2` 的主要问题是长 prompt + tool-call 不稳，parse errors 均值 3.8，最后没有有效 intervention。

## 5. 第三阶段：分子性质任务

这一步是为了回应 AI4Science 文档里的“分子发现”方向。我们没有一上来做复杂生成式分子设计，而是先选真实分子性质数据，把它转成有限候选池搜索。

第一组数据集：

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

### 5.2 最新分子 transfer：FreeSolv -> Lipophilicity

这是目前最能展示 transfer 明星优势的一组结果。FreeSolv 和 Lipophilicity 都是 MoleculeNet 分子性质任务，并且共享一批 public SMILES descriptor bins。所以这里除了 role-level transfer，我们还加入了保守的 shared-vocabulary value prior。

设置：

- Source：`real_moleculenet_freesolv`
- Target：`real_moleculenet_lipophilicity`
- Seeds：50
- Initial observations：5
- Reveal budget：10
- Source observations：192
- Transfer 方式：role-level transfer card + shared descriptor value prior + gate

结果：

| 方法 | Final Best | Delta vs Incumbent | AUC | Delta AUC | Top-10 Hit | Bad Interventions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_care_random | 85.7650 | -1.5425 | 84.2917 | -1.2743 | 0.0200 | 0.0000 |
| incumbent | 87.3075 | 0.0000 | 85.5660 | 0.0000 | 0.0400 | 0.0000 |
| transfer_value_prior_gate_v1 | 90.0625 | +2.7550 | 87.9660 | +2.4000 | 0.3000 | 3.4600 |
| transfer_value_prior_strict_gate_v1 | 88.1625 | +0.8550 | 86.1820 | +0.6160 | 0.1600 | 1.9000 |

怎么讲：

这组结果最适合用来说明 CARE 2.0 的跨领域 transfer 是有明星优势的。`transfer_value_prior_gate_v1` 不只是 final best 提升，AUC 也提升了 2.4，说明它不是最后偶然撞到一个好点，而是在整个 replay 过程中更早进入高价值区域。top-10 hit 从 incumbent 的 4% 提到 30%，这个信号很直观。

需要同时讲清楚限制：这个模式只适合共享 descriptor vocabulary 的任务。它比 strict gate 更激进，所以 bad interventions 也更多。当前它证明了 transfer 的上限和潜力，下一步要把这个优势和更好的安全 gate 结合起来。

再往强 baseline 上推，100-seed hybrid GP-UCB 结果更保守，但也更有说服力。10 轮预算下，`hybrid_value_prior_gp_ucb_gate_v1` 把 GP-UCB 的 final best 从 88.9038 小幅提到 88.9225，AUC 从 86.5433 提到 86.7037，top-10 hit 从 0.11 提到 0.21。final best 基本接近持平，但 top-10 hit 接近翻倍，说明现在的迁移更像是在帮助 early discovery，而不是完全替代 target-only GP。低预算时这个信号更明显：5 轮预算下 final best +0.1463、AUC +0.3117；3 轮预算下 final best +0.2774、AUC +0.4091。

我们也试了 warm-start，只让 transfer 影响前 3 或 5 轮，然后把控制权交还给 GP-UCB。这个方向没有成为更强策略：warm3 只剩很小收益，warm5 的 final best 还低于 GP-UCB。也就是说，对 FreeSolv -> Lipophilicity 这组共享 descriptor transfer 来说，目前更有效的是让 value-prior skill 在整个短预算 replay 中持续参与，而不是只做开局引导。

真实 LLM follow-up 在这个方向给了一个更积极的信号。10 seeds 下，`llm_transfer_gate_v1` final best 是 87.1750，比 incumbent 高 +0.5250；AUC 是 85.6050，比 incumbent 高 +0.6950；top-10 hit 从 0 提到 0.1。它和确定性 `transfer_value_prior_gate_v1` 很接近，说明在共享 descriptor 空间里，LLM 可以读 transfer card 和 target evidence，并生成有用的 bounded adjustment。`llm_audit_transfer_gate_v1` 仍然太保守，基本没有带来增益。

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

最新补的知识库卡片把 skill transfer 明确拆成六层，而不是只停留在 acquisition 层：representation transfer 负责 source/target 字段映射，mechanism transfer 负责可复用科学假说，model transfer 负责 kernel/embedding/feature transform，acquisition transfer 负责候选排序和探索策略，gate/risk transfer 负责识别 negative transfer，workflow transfer 负责实验预算、审计和数据边界。`transfer_weighted_gp_kernel` 现在只是其中一个 model/acquisition binding；真正的 CARE 2.0 skill artifact 应该把这六层一起记录下来。

## 9. 现在能得出的结论

第一，代码和实验框架已经从单一 synthetic task 扩到了多个数据集，包括真实 HTE 和真实分子性质数据。这说明 CARE 2.0 的 platform interface 是可行的。

第二，现在已经有真实数据 transfer 正例，但强度要分开讲。分子性质方向，`FreeSolv -> Lipophilicity` 在 50 seeds 下 final best 提升 +2.7550，AUC 提升 +2.4000，top-10 hit 从 4% 到 30%，这是 public-incumbent 设置下最亮眼的 transfer 上限。进一步把 transfer 叠到更强的 GP-UCB 上，100-seed 结果不再是大幅 final-best 碾压，但仍有 early-discovery 增益：10 轮预算 top-10 hit 从 0.11 到 0.21，5 轮和 3 轮低预算下 final best / AUC 都稳定为正。transfer-weighted GP kernel 的 `scale=1.5` 也有小幅正收益，final best +0.0875、AUC +0.2230。反应 HTE 方向，`Suzuki-Miyaura -> Buchwald-Hartwig` 相比 public incumbent 有提升，final best +2.4389，AUC +1.1066。GP-UCB target-only baseline 更强以后，简单 additive hybrid 还没赢 final best；但 transfer-weighted GP kernel 已经把 GP-UCB 从 final best 91.1145 / AUC 82.9700 提到 91.4146 / 83.2862。这说明反应方向不是只能赢弱 incumbent，skill 进入 acquisition geometry 后已经有小幅超过强 baseline 的信号。

第三，结果还不是“所有方向都提升”。BH -> Suzuki 这类反向迁移目前不稳定，ChemLex 和材料方向还需要更强的真实数据与更明确的 transfer map。这个边界反而是有价值的：CARE 2.0 不是盲目把 source knowledge 往 target 上套，而是要识别什么时候能迁移，什么时候应该保守。

## 10. 下一步建议

接下来建议按四个优先级推进。

第一，继续补真实数据。真实 ChemLex、Pfizer 零膨胀数据和材料方向 Matbench / Materials Project 仍然重要。现在 FreeSolv -> Lipophilicity 是更强的 transfer 正例；Suzuki -> BH 是正向但需要进一步优化的反应方向结果。下一步要看这些 transfer 机制能不能继续扩到 ChemLex 和材料 property task。

第二，做 gate calibration。当前最强的 value-prior transfer 能打出明显优势，但 bad interventions 也变多。下一版应该保留它的 top10 hit 和 AUC 优势，同时用 target confirmation、risk-aware gate 或 LLM audit 降低坏 intervention。

第三，继续做 acquisition-level skill optimization。现在 challenger 主要是规则化 factor evidence、shared descriptor prior，以及一版真实 LLM proposer。新 baseline 显示 GP-UCB 在反应 HTE 上很强，简单 additive transfer adjustment 不够；最新 transfer-weighted kernel 说明，把 skill 用来改 GP kernel field weights 是可行方向。下一步应该把 `scale`、posterior mean/uncertainty/exploration weight、candidate filtering 和 gate threshold 放进一个 calibration/search loop。LLM 也应该产出更受约束的 structured proposal、rationale 和 skill artifact，再由 gate 审查，而不是让 LLM 直接决定实验。强模型 follow-up 支持这个判断：`gpt-5.5` 能改善 bounded adjustment，但仍没有超过 deterministic rule；更值得做的是让 LLM 搜 rule，而不是只让它调候选分数。

最新 prompt follow-up 进一步确认了这点。我们按群里反馈把 LLM 从“像 AI 审稿一样评论”改成“实验策略 proposer”：prompt 变短，要求只根据 transfer card 和已揭示 target evidence 提规则；parser 强制校验 prefer/penalize 方向；v3 版本还把 LLM weight 变成建议上限，实际权重由 target support、target effect 和 transfer role weight 重新校准，同一候选命中多条 LLM 规则时取平均信号而不是直接累加。结果是 v3 的 LLM proposer 比直接加权版本更稳，AUC 到 80.8906，bad interventions 降到 0.6；但 final best 只有 86.1182，仍低于 deterministic `transfer_gate_v1` 的 91.5394。1200-token 重跑把 parse error 降到 0，但指标没有变好，说明瓶颈不是 JSON 截断，而是 LLM 选择规则本身还不够强。这个结果不应该包装成 LLM 已经赢了，而应该作为下一步 rule evolution / policy selector 的依据。

第四，补跨域任务。分子方向可以从单属性扩到 LogP/QED/SA 多目标；材料方向可以接 Matbench 或 Materials Project 中能转成 finite-pool replay 的 property task。这样就能更贴近“化学、材料、药物多个领域的新物质发现平台”的 CARE 2.0 目标。
