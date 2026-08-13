# CARE 2.0 LLM 参与版：精简版逐页中文 Speaker Note

对应 PPT：`CARE2.0_LLM_participating_status_2026-08-12.pptx`

本讲稿对应 18 页精简汇报版。每页先讲“本页核心”，再按“详细讲解”展开；遇到追问时使用“名词解释”；最后用“转场”自然进入下一页。讲解中刻意区分了已经成立的结果、离线证据和仍待验证的结论。

## 第 1 页：封面：CARE 2.0 的定位

**本页核心：** CARE 2.0 的重点不是发明一个永远获胜的优化器，而是建立一套可以验证、拒绝和复用历史实验经验的迁移框架。

**详细讲解：**

开场先把项目定位讲清楚。CARE 2.0 研究的是：过去做过的实验，能不能在一个新实验刚开始、数据很少的时候帮助我们更快选到好条件。这里的 source 是已经完成、结果已知的历史实验，target 是准备开始的新实验。我们希望把 source 中真正有用的经验整理成可执行的 Skill，再用它改变 target 的开局；如果证据不足，就拒绝迁移，回到 target-only 方法。

右侧四个词可以按顺序解释：历史实验提供数据；系统把数据整理成可执行 Skill；Skill影响新实验的决策；新实验的成功和失败再回到知识库。最近最重要的进展，是 v1 在独立验证中失败后，我们没有隐藏失败，而是根据失败 trace 修改出 v2，并在一个冻结后才执行的外部 Suzuki target 上看到明显提升。

**名词解释：** source=历史任务；target=新任务；Skill=带执行逻辑、适用边界和证据记录的策略包。

**转场：** 下一页进入：最新状态：先给结论。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-status-and-ppt/CARE2_CURRENT_STATUS_CN.md`

## 第 2 页：最新状态：先给结论

**本页核心：** 固定 v2 已有一个冻结外部正例；随后完成的真实 LLM 组件实验在五个 target 上平均 AUC 提高 2.12，但仍是回顾性证据。

**详细讲解：**

这一页把两条证据分开讲。第一条是严格的外部证据：Frozen v2 在 Suzuki MINLP2 上相对更强 target-only baseline 的 best-so-far AUC 提高 8.66。这里的 AUC 表示搜索过程中多早找到高质量实验。这个结果在 target outcome 揭示前已经冻结，但该切片没有使用 LLM。

第二条是随后完成的真实 LLM 组件实验。我们通过 CommonStack 实际调用 openai/gpt-5.4，让模型在看不到 target outcome 的情况下生成 initial-design hypothesis；按任务结构选择 raw 或 compiled 方案后，五个 target 的平均 AUC 相对 fixed v2 提高 2.12，4/5 不下降。不过 95% 置信区间是 [-0.226, 4.473]，而且这五个 target 以前都被项目使用过，所以这是回顾性组件证据，不是新的外部确认。

**名词解释：** AUC=搜索全过程的效率；Frozen=策略和参数在看 target 结果前已锁定；回顾性组件实验=在已有任务上检查模块行为，不能替代全新 target。

**转场：** 下一页进入：问题定义：有限预算的逐轮实验。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-v2-external-confirmation/README.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/README.md`

## 第 3 页：问题定义：有限预算的逐轮实验

**本页核心：** CARE 2.0 不是普通预测任务，而是每轮只能做一个实验、做完才看到结果的序贯决策问题。

**详细讲解：**

左边是算法开始前允许知道的信息：已经完成的 source experiments、target 的变量和候选条件、固定预算，以及尚未揭示的 target outcome。中间是每一轮真实发生的事：算法选择一个候选条件，只看到这个被选条件的结果，然后更新 target model，再做下一轮选择。右边是目标：在相同预算下更早找到高质量候选，同时避免因为错误迁移浪费实验。

这里与普通训练集、测试集的区别是，算法不能一次看到完整 target 表格。它必须像真实实验一样逐轮揭示 outcome。我们真正比较的是 source history 是否改变了 target 的选点顺序，以及这种改变是否让有效实验更早发生。

**名词解释：** candidate=可选择的实验条件；outcome=实验结果，例如产率；budget=最多允许做多少次 target 实验。

**转场：** 下一页进入：v2 执行架构。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/CARE2_METHOD.md`

## 第 4 页：v2 执行架构

**本页核心：** source 只负责前三个初始实验，之后所有方法都使用完全相同的 target-only GP-UCB。

**详细讲解：**

从左向右讲。首先，每个历史 campaign 单独训练一个 source expert。因为不同 campaign 的 outcome 尺度不同，我们不直接拼接原始数值，而是把每个 expert 的预测转成候选排序，再用中位数形成共识。TransferSkill 根据这个共识选择三个初始实验。

关键控制在黄色分界线之后：三个初始点做完后，source 信息完全退出。随后 10 轮所有方法都运行同一个 target-only GP-UCB，只用已经揭示的 target outcomes 建模。这样，Frozen v2、随机初始化和 space filling 之间唯一的区别就是前三个点怎么选，后续优化器、预算和观测规则完全一致。因此外部结果可以归因于 source-guided initial design，而不是后续偷偷使用了不同算法。

**名词解释：** source expert=在单个历史 campaign 上训练的模型；initial design=优化开始前先做的少量初始实验。

**转场：** 下一页进入：Replay Harness 如何运行。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`

## 第 5 页：Replay Harness 如何运行

**本页核心：** 完整数据表只充当隐藏环境；算法每轮只能看到自己已经选择过的实验结果。

**详细讲解：**

Replay 的做法是把已有真实数据集当作一个可以查询的实验环境，而不是把整张表交给算法。开始时 target outcomes 全部隐藏。每一轮算法选择一个 candidate，环境只揭示这个点的真实 outcome，算法据此更新模型，再选择下一点。每轮的 candidate、score、outcome 和当前 best-so-far 都进入 trace。

所有对照使用相同 candidate pool、相同初始点数量、相同 reveal budget 和相同 seeds。这里 seed 是随机数种子，用来复现实验随机性，不是提前挑选的好实验点。Replay 仍然不能替代新的 wet-lab 实验，但它能严格检查数据泄漏，并评价有限预算下哪种策略更早找到好条件。

**名词解释：** seed=控制随机过程的编号；reveal=执行一个候选后揭示结果；trace=逐轮决策记录。

**转场：** 下一页进入：Hypothesis 的生成与执行。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/CARE2_METHOD.md`

## 第 6 页：Hypothesis 的生成与执行

**本页核心：** Hypothesis 只有被编译成明确的选点和停止规则，才真正进入实验。

**详细讲解：**

完整平台给 hypothesis generator 的输入包括 source history、target 的变量角色、候选空间、预算以及 SkillBank 中的成功和失败案例。输出不能只是“这个 source 看起来相似”，而必须结构化为：适用哪些 source、变量如何对齐、什么条件应该优先或惩罚、允许进入什么质量区域、置信度多高，以及在什么情况下应该失败或回退。

结构化输出再被编译成 TransferSkill，用来选择 candidate。target outcome 揭示后，系统更新 evidence status：支持、反驳、暂缓或者拒绝。最新 v2 的 hypothesis 不是 LLM 生成，而是固定规则，目的是先验证从 hypothesis 到执行再到反馈的链路。

**名词解释：** role map=source 和 target 变量的语义对应关系；operator=把 hypothesis 转成选点动作的执行算子。

**转场：** 下一页进入：v1 失败如何变成 v2 质量约束。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/CARE2_METHOD.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`

## 第 7 页：v1 失败如何变成 v2 质量约束

**本页核心：** v2 先限定 source 共识的高质量区域，再在区域内部选择互补实验。

**详细讲解：**

v1 的第一个点取 source 排名最高的候选，第二、第三个点主要追求距离远、覆盖广。问题是“离得远”不代表“质量仍然好”，所以多样性可能把点推入 source 自己也不看好的区域。

v2 保留第一个 source consensus 最优点，但先建立 top-50% admissible region，也就是只允许第二、第三个点从 source 排名前一半的候选中产生；然后在这个质量区域内用 maximin 选择与已选点距离最大的候选。它同时保留质量和覆盖。前三个点选完后 source prior 退出，后面仍是统一的 target-only GP-UCB。

这里的 top-50% 是当前冻结协议中的经验参数，不应称为学界通用常数，后续需要做 threshold sensitivity。

**名词解释：** admissible region=允许选点的候选区域；maximin=选择与现有点最小距离最大的候选，用于扩大覆盖。

**转场：** 下一页进入：评价指标。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`

## 第 8 页：评价指标

**本页核心：** 主指标看整个搜索过程，辅助指标看最终质量、胜负、负迁移和实验节省。

**详细讲解：**

Primary metric 是 best-so-far AUC。每一轮记录到目前为止观察到的最好 outcome，再对整条曲线求平均或面积。它奖励“尽早找到好条件”，特别适合实验昂贵的场景。Secondary metrics 包括 final best、达到阈值所需轮数、win/non-loss、negative transfer rate 和 gate acceptance rate。

需要区分两种统计单位。对多个独立 campaign 的开发或验证结果，可以在 task level 计算置信区间；同一个 target 上的 100 个 random seeds 只描述随机初始化分布，不能当作 100 个独立科学任务。因此外部 target 的 +8.66 是很大的实际 effect，但暂时不能称为跨任务统计显著。

**名词解释：** best-so-far=截至当前轮找到的最好结果；CI=置信区间；effect size=提升幅度本身，而非只看 p 值。

**转场：** 下一页进入：Calibration Gate。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-v2-external-confirmation/README.md`

## 第 9 页：Calibration Gate

**本页核心：** Gate 先检查稳定性和负迁移风险，再从合格策略中选择平均收益最高者。

**详细讲解：**

第一步是资格检查：开发任务上 AUC delta 的 95% CI lower bound 必须大于 0，同时 task non-loss rate 至少为 0.8。前者要求平均收益具有统计支持，后者要求不能靠少数大胜掩盖大量失败。第二步只在 eligible routes 中选择 mean AUC gain 最大的策略，CI lower bound 用作并列时的次级标准。

要诚实说明阈值来源：CI lower > 0 有明确统计含义；non-loss 0.8 是本项目预先冻结的经验门槛，不是普遍公认常数。后续应对 0.7、0.8、0.9 以及 admissible region 阈值做 sensitivity analysis，确认结论不依赖单一超参数。

**名词解释：** gate=迁移准入与路由机制；non-loss=相对 baseline 没有变差的任务比例；eligible=通过资格检查。

**转场：** 下一页进入：Baseline Protocol。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`

## 第 10 页：Baseline Protocol

**本页核心：** Frozen v2 必须同时超过随机初始化和确定性的空间覆盖，对比口径完全一致。

**详细讲解：**

第一列 Frozen v2 的前三个点由 source-guided skill 选择；第二列 random-initial GP-UCB 的前三个点随机选择，并用 100 个 seeds 描述随机性；第三列 space filling 用确定性规则最大化候选空间覆盖。三者之后都运行同一个 target-only GP-UCB、做 10 次 reveal，而且都不能预读 target outcome。

Random baseline 不是每轮都随机。只有前三个初始点随机，后续仍然是强的 GP-UCB。Space filling 也不是弱对照，它能给 GP 一个分散且覆盖面好的初始设计。主比较对象是每个 campaign 上表现更强的 target-only baseline，而不是挑一个最弱对照。

**名词解释：** space filling=优先覆盖参数空间的确定性初始设计；stronger baseline=在给定 target 上两种 target-only 对照中更强者。

**转场：** 下一页进入：从 v1 到 v2 的证据链。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`

## 第 11 页：从 v1 到 v2 的证据链

**本页核心：** 只有冻结后外部正向结果能支持泛化；开发结果和 post-hoc 结果不能替代它。

**详细讲解：**

从左到右看五根柱。v1 在 9 个开发任务上 +1.32，但在 4 个独立任务上变成 -1.20；这说明 v1 没有泛化。v2 在 9 个开发任务上 +1.39，说明它可以成为新的冻结候选，但仍不算外部证据。真正改变项目状态的是 v2 在外部 Suzuki MINLP2 上 AUC +8.66。最右侧旧四任务 post-hoc 约 +0.38，只能支持修复方向，不能当成新的 held-out evidence。

汇报时要强调这条逻辑：开发用来选策略，独立验证用来检验策略；独立验证失败后可以分析和修复，但旧验证集从此变成开发信息，不能反复计算成新的泛化证据。

**名词解释：** held-out=冻结前未用于选择策略的数据；post-hoc=看过结果以后进行的分析。

**转场：** 下一页进入：外部结果：过程与终点同时改善。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-confirmation/README.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-v2-external-confirmation/README.md`

## 第 12 页：外部结果：过程与终点同时改善

**本页核心：** Frozen v2 的优势不只体现在最终最好值，也体现在更早进入高产率区域。

**详细讲解：**

左图比较 best-so-far AUC：Frozen v2 为 98.06，space filling 为 89.40，100 次 random 的均值为 88.84。因为 space filling 是两个 target-only 对照中更强者，主结果写成相对 stronger baseline +8.66。右图比较预算结束时的 final best：v2 找到 100% 产率，space filling 为 91.82，random 平均为 94.77。

100 个 random runs 中，只有 15 个 AUC 达到或超过 v2，说明 v2 位于随机初始化分布的高端。这里要避免把 100 seeds 写成 100 个独立任务；它们只说明同一个 target 上随机初始化的波动。最稳妥的结论是“在一个外部 target 上观察到大的实际提升”。

**名词解释：** final best=预算结束时找到的最高 outcome；random mean=同一算法更换随机初始化后的平均结果。

**转场：** 下一页进入：传统 Transfer BO 对比。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-v2-external-confirmation/README.md`

## 第 13 页：传统 Transfer BO 对比

**本页核心：** RGPE、ICM 和混合路由各有优势，但没有一个固定迁移方法在所有任务都可靠。

**详细讲解：**

RGPE 会根据 source 模型在 target 观测上的排序损失给不同模型加权。它在 dielectric 到 experimental gap 上非常强，但在另外多条 final-score 比较中显著变差，说明经典方法也会负迁移。Two-task ICM GP 通过多任务协方差联合建模，整体更保守，但仍不是普遍最优。Hybrid router 在 calibration 信息帮助下选择方法，AUC 在五条路径上为正，支持路由机制的价值。

必须解释深蓝框中的限制：在这五条路径里，CARE 自己的 source-transfer candidates 全部被 gate 拒绝，所以 CARE 的正向结果主要来自 target-only fallback。不能把“系统避免了负迁移”写成“CARE source transfer 全面击败传统方法”。当前平台价值是比较、路由、拒绝和审计。

**名词解释：** RGPE=按目标任务上的排序表现给多个 GP 专家加权；ICM=用任务间协方差联合建模的多任务 GP。

**转场：** 下一页进入：真实 LLM 如何进入执行链。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-08-classical-transfer-confirmation/README.md`

## 第 14 页：真实 LLM 如何进入执行链

**本页核心：** LLM 不再只是写解释，而是在看不到 target outcome 的情况下冻结一个可执行 initial-design hypothesis。

**详细讲解：**

这一页讲清楚本轮新增的真实 LLM 调用。输入包括完成的 source outcomes、source 层统计、target schema、公开候选条件和 source-prior 排名短名单；明确不包含任何 target outcome。使用 CommonStack 上的 openai/gpt-5.4，模型输出一个 JSON hypothesis：三条 candidate IDs、各自角色、机制解释、置信度、失败条件和是否 abstain。原始 prompt、原始 response、token usage、API timing 和 SHA-256 都保存。

输出有两种执行方式。Raw LLM 直接执行模型选出的三个点；Compiled LLM 把模型提供的语义锚点，与 source 共识最优点和 top-50% 质量区内的 maximin 几何探针组合。三点以后，两种方法都退出 source 和 LLM，统一运行 target-only GP-UCB。这样 LLM 真正改变了实验开局，但后续预算和优化器仍可公平比较。

**名词解释：** semantic anchor=LLM 根据科学机制挑出的重点条件；compiler=把语义建议转换成满足质量和几何约束的执行方案；abstain=LLM 主动拒绝迁移。

**转场：** 下一页进入：Suzuki LLM Case Study。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/README.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/scripts/run_llm_initial_design_hypothesis.py`

## 第 15 页：Suzuki LLM Case Study

**本页核心：** LLM 给出了正确的反应区域，但直接选三点覆盖不足；安全编译后 AUC 反超 fixed v2。

**详细讲解：**

真实模型在不知道 MINLP2 产率的情况下提出：同底物 source 支持高温、高 loading 的 P2L1 XPhos Cl 区域，温度是主要迁移因子，置信度 0.73。Raw LLM 选了三个都集中在该催化剂家族的点，虽然初始质量不错，但给后续 GP 的几何信息不足，AUC 只有 91.819，低于 fixed v2 的 98.061。

安全 compiler 没有丢掉 LLM。它保留 LLM 的 candidate 024 作为 semantic anchor，加入 source 共识最优 candidate 039，再从 source top-50% 区域中选择 maximin geometry probe 013。这个三点组合 AUC 达到 100，比 fixed v2 高 1.939，二者 final best 都是 100%。这说明合理角色不是让 LLM 替代数值优化，而是让 LLM提出可解释的科学方向，由 compiler 保证可优化性。

**名词解释：** Raw LLM=直接执行三点；Compiled LLM=保留一条 LLM 语义决策并加确定性安全约束；AUC=越早找到好结果越高。

**转场：** 下一页进入：五任务 LLM 组件结果。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/suzuki_minlp2/evaluation_compiled/summary.json`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/suzuki_minlp2/llm_hypothesis_record.json`

## 第 16 页：五任务 LLM 组件结果

**本页核心：** 任务结构路由在五个 target 上平均 AUC 比 fixed v2 高 2.12，4/5 不下降，但置信区间仍跨零。

**详细讲解：**

五个 target 都是真实 API 调用，共五条主 hypothesis、66,293 tokens。候选路由完全由任务结构决定：只有一个 completed source 时采用 compiled LLM；有多个 source 时采用 raw semantic design。每个 target 的后续 GP、预算和 candidate pool 与 fixed v2 完全一致。

逐任务 AUC delta 是：Suzuki +1.939，Morpholine-AlPhos +5.203，Phenethylamine-AlPhos -0.947，Morpholine-tBuBrettPhos +4.422，preliminary 0。平均 +2.123，win 3/5，non-loss 4/5，final best 5/5 不下降。95% CI 为 [-0.226, 4.473]，仍然跨 0，所以不能写成统计上已经证明 LLM 普遍优于 fixed rule。

还要说明，这条路由是在观察组件行为后形成，五个 target 也都曾被项目使用，因此属于 retrospective ablation。正确下一步是冻结 prompt、compiler 和 route，在全新的 target 上执行。

**名词解释：** 任务结构路由=按 source 数量选择 raw 或 compiled，不读取 target outcome；retrospective ablation=回顾性组件实验，不是新的外部确认。

**转场：** 下一页进入：结论边界。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/aggregate/suite_summary.json`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/README.md`

## 第 17 页：结论边界

**本页核心：** 已经成立的是可信迁移流程、一个外部固定规则正例和真实 LLM 的正向组件信号；尚未成立的是 LLM 外部普遍优势。

**详细讲解：**

左栏是可以明确说的：我们建立了统一、可审计的 replay harness；固定 v2 在一个真实外部 Suzuki target 上相对强 baseline 的 AUC 提高 8.66；真实 LLM 已经生成并执行结构化 hypothesis；五任务回顾性组件实验平均 AUC 提高 2.12。

中栏是不能说的：任意跨领域稳定迁移、LLM 已在全新 target 普遍优于 fixed rule、task-level CI 已排除零增益，以及 source 在优化全程持续贡献。右栏是项目真正的定位：可信赖的科学知识迁移框架，强调强 baseline、冻结协议、负迁移审计、exact fallback，以及成功和失败共同沉淀。

一句话收束：历史实验可以形成可执行、可拒绝、可验证的 Skill，但每个泛化结论都要由冻结后的新任务来支持。

**名词解释：** 可信迁移不等于每次都迁移；能够识别不该迁移并安全回退，也是系统能力。

**转场：** 下一页进入：下一步实验。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-status-and-ppt/CARE2_CURRENT_STATUS_CN.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/README.md`

## 第 18 页：下一步实验

**本页核心：** 优先冻结 prompt、compiler 和 route，再用真正外部 target 检验 LLM 的独立增量。

**详细讲解：**

P0 第一项是扩大 external targets：冻结同一个 v2 skill family，在更多未参与开发的 C-N、Suzuki 或相邻 reaction campaigns 上执行。只有多个独立 target 才能计算 task-level CI。P0 第二项是冻结统一 Confirmation Protocol，包括任务根列表、预算、primary metric、selection hash 和 skill fingerprint，防止每个结果出来后改变口径。

针对 LLM，还要额外冻结 prompt、模型版本、raw/compiled route、compiler 参数和 trace schema。面对新 target 时不能再根据结果选择哪条路线。确认实验仍然比较 fixed v2、LLM hypothesis-only 和 matched random hypothesis，所有组使用相同 source、candidate pool、initial-design size 和 target budget。之后再研究 skill accumulation 和远距离领域扩展。

**名词解释：** Confirmation Protocol=事先冻结任务、预算、指标和版本的确认性实验协议；skill accumulation=随着完成任务增加，SkillBank 是否带来可重复收益。

**转场：** 汇报结束，进入讨论。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-status-and-ppt/CARE2_CURRENT_STATUS_CN.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/README.md`
