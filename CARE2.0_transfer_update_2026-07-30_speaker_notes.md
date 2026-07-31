# CARE 2.0 Transfer Update Speaker Notes

配套文件：`CARE2.0_transfer_update_2026-07-30.pptx`  
建议汇报时间：20-25 分钟，讨论时间另计。  
核心表述：CARE 2.0 当前验证的是“同领域、跨数据集的可信迁移”，不是任意学科之间的普适迁移。

## Slide 1｜CARE 2.0：可信赖的跨任务科学知识迁移框架

### 建议讲稿

这次更新关注的不是再寻找一条更强的固定优化规则，而是一个更基础的问题：历史实验经验能否被整理成可执行、可验证、也可以被拒绝的迁移策略。

CARE 2.0 把一次迁移拆成完整的证据链：先从 source 任务提取经验，生成迁移 Hypothesis，再编译成实际影响 Bayesian optimization 的策略补丁；只有通过独立 calibration 的路线才进入正式 target replay，否则精确回退到 target-only 方法。因此，这里的“可信赖”包含两层含义：证据充分时可以获得正向迁移，证据不足时不会强行使用 source prior。

### 本页需要强调

- 贡献不是某一条人工规则，而是“生成、执行、验证、回退、沉淀”的迁移框架。
- 当前结果来自 frozen replay，不是展示性单次运行。
- “跨任务”目前主要指同一科学领域中的不同真实数据集。

### 过渡

接下来先明确我们究竟在解决什么问题，以及目前的证据边界在哪里。

## Slide 2｜核心研究问题

### 建议讲稿

我们把问题定义为：在固定 target 实验预算的条件下，系统能否利用 source 的历史实验与 outcome，比只看 target 数据的强基线更快、更稳定地找到高质量候选，同时避免负迁移。

这里有三个评价维度。第一是最终质量，也就是预算结束时找到的最好结果；第二是过程效率，用 best-so-far AUC 和达到同等质量所节省的实验轮数衡量；第三是安全性，包括负迁移率、Gate 接受率和回退是否与 matched target-only 完全一致。

当前实验证据覆盖反应、材料和分子三个领域，但迁移发生在各自领域内部。例如材料性质数据集之间迁移、分子性质数据集之间迁移。我们还没有用正式实验支持“化学经验可以稳定迁移到材料”这样的远距离跨领域结论。

### 可能追问

**为什么不直接和 random search 比？**  
Random search 过弱，只能作为完整性对照。主比较对象是 GP-UCB、GP-EI、target acquisition portfolio，以及 matched target-only LLM。

**这里的泛化是什么？**  
不是在同一个 source-target pair 上换 seed，而是在多个预先固定的 source-target pair 上使用同一套生成、校准和部署协议。

### 过渡

在这个问题定义下，整个系统从历史实验到 target 决策的流程如下。

## Slide 3｜可信迁移的总体框架

### 建议讲稿

框架从 Historical Experiments 开始。历史数据首先被整理成统一的 TaskSpec、变量角色、outcome summary、约束和审计记录。LLM 读取 source 与 target 的公开任务描述，生成一组结构化 Hypothesis，而不是直接猜测下一个实验点。

随后，Patch Compiler 把 Hypothesis 转成有限范围内的可执行参数，例如角色权重、kernel scale、探索系数和 source-prior strength。Calibration Gate 使用独立 calibration seeds 比较 transfer candidate 与两类 anchor：matched target-only LLM 和 strongest target-only BO。

通过 Gate 的路线进入正式 target replay；未通过的路线执行精确 target-only fallback。正式 held-out 结果只用于评估冻结后的策略，不再用于选择路线。最后，成功路线、失败路线、Gate 决策和 reasoning trace 都写入 memory，为后续任务提供可审计经验。

### 本页需要强调

- Calibration 和 held-out 是不重叠的两组 seeds。
- held-out 阶段不改 prompt、不改 threshold、不重新选路线。
- 当前代码实际只有 `Transfer` 和 `Freeze/Fallback` 两种输出；`Hold` 是下一版本计划。

### 过渡

下面具体解释 LLM 看到什么、生成什么，以及 held-out outcome 为什么不会泄漏。

## Slide 4｜Hypothesis 如何生成

### 建议讲稿

LLM 的输入分为三部分。Target 侧只提供公开的 schema、变量角色、优化方向和预算；Source 侧提供已经观测到的历史 outcomes、effect summary 和 uncertainty；Memory 提供经过审计的成功或失败 SkillCard、约束和历史 trace。

LLM 看不到 target 的 held-out outcome。这里的 held-out outcome 指预先留作最终评估的 target 真实结果。它不能进入 prompt、Gate 调参或候选打分，只能在策略冻结以后用于最终 replay。

LLM 输出的是多个候选 Hypothesis。当前主要有三类：角色映射，即哪些 source 与 target 字段承担相似科学角色；弱值先验，即 source outcome 是否可作为 bounded prior；探索调度，即在证据不确定时是否提高 GP-UCB 的探索程度。输出最终被解析为 `KernelSkillPatch`，包括 role multiplier、kernel scale、`gp_beta` 和 `source_prior_strength` 等字段。

### 可能追问

**LLM 是否直接决定候选实验？**  
不是。LLM 不输出 candidate ID，也不读取 target outcome。候选分数由确定性的 GP、prior calibration 和 acquisition 公式计算。

**为什么要生成多个 Hypothesis？**  
因为 source-target 之间可能共享不同层面的结构。多个 patch 可以由 target calibration 选择和加权，而不是把一次自然语言判断当作最终策略。

### 过渡

Hypothesis 生成后，还需要被编译成真正能改变优化过程的执行逻辑。

## Slide 5｜Hypothesis 如何执行

### 建议讲稿

执行层以 target-only acquisition 为 anchor。Target GP 根据当前已观测 target 数据得到均值和不确定性，并计算 UCB、EI 或二者的 portfolio。LLM patch 可以改变 kernel 中不同变量角色的权重，也可以引入根据真实 source outcomes 计算的邻域、加性或 interaction prior。

Source prior 不会直接覆盖 target 模型。系统先用已经 reveal 的 target observations 做 leave-one-out calibration，判断 prior 的方向和预测价值；证据较弱时，transfer mass 会连续收缩到零。最终 acquisition 可以概括为：

`CARE score = (1 - transfer mass) × target anchor + transfer mass × source-informed experts`

因此 LLM 决定“考虑哪些科学关系以及如何构造候选 patch”，Python 决定数值校准、边界约束、候选排序和 Gate 执行。Target budget、candidate pool、初始观测、seed split 和 comparator 始终保持不变。

### 可能追问

**这是不是仍然由硬编码规则控制？**  
数值安全边界确实由代码执行，这是为了可复现和防止 LLM 任意改变实验协议；但 kernel geometry、角色权重、探索计划和 source prior 的候选结构来自 LLM patch。当前需要进一步验证的是 LLM patch 相对于 similarity-only 或 no-validation patch 的独立贡献。

**每轮是否调用 LLM？**  
当前正式 replay 使用预先生成并冻结的 patch。每个 held-out seed 不重新调用 API，避免模型波动和测试集反馈进入策略。

### 过渡

下面用三条实际路线说明从 source evidence 到 held-out 结果的完整链条。

## Slide 6｜三条端到端代表性路线

### 建议讲稿

第一条是 ChemLex Acid-Amine 到 Buchwald-Hartwig，代表反应体系之间的迁移。Hypothesis 是反应物角色与 outcome interaction 可能共享结构，执行上使用角色映射和 bounded source prior。

第二条是 Matbench dielectric 到 experimental band gap，代表材料性质任务之间的迁移。这里迁移的重点不是某个具体高值样本，而是共享 composition/property 表示对 kernel geometry 和候选排序的帮助。

第三条是 FreeSolv 到 Lipophilicity，代表分子性质任务之间的迁移。两个任务共享分子描述符空间，系统尝试将 solvation signal 作为受约束的排序先验。

这三条路线都在 100 个独立 held-out seeds 上超过 matched LLM 和各自最强 target-only BO，并且减少达到基准质量所需的实验轮数。

### 本页需要强调

- 这三个例子覆盖三类不同的迁移机制，不是同一规则换数据集。
- 科学合理性仍需领域专家审查，数值增益不能替代机理判断。
- 下一步应把专家对 Hypothesis 的评价也作为数据沉淀。

### 过渡

下一页展开其中最完整的一条 ChemLex 到 BH 案例。

## Slide 7｜ChemLex → BH 完整案例

### 建议讲稿

Source 是真实 ChemLex Acid-Amine wetlab outcomes。系统从中提取 acid、amine 等角色及其 outcome interaction。Target 是 Buchwald-Hartwig 数据集；在 Hypothesis 生成阶段只公开 BH 的 schema、预算和变量角色，候选池预先固定，target outcome 不可见。

LLM 生成的核心 Hypothesis 有两点：共享反应角色可以迁移；source outcome 只能作为弱先验，不能替代 target observation。Compiler 把它转换为 role multiplier、kernel scale、`gp_beta` 和 `source_prior_strength`。

先在 50 个 calibration seeds 上同时与 matched target-only LLM 和 strongest BO 比较。路线通过后被冻结，再进入 100 个不重叠 held-out seeds。最终相对 matched LLM 的 final-best 提升为 `+9.20`，相对 strongest BO 为 `+8.10`，达到 matched LLM 最终质量平均节省 `2.99` 轮。

### 专家评审重点

- Acid-Amine 与 BH 的角色映射是否有化学依据。
- Source interaction 是否可能只是数据集偏差。
- 弱先验的方向是否符合反应机理。
- 是否需要加入反应指纹或更明确的化学 descriptor。

### 过渡

单个案例只能说明可行性，下面看预先固定的七条路线整体结果。

## Slide 8｜Held-out 基线结果

### 建议讲稿

这一页必须区分左右两种比较。

左图比较完整 CARE deployment 与每个任务最强的 target-only BO。完整 deployment 包含两种情况：通过 Gate 时使用 transfer router；未通过时使用 matched target-only LLM fallback。七条路线的 final-best delta 都为正，因此这里是 `7/7` 超过 strongest target-only BO。

右图比较同一个完整 deployment 与 matched target-only LLM，目的是隔离 source-outcome transfer 的净贡献。四条路线部署 transfer，且 95% CI 均在零以上；三条路线被 Gate 拒绝并精确回退，因此差值严格为零。这里的结论是 `4/7` 正向、`3/7` 打平、`0/7` 负向部署。

具体来说，ChemLex 到 BH 相对 matched LLM 为 `+9.20`；dielectric 到 gap 为 `+32.70`；gap 到 dielectric 为 `+18.06`；FreeSolv 到 Lipophilicity 为 `+2.98`。

### 可能追问

**为什么 fallback 路线在左图仍可能超过 BO？**  
因为 fallback 是 matched target-only LLM，它本身可能优于 GP-UCB 或 GP-EI。左图评价完整系统，右图才隔离 source transfer。

**能否说七条路线都迁移成功？**  
不能。只有四条实际部署了 source-outcome transfer；另外三条是安全回退。

### 过渡

把七条路线放到迁移矩阵中，可以进一步看出哪些关系已有证据，哪些仍是空白。

## Slide 9｜迁移矩阵与证据边界

### 建议讲稿

矩阵中的绿色单元格表示 frozen held-out 中确认的正向路线；红色表示 raw source route 出现负迁移并被 Gate 拒绝；灰色表示证据不稳定或最终回退。

目前绿色路线都属于同领域跨数据集：反应到反应、材料到材料、分子到分子。材料和分子任务表现较好，可能因为共享 descriptor、相近的目标结构以及重叠的参数空间。但目前这些只是机制 Hypothesis，还没有通过 descriptor ablation 或 representation control 被单独验证。

化学到材料、材料到分子等真正的 off-diagonal 跨领域单元格还没有正式结果。因此现在可以说框架在多个科学领域中表现出同领域迁移的泛化能力，但不能声称已经实现远距离跨领域迁移。

### 过渡

除了看提升百分比，我们还需要明确什么结果可以称为稳定，什么结果可以在论文中称为统计显著。

## Slide 10｜统计标准

### 建议讲稿

当前 replay 已经使用同一 held-out seed 上的 paired delta，同时报告 final-best、best-so-far AUC、paired bootstrap 95% CI、non-loss rate 和五个 seed block 的稳定性。这些指标可以回答：相对 matched baseline，改善方向是否稳定。

但论文中的“显著”需要更严格。Confirm run 计划对每条路线使用 paired permutation test，并对多路线比较使用 Benjamini-Hochberg FDR 控制。除了统计显著，还要在运行前定义 task-specific minimum detectable or meaningful effect，避免样本量很大时把很小的差异包装成重要贡献。

所以 2% 是否显著不能只看百分比；它取决于 seed-level 方差和置信区间。17% 也不能自动称为显著；而即便提升 100%，如果只有很少的独立重复，也可能非常不稳定。

### 三个名词

- `95% CI`：效果的配对不确定性区间；跨零说明方向仍不稳定。
- `BH-FDR q < 0.05`：在多条路线同时检验后控制 false discovery rate。
- `MDE`：在 confirm run 前定义、对实际实验有意义的最小效果。

### 过渡

统计检验回答“结果是否可信”，Calibration Gate 回答的是“这条路线是否允许部署”，两者不能混为一谈。

## Slide 11｜Calibration Gate

### 建议讲稿

当前 Gate 只有两种实际输出：`Transfer` 或 `Freeze`。路线需要同时满足六项条件才能部署 transfer：final mean 不为负、AUC mean 不为负、至少 60% seeds 不亏、至少 4/5 seed blocks 为正、风险缓冲后的 composite score 达标、paired CI 下界不为负。只要有一项未通过，就精确回退 matched target-only。

50 个 calibration seeds 用于选择路线，100 个不重叠 held-out seeds 只评估冻结结果。Gate 不是在 held-out 上看完结果再决定迁移。

这里要诚实说明参数来源。Paired comparison、uncertainty interval 和 strong BO anchor 属于通用评估原则；`0.60`、`0.80` 和 `0.25` 是当前 calibration replay 中固定的经验默认值，不是文献中的普适阈值。下一轮需要做 threshold grid、bootstrap sensitivity 和 leave-one-route-out calibration。

### 可能追问

**为什么没有 Hold？**  
当前代码为了保证解释清晰，只部署二选一策略。Hold 可以在下一版表示“方向合理但证据不足，需要追加 calibration”，目前尚未启用。

**Gate 会不会只是在挑正结果？**  
如果 calibration 与 held-out 独立、路线和 threshold 在 held-out 前冻结，Gate 是合法的模型选择与安全部署过程。真正需要避免的是反复查看 held-out 后修改规则。

### 过渡

下一页用 baseline 和 ablation 说明完整结果来自哪些模块，以及还有哪些贡献实验尚未补齐。

## Slide 12｜Baseline 与贡献验证

### 建议讲稿

目前最完整的两类 baseline 是 matched target-only LLM 和每个任务的 strongest target-only BO。相对 matched LLM，四条路线获得 source-outcome 增益，三条路线精确回退；相对 strongest BO，完整部署系统七条路线全部为正。

Gate 的必要性可以从 raw transfer 看出。若取消 Gate 并强制部署 source route，ESOL 到 Lipophilicity 和 Lipophilicity 到 FreeSolv 出现显著负迁移，Phonons 到 dielectric 的区间跨零。也就是说，安全回退并不是装饰性模块，它实际阻止了不可靠 source prior 进入正式部署。

Random-transfer 和 similarity/no-validation 已有代码或局部结果，但尚未在完全一致的七路线、同一预算和同一 seed split 上全部补跑，因此这里明确标为 `PARTIAL`。不能把它们写成已经完成的主论文结论。

### 两个数值例子

- ChemLex → BH：`+9.20` vs matched LLM，`+8.10` vs strongest BO。
- FreeSolv → Lipophilicity：`+2.98` vs matched LLM，`+4.99` vs strongest BO。

### 下一步必须补齐

- Random-transfer null。
- Similarity-only router。
- No hypothesis validation。
- Without Gate。
- Full CARE 2.0。

所有方法必须使用相同 target budget、initial observations、candidate pool 和 held-out seeds。

### 过渡

除了最终性能，实验成本更直接的指标是系统能少做多少轮 target 实验。

## Slide 13｜实验轮数与预算效率

### 建议讲稿

右侧第一组数字表示：达到 matched target-only LLM 最终质量时，CARE 平均少用多少轮 target acquisition。括号内是 paired 95% CI。

- ChemLex → BH：节省 `2.99` 轮，95% CI 为 `[2.05, 3.93]`。
- Dielectric → Gap：节省 `6.00` 轮，95% CI 为 `[5.14, 6.86]`。
- Gap → Dielectric：节省 `5.03` 轮，95% CI 为 `[4.27, 5.79]`。
- FreeSolv → Lipophilicity：节省 `2.23` 轮，95% CI 为 `[1.37, 3.09]`。

另外，提前进入全局 top-10 候选集合所节省的平均轮数分别是 `3.62`、`9.57`、`8.46` 和 `4.80`。

Round 0 表示初始 observations。若某条路径在预算内没有达到比较阈值，则按 `budget + 1` 做右删失。Fallback 路线与 matched LLM 的逐 seed 轨迹一致，因此 round saving 为零。

当前可以支持的结论是迁移减少了达到同等质量所需的实验轮数。不能仅凭“早期增益高于最终增益”就声称边际收益递减；要证明饱和，需要系统改变 source-history size，并比较线性与饱和模型。

### 过渡

最后两页说明在形成论文结论之前，还要冻结和补齐哪些实验。

## Slide 14｜论文级 Confirm Run

### 建议讲稿

下一轮的重点不是继续手动寻找更好看的 pair，而是冻结统一的 confirm protocol。

第一步冻结 route list、calibration/held-out split、threshold grid、primary metric 和各任务 MDE。第二步在完全相同的预算、初始点、候选池和 seeds 下跑 full method 与所有 baseline/ablation。第三步对每个模块做配对贡献归因，并同时报告效果与安全性。

论文主表至少应包含 final-best effect、AUC effect、paired 95% CI、BH-FDR q-value、task-specific practical effect、negative-transfer rate、Gate acceptance rate、exact-fallback rate 和 rounds saved。

在 confirm run 开始后，不再根据 held-out 结果修改 prompt、threshold 或 route。若需要调整，必须形成新的开发版本并重新生成独立 confirm split。

### 需要形成的论文证据

- Hypothesis generation 是否优于 similarity-only。
- Source-outcome prior 是否优于只用 schema。
- Gate 是否显著降低负迁移。
- LLM patch 是否在多个 pair 上提供一致增益。
- 知识库中的成功与失败 SkillCard 是否改善后续任务。

### 过渡

基于目前已经完成的实验，最后给出可以成立的结论和不能越过的边界。

## Slide 15｜当前结论

### 建议讲稿

目前最稳妥的结论是：CARE 2.0 已在反应、材料和分子三个领域中，展示同领域跨数据集的可审计正向迁移。七条预先固定路线中，四条部署 source-outcome transfer 并稳定超过 matched target-only LLM；三条因证据不足精确回退，正式 held-out 中没有部署负迁移。

CARE 2.0 的核心贡献不是保证任意 source-target pair 都提高，而是把 source experience 变成结构化 Hypothesis、受约束的可执行 patch、独立 calibration 决策和可审计 fallback。成功路线、失败路线和 reasoning trace 都可以继续沉淀为知识库资产。

当前还不能声称远距离跨领域迁移已经得到验证，也不能把所有 calibration threshold 描述为文献标准。下一步需要完成统一七路线的 baseline/ablation、正式多重比较统计、threshold sensitivity、更多 off-diagonal pair，以及领域专家对代表性 Hypothesis 的科学审查。

### 建议收尾

一句话概括：CARE 2.0 目前证明的不是“LLM 总能迁移”，而是“LLM 可以提出迁移策略，系统能够用独立证据判断何时采用、何时拒绝，并在多个真实科学数据集上获得可复现的收益”。

## 会后可能被问到的问题

### 1. LLM 的独立贡献是否已经完全证明？

还没有。当前已证明完整 LLM transfer router 在四条路线产生正向 source-outcome 增益，但 similarity-only、random-transfer 和 no-validation 尚未在统一七路线协议上全部跑完。正式论文需要用这些 ablation 隔离 LLM Hypothesis 的贡献。

### 2. 结果是否存在 held-out 泄漏？

当前协议使用 50 个 calibration seeds 选择并冻结路线，再使用 100 个不重叠 held-out seeds 评估。LLM 不读取 held-out target outcome，held-out 阶段也不重新调用模型或修改规则。后续仍应把 route list、prompt hash、threshold 和 split manifest 一并冻结发布。

### 3. 为什么可以在 calibration 后选择是否迁移？

真实部署本来就需要 validation 或 calibration 来决定采用哪个模型。只要 calibration 与最终 held-out 严格独立，这属于模型选择而不是测试集调参。问题不在于使用 calibration，而在于是否反复查看 held-out 后继续改规则。

### 4. 这是不是只证明了 Gate，而没有证明迁移？

不是。四条通过 Gate 的路线相对 matched target-only LLM 的 final-best 和 AUC 都为正，说明 source outcomes 提供了净增益；三条 fallback 说明 Gate 提供安全性。迁移增益和安全回退是两个需要分别报告的贡献。

### 5. 能否称为跨领域迁移？

目前不建议。更准确的表述是“在反应、材料、分子多个领域中验证了同领域跨数据集迁移”。真正跨领域的 off-diagonal pair 仍属于下一阶段探索。

## 主要结果文件

- `experiments/care_replay/results/2026-07-28-evaluation-completion/baseline_comparison.csv`
- `experiments/care_replay/results/2026-07-28-evaluation-completion/iteration_efficiency.csv`
- `experiments/care_replay/results/2026-07-28-evaluation-completion/baseline_protocol_coverage.csv`
- `experiments/care_replay/results/2026-07-28-evaluation-completion/LLM_RULE_FORMULAS.md`
- `experiments/care_replay/results/2026-07-24-source-outcome-transfer/`

