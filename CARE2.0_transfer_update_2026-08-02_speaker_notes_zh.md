# CARE 2.0 Transfer Update 中文 Speaker Notes

配套文件：`CARE2.0_transfer_update_2026-08-02.pptx`  
建议汇报时间：30-35 分钟，讨论时间另计。  
适用场景：项目组内部汇报、专家评审、论文方法讨论。  

## 汇报总口径

这次汇报最重要的一句话是：**CARE 2.0 不是假设所有历史经验都可以迁移，而是把“迁移什么、什么时候迁移、什么时候拒绝”变成一套可执行、可验证、可回退的协议。**

当前已经得到支持的是：在反应、材料和分子三个领域中，存在同领域、跨数据集的正向迁移；七条冻结路线中四条部署了 source-outcome transfer，三条因为证据不足而精确回退，正式 held-out 评估中没有部署已知负迁移。当前尚未证明化学到材料、材料到分子等远距离跨领域迁移。

汇报时要始终区分下面三件事：

1. **完整系统是否超过强 BO**：完整系统包含 transfer 和 fallback 两种部署结果。
2. **source transfer 是否有净增益**：必须与 matched target-only LLM 比较。
3. **Gate 是否提供安全性**：看 raw transfer 中的负迁移是否被拦截，以及 fallback 是否逐 seed 精确复现 target-only。

不要把“7/7 超过 strongest BO”说成“7/7 都迁移成功”。真正部署 source transfer 的是 4/7，另外 3/7 是安全回退。

## 汇报前术语速查

### Source task 与 target task

Source task 是已经积累了实验结果的历史任务。Target task 是当前要优化的新任务。一次 transfer route 就是一条预先声明的 `source -> target` 路线，例如 `FreeSolv -> Lipophilicity`。

### Replay harness

本项目目前使用的是 finite-pool sequential replay，而不是每一步都去实验室重新做湿实验。流程是：从真实公开数据集中冻结候选池和真实结果；算法一开始只能看到少量 initial observations；每一轮选择一个尚未观察的候选；replay harness 再揭示该候选在真实数据集中的结果；算法据此更新模型并选择下一轮，直到用完固定预算。

因此，replay 保留了“实验昂贵、结果只能在选择后揭示”的顺序决策结构，同时允许我们在完全相同的候选池、预算和随机种子下比较不同方法。这里的 seed 表示一条独立的初始化和决策轨迹，不等于重新采集了一套新的湿实验数据。

### Target-only

Target-only 方法只使用 target 已经揭示的 observation，不读取 source outcome。它回答的是：如果完全不迁移，单靠目标任务自身数据能做到什么程度。

### Matched target-only LLM

Matched target-only LLM 与 CARE 使用同样的 target 数据集、预算、initial observations、candidate pool、seed 和 LLM 执行约束，但不提供 source outcomes。它是隔离 source 信息净贡献的主要对照。

### Strongest target-only BO

指在冻结的 target-only Bayesian optimization 候选中，由 calibration 数据选择出的最强方法，例如 GP-UCB、GP-EI 或 acquisition portfolio。它不是在 held-out 结果上事后挑选的 oracle。

### Hypothesis 与 patch

Hypothesis 是结构化的迁移假设，例如“两个反应任务中的底物角色可以对齐”或“溶剂化信号可作为脂溶性候选排序的弱先验”。Patch 是 Hypothesis 的可执行版本，包括 kernel scale、role multiplier、`gp_beta`、source-prior strength 和 transfer mass 等受约束参数。

### Calibration、held-out 与 Gate

Calibration seeds 用于比较候选路线并冻结部署决定。Held-out seeds 只评估已经冻结的策略，不再改 prompt、路线、参数或 threshold。Gate 根据 calibration 证据输出 Transfer 或 Freeze/Fallback。当前版本没有实际启用 Hold。

### Exact fallback

如果迁移路线未通过 Gate，系统部署 matched target-only LLM。所谓 exact，是指 fallback 在同一 seed 下复现 target-only 的候选选择和结果，而不是换成另一条近似规则。

### Final best、AUC 与 rounds saved

- `Final best`：预算结束时找到的最好结果。
- `Best-so-far AUC`：整个实验过程的累计表现，越早发现好候选，AUC 越高。
- `Rounds saved`：CARE 达到 matched baseline 最终质量时，平均少使用的 target 实验轮数。

---

## Slide 1｜CARE 2.0：可信赖的跨任务科学知识迁移框架

**建议时间：1 分钟**

### 建议讲稿

今天汇报 CARE 2.0 的最新进展。我们这一阶段没有把重点放在再做一条针对某个数据集的最优规则，而是在回答一个更一般的问题：一个 AI Scientist 如何把历史实验经验转成可以在新任务中执行的策略，同时能够判断这条经验不该迁移。

所以标题里用了“可信赖”三个字。这里的可信赖不是说系统永远正确，而是指每一次迁移都有明确的 source evidence、结构化 Hypothesis、受约束的执行 patch、独立的 calibration 决策和 held-out 验证。如果证据不足，系统必须回到 target-only，而不是为了追求正结果强行迁移。

整套方法可以概括为三个动作：第一，生成可解释的迁移 Hypothesis；第二，把 Hypothesis 编译成真正能影响实验选择的 acquisition patch；第三，用独立证据决定部署还是回退。

### 本页要强调

- 贡献对象是一套迁移与拒绝协议，不是单条人工规则。
- 当前 PPT 汇报的是 8 月 2 日冻结版本对应的结果。
- 当前“跨任务”主要是同领域中的跨数据集，不是任意学科之间的迁移。

### 不要这样讲

- 不要说“我们已经实现通用跨领域 AI Scientist”。
- 不要说“LLM 可以自动发现所有可迁移规律”。
- 不要把 replay 结果说成新做了数百轮湿实验。

### 过渡

在介绍系统之前，先明确我们对“具备迁移能力的 AI Scientist”到底如何定义。

---

## Slide 2｜核心研究问题

**建议时间：2-3 分钟**

### 建议讲稿

我们把研究问题定义为一个固定预算下的顺序决策问题。给定一个 source task 的历史实验和 outcome、一个 target task 的公开 schema、固定的候选池和 target 实验预算，系统需要回答四个问题：迁移什么、迁移多少、何时拒绝，以及下一轮在 target 中选择哪个候选。

这里的“通用”不是指 LLM 能谈论很多学科，而是指同一套接口和决策协议可以处理不同类型的科学任务。反应任务的变量可能是底物、催化剂和溶剂；材料任务可能是组成与性质描述符；分子任务可能是结构描述符。只要它们能够被整理成统一的 TaskSpec、变量角色、候选池和目标函数，后面的 Hypothesis、patch、Gate 和 audit 流程就可以复用。

我们的核心贡献分成两层。第一层是 LLM 生成可执行 Hypothesis，并将它落到 acquisition geometry 上，而不是只输出自然语言建议。第二层是 Calibration Gate 使用独立开发 seeds 决定是否部署。也就是说，LLM 可以提出迁移方案，但不能凭一句解释直接控制正式实验。

证据边界需要在这一页讲清楚。目前实验覆盖反应、材料和分子三个领域，证明了框架能在多个领域中运行，也观察到了各领域内部的跨数据集正向迁移。但是化学到材料、材料到分子等 off-diagonal 路线还没有形成正式证据，所以现在不能宣称远距离跨领域迁移已经建立。

### 补充解释：实验是怎样运行的

每个 target 数据集先被冻结为有限候选池。算法只看到 initial observations，之后每轮提出一个候选，系统再从真实数据中揭示它的 outcome。所有方法使用相同 seed、初始点、预算和候选池，因此比较的是算法利用有限实验预算的效率，而不是谁看到了更多数据。

### 可能追问

**为什么不把 random search 作为主 baseline？**  
Random search 可以保留作最低基准，但它太弱，不能支撑论文的核心结论。主 baseline 必须是 strongest target-only BO 和 matched target-only LLM。

**这里的泛化到底是什么？**  
目前的泛化是同一套协议跨多个预先固定的 source-target pair 工作，而不是在同一个 pair 上换几个 seed。远距离跨领域泛化还没有验证。

### 过渡

下一页从左到右看一次完整的迁移是如何产生、执行和被审计的。

---

## Slide 3｜从历史实验到可验证迁移

**建议时间：3 分钟**

### 建议讲稿

[指向左侧] 流程从 Historical Experiments 开始。历史实验不是直接作为一张表丢给 LLM，而是先整理成 TaskSpec、观测结果和 audit。TaskSpec 描述任务目标、变量类型、字段角色、优化方向、约束和预算。

下一步是 Feature and Evidence Extraction。这里提取的不只是字段名，还包括变量角色、descriptor、source outcome 的效应摘要、不确定性和失败记录。这一步的目的，是把某个数据集中的具体列转换成可比较的科学角色。

[指向中间] LLM 在这些证据上生成多个有边界的 Hypothesis。系统随后估计 source-target 的 schema 是否兼容、source 证据是否稳定，并把自然语言假设编译为受约束 patch。LLM 不是直接给出“做第 37 个实验”，而是提出应该改变哪些角色权重、先验或探索参数。

[指向三条分支] Calibration Gate 有三种概念状态：Transfer、Hold 和 Freeze/Reject。当前代码真正实现的是 Transfer 与 Freeze/Fallback。通过 Gate 的路线进入 target replay；未通过的路线精确回退。Hold 目前只是下一版本的设计，用于表示方向合理但证据不够、需要追加 calibration。

[指向下方] 正式 target replay 使用固定预算。统计模块按相同 seeds 计算配对增益、置信区间和稳定性。最后，Hypothesis、patch、Gate decision、held-out outcome、成功案例和失败案例都会写入 memory。

这张图对应三项贡献：LLM 回答“迁移什么”，Calibration Gate 回答“什么时候允许迁移”，held-out 与 memory 回答“结果怎样被复核并沉淀”。

### 需要特别说明

- Calibration 与 held-out 使用不重叠 seeds。
- Held-out 期间不重新选择路线，也不根据结果修改 threshold。
- Memory 中保留 rejected case，不只保留成功路线，否则知识库会形成幸存者偏差。

### 可能追问

**为什么需要 Gate，LLM 自己给置信度不行吗？**  
LLM 的语言置信度没有经过当前任务的数值校准。Gate 使用实际 replay 证据比较 transfer 与 matched anchors，承担的是部署层的统计决策，不能由自然语言自信程度替代。

### 过渡

下面分别展开最关键的两个环节：Hypothesis 是如何生成的，以及它如何真正影响 target acquisition。

---

## Slide 4｜Hypothesis 生成

**建议时间：3 分钟**

### 建议讲稿

这一页最重要的是说明 LLM 看到了什么、没有看到什么，以及最终输出是什么。

[指向左侧] Target 侧只给公开元数据，包括 schema、变量角色、优化方向和预算。Source 侧给已经合法观测到的历史 outcomes、effect summaries 和 uncertainty。Memory 提供经过审计的 SkillCard、失败路线、约束和历史 trace。

LLM 看不到 target held-out outcome。这里的 held-out outcome 是预先保留给最终评估的 target 结果。在 replay 中，某个 target 候选只有被算法选择后，其 outcome 才会揭示给在线模型。未选择候选的结果不能进入 prompt、patch 生成或 calibration 以外的路线调参。

[指向右侧] 当前 Hypothesis 主要分为三类。第一类是 role mapping，例如 source 与 target 中哪些字段都承担底物、溶剂、组成或分子描述符的角色。第二类是 value prior，即 source outcome 是否能作为一个有上限的弱先验。第三类是 exploration hypothesis，即在不确定性较高时是否提高 GP-UCB 的探索程度。

这些候选还要经过 schema compatibility、source evidence stability 和 calibration improvement 三层筛选。最终输出不是一段散文，而是 `KernelSkillPatch`。它包含 role weight、kernel scale、`gp_beta` 和 source-prior strength 等结构化字段，原始输出与解析结果都会保存。

### LLM 在系统中的准确角色

LLM 更接近“策略假设生成器”和“科学角色映射器”，不是 outcome predictor，也不是最终候选打分器。数值 prior 来自真实 source outcomes，target posterior 来自 GP，候选排序和边界约束由确定性代码完成。

### 可能追问

**LLM 是否每一轮都调用？**  
当前冻结实验不是每个 held-out seed、每一轮重新请求 API。LLM 先生成并冻结候选 patch，随后 replay 重复执行同一结构化策略。这样可以避免 API 波动，也防止 held-out 反馈进入 prompt。

**这还算 LLM 方法吗？**  
算，但需要准确表述。LLM 的贡献是生成可复用的策略结构，数值执行由可复现代码完成。论文中的 ablation 仍需证明 LLM Hypothesis 优于 similarity-only 或 random patch。

### 过渡

有了 Hypothesis 以后，下一步是把它编译成可以改变 Bayesian optimization 行为的 patch。

---

## Slide 5｜Hypothesis 执行

**建议时间：3 分钟**

### 建议讲稿

Hypothesis 首先被保存为 `HypothesisEntry`，其中包含 claim、mechanism、证据来源和适用边界。Patch Compiler 再把它翻译成有上下限的数值参数。这样做的目的，是让同一个 Hypothesis 可以被执行、比较和复现，而不是停留在自然语言层面。

Target-only GP 始终是执行锚点。它根据当前已经揭示的 target observations 计算 posterior mean 和 uncertainty，再通过 UCB、EI 或 portfolio 给候选打分。迁移 patch 可以改变四类内容：变量角色对应的 kernel scale、GP 的探索计划、source prior 的强度，以及 target anchor 与 transfer expert 的混合权重。

可以用一个简化表达帮助听众理解：

`CARE acquisition = (1 - M) × target-only acquisition + M × source-informed acquisition`

其中 `M` 是 transfer mass。证据强时，`M` 可以大于零；证据弱时，它会被限制或直接降为零。这个公式只是解释结构，正式代码还包含不同 prior expert、role weight 和边界约束。

本页右下角列出的内容始终冻结：target 数据集、优化方向、candidate pool、initial observations、总预算、held-out seeds、baseline 身份和评价指标。迁移只能改变允许的 patch 参数，不能偷偷增加实验次数，也不能更换更容易的候选池。

完成 calibration 后，Gate 冻结选中的 route identity。Held-out 阶段只能执行冻结路线，不能重新在多个候选中挑最好结果。每轮 candidate、已揭示 outcome、acquisition diagnostics、Gate decision 和最终指标都会进入 audit。

### 需要诚实说明的归因问题

有些 patch 会影响 initial design，有些会持续影响后续 acquisition。两者都属于知识迁移，但科学解释不同。汇报完整系统结果时，可以说 route 有效；若要声称“持续 source prior 在每轮都产生独立贡献”，必须增加 warm-start-only control。现有报告已经开始做这类归因，但统一七路线 ablation 仍需补齐。

### 可能追问

**为什么还要硬编码边界？**  
因为 target 实验协议、预算和数值安全约束需要可复现。LLM 提出策略，代码负责把策略限制在合法范围内。这不是削弱 LLM，而是把模型输出变成可以审计的实验干预。

### 过渡

下面先不看抽象模块，直接看反应、材料和分子三类真实路线的结果。

---

## Slide 6｜三条代表性链条

**建议时间：3 分钟**

### 建议讲稿

这页选了三条代表性路线，分别对应反应、材料和分子任务。它们不是把同一条规则机械复制到三个数据集，而是使用不同的 source evidence 和迁移机制。

第一条是 `ChemLex Acid-Amine -> Buchwald-Hartwig`。ChemLex 是真实 acid-amine wetlab outcome 数据，target 是 Buchwald-Hartwig 反应 HTE。Hypothesis 是两类反应任务中的反应物角色和 outcome interaction 可以提供迁移信号。完整部署相对 matched LLM 的 final-best 提升 `9.20`，相对 strongest BO 提升 `8.10`，达到 matched LLM 最终质量平均节省 `2.99` 轮。

第二条是 `Matbench dielectric -> experimental band gap`。这不是把介电常数的高值直接当作带隙高值，而是假设两个材料任务共享 composition/property representation，source 可以改变 kernel geometry 和候选排序。相对 matched LLM 提升 `32.70`，相对 strongest BO 提升 `35.46`，节省 `6.00` 轮。

第三条是 `FreeSolv -> Lipophilicity`。FreeSolv 的目标是水合自由能，Lipophilicity 反映分子在不同相中的分配倾向。两者并不等价，但共享分子结构描述符，而且溶剂化信号可能帮助排序。系统只允许 bounded outcome prior，不允许 source outcome 覆盖 target observation。结果相对 matched LLM 提升 `2.98`，相对 strongest BO 提升 `4.99`，节省 `2.23` 轮。

### 三个数据集族的简要解释

- ChemLex/BH：真实反应组合与条件优化数据。
- Matbench dielectric、expt_gap、phonons：材料组成到性质的有限池任务。
- ESOL、FreeSolv、Lipophilicity：共享分子结构表示、但目标性质不同的 MoleculeNet 类任务。

### 本页结论

我们已经证明同一套框架能够承载三类不同的科学任务。数字说明正向迁移存在，但不能代替领域机理审查。特别是材料和分子路线中“共享 representation 导致可迁移”的解释，目前仍是待验证 Hypothesis，不是因果结论。

### 过渡

下一页把三条路线拆开，说明 source evidence 到底被翻译成了什么 patch。

---

## Slide 7｜逐路线解释 source evidence、Hypothesis 与 patch

**建议时间：2-3 分钟**

### 建议讲稿

这一页与上一页的区别是：上一页强调结果，这一页强调每条路线“具体迁移了什么”。

对 ChemLex 到 BH，source evidence 是 acid、amine 等角色和 outcome interaction。LLM 的 Hypothesis 是共享反应角色可以迁移，但 source outcome 只能作为弱 prior。执行层因此使用 role multipliers、kernel scale 和 bounded prior strength。

对 dielectric 到 gap，source evidence 是材料 composition/property observations。Hypothesis 不是两个物性相同，而是同一个组成描述空间可能包含可复用的局部几何。执行时主要改变 role-weighted kernel，并加入受约束 prior。

对 FreeSolv 到 Lipophilicity，source evidence 是共享 molecular descriptor 和 solvation outcome。Hypothesis 是溶剂化相关信号可以帮助候选排序。执行时使用 bounded outcome prior，并始终保留 target acquisition 作为 anchor。

三条路线都遵循同一冻结协议：target held-out outcome 不进入 Hypothesis 生成；calibration 只使用独立开发 seeds；路线冻结后才运行 100 个 held-out seeds。结果不是从多个 held-out patch 中选出来的最好一个。

### 建议现场说法

“我们迁移的不是一个答案，也不是 source 中的最优样本本身，而是对 target 搜索空间的结构化偏置。这个偏置必须能被编译、限制和独立验证。”

### 可能追问

**这些 Hypothesis 是不是人工写出来的？**  
当前系统中，LLM 基于 source/target card 生成结构化候选，代码负责解析和约束。部分候选集合、参数范围和字段 vocabulary 是工程上预先定义的。正式论文需要用保存的 model record、prompt hash 和 similarity-only ablation 证明 LLM 生成部分的独立价值。

### 过渡

下面以 ChemLex 到 BH 为例，把证据链完整走一遍。

---

## Slide 8｜ChemLex → BH 完整案例

**建议时间：3 分钟**

### 建议讲稿

[指向左栏] Source 是 ChemLex Acid-Amine 的真实 wetlab outcomes。系统提取 acid、amine 等角色以及它们与 outcome 的 interaction。Target 是 BH 数据集。在 Hypothesis 阶段，系统知道 BH 的字段角色、优化方向、预算和候选池结构，但不知道 held-out outcome。

[指向中栏] LLM 给出的核心 Hypothesis 有两点。第一，共享反应角色可能具有可迁移结构；第二，source outcome 只能作为弱 prior，不能替代 target observation。Compiler 把它翻译成 role multiplier、kernel scale、`gp_beta` 和 `source_prior_strength`。

[指向右栏] 先在 50 个 calibration seeds 上与 matched target-only LLM 和 strongest BO 比较。只有同时满足 Gate 条件，route 才被冻结为 transfer。随后使用 100 个不重叠的 held-out seeds 评估。最终 final-best 相对 matched LLM 为 `+9.20`，相对 strongest BO 为 `+8.10`；达到 matched LLM 最终质量平均节省 `2.99` 轮。

这里需要提醒听众：本页证明的是完整 frozen route 有效，不应仅凭这张图断言所有增益都来自每轮连续 prior adjustment。反应路线中的 initial design、role kernel 和持续 prior 分别贡献多少，需要通过 warm-start-only、without-prior 和 without-kernel ablation 进一步拆分。

### 领域专家需要审查的问题

- Acid-Amine 与 BH 的角色对齐是否有反应机理依据？
- Source interaction 是否可能只是数据集采样偏差？
- 哪些角色应该允许迁移，哪些条件应保持 target-specific？
- 是否需要加入反应指纹、官能团或催化剂 descriptor，替代纯 schema 对齐？

### 过渡

一个案例只能说明可行性，下一页看七条预先固定路线的整体 held-out 结果。

---

## Slide 9｜四条路线同时超过 matched LLM 与 strongest BO

**建议时间：3 分钟**

### 建议讲稿

这页有左右两个 forest plot，必须分别解释。

[指向左图] 左图比较的是 CARE 完整部署与每个 target 的 strongest target-only BO。绿色表示部署 transfer 的路线，灰色表示 Gate 拒绝后部署 matched target-only LLM 的 fallback。七条完整部署路线相对 strongest BO 都为正，但这里包含 fallback 的贡献，所以不能解释成七条 source transfer 都成功。

[指向右图] 右图比较 source transfer deployment 与 matched target-only LLM，目的是隔离 source outcome 的净贡献。四条绿色路线的配对 final-best 置信区间在零以上；三条灰色路线因为 exact fallback，与 matched LLM 的差值严格为零。

七条路线的最终部署情况是：

- Transfer：ChemLex -> BH、Dielectric -> Gap、Gap -> Dielectric、FreeSolv -> Lipophilicity。
- Exact fallback：Phonons -> Dielectric、ESOL -> Lipophilicity、Lipophilicity -> FreeSolv。

实验协议是 50 个 calibration seeds 冻结路线，100 个不重叠 held-out seeds 做最终评估。结果可以表述为：`4/7` 获得 source transfer 净增益，`3/7` 精确打平，正式部署中 `0/7` 负向。

### 一定要说明的统计边界

当前主图使用配对 held-out delta 与 95% CI。论文级 confirm run 还应补 paired permutation test、BH-FDR 和预先定义的 MDE。因此现在可以说“区间稳定高于零”或“在当前冻结 replay 中稳定为正”，但不要把所有结果直接称为已经通过最终多重比较校正的统计显著。

### 可能追问

**为什么 fallback 路线在左图仍然超过 BO？**  
因为 fallback 是 matched target-only LLM，而 matched LLM 本身可能优于 GP-UCB 或 GP-EI。左图评价完整部署性能；右图才评价 source transfer 的额外贡献。

**Gate 是不是看完 held-out 才决定的？**  
不是。Route 在 calibration 上决定并冻结，held-out 只执行冻结策略。

### 过渡

把这些结果放进 source-target 矩阵后，可以更直观看到哪些迁移关系已有证据，哪些仍是空白。

---

## Slide 10｜迁移矩阵与证据边界

**建议时间：3 分钟**

### 建议讲稿

矩阵纵轴是 source task，横轴是 target task。绿色表示 raw route 通过 Gate，并在 held-out 上部署正向迁移；红色表示 raw source route 出现负向信号，被 Gate 拒绝；灰色表示证据不稳定或没有实验覆盖。

当前四个正例全部在领域内部：一个 Reaction -> Reaction，两个 Materials -> Materials，一个 Molecular -> Molecular。三个 fallback 包括一个材料路线和两个分子路线。ESOL -> Lipophilicity 与 Lipophilicity -> FreeSolv 的 raw transfer 是负向，Phonons -> Dielectric 的 raw signal 略正但区间跨零，因此都没有部署。

矩阵中的空白不能解释成迁移失败，它只表示还没有按冻结协议运行。Reaction -> Materials、Reaction -> Molecular、Materials -> Molecular 等 off-diagonal 路线仍未验证。

右下角给出了一个机制 Hypothesis：材料和分子任务可能因为共享 representation、相近 objective 或参数空间重叠而更容易迁移。但这只是解释候选。要验证它，需要增加 representation ablation、descriptor overlap 指标和不同 task-distance 分层实验。

### 这页应得出的结论

可以说：“CARE 2.0 已经在三个科学领域中得到同领域跨数据集证据。”  
不能说：“CARE 2.0 已经证明跨学科知识可以普遍迁移。”

### 可能追问

**为什么叫通用框架，但还没有跨领域结果？**  
“通用”首先指统一接口、统一验证协议和跨任务复用能力。远距离跨领域是这个框架要检验的下一层问题，而不是由框架名称自动成立的结论。

### 过渡

有了正负路线之后，下一步需要明确我们用什么指标判断提升、稳定性和实际意义。

---

## Slide 11｜评价指标

**建议时间：3 分钟**

### 建议讲稿

我们不只看最终提升百分比，而是从性能、过程、稳定性和实际成本四个方面评价。

第一，`paired final-best delta` 衡量预算结束时最好结果的差值。第二，`best-so-far AUC` 衡量整个搜索过程，能够区分“最后一轮偶然找到好点”和“前几轮就持续找到好点”。第三，`non-loss rate` 和五个 seed block 检查正向结果是否只由少数 seeds 拉高。第四，`rounds saved` 把算法增益转换成少做多少次 target 实验。

为什么使用 paired comparison？因为 CARE 与 baseline 在同一个 seed 下共享 initial observations、candidate pool 和预算。先做逐 seed 差值，再对差值做统计，可以消除很多由初始化难度带来的方差。

当前报告使用 paired bootstrap 95% CI。置信区间跨过零，表示方向仍不稳定。对于论文 confirm run，我们计划使用 paired permutation test，并对多条路线做 Benjamini-Hochberg FDR 校正。

还需要区分统计显著和实际有意义。样本量很大时，极小差异也可能得到很小的 p 值。因此每个任务应在运行前定义 task-specific MDE，也就是对真实实验有意义的最低效果。最终报告应同时给 effect、95% CI、q-value 和 MDE。

### 如何解释 2%、17% 和 100% 提升

- 2% 可能很稳定，也可能完全被方差淹没，不能只看百分比。
- 17% 如果在独立 seeds 上区间稳定，通常比 2% 更有实际意义，但仍需检验。
- 100% 如果 baseline 很小、重复次数很少或由极端值驱动，也可能不稳定。

### 当前完成与下一步的区别

当前已经完成 paired delta、AUC、bootstrap CI、non-loss 和 block stability。Permutation test、BH-FDR 和 task-specific MDE 应作为冻结 confirm protocol 的正式组成部分，不要在汇报中误说成所有路线已经完成最终论文统计。

### 过渡

评价指标告诉我们结果是否稳定，Calibration Gate 则决定一条迁移路线是否允许部署。

---

## Slide 12｜什么时候该迁移：Calibration Gate

**建议时间：3-4 分钟**

### 建议讲稿

Gate 的作用可以概括为一句话：证据够强才迁移，否则回到 target-only。它不是用来把负结果隐藏掉，而是一个预先定义的部署安全层。

一条候选 route 要同时与两个 anchor 比较：matched target-only LLM 和 strongest target-only BO。当前规则要求六项全部通过：final mean 不亏、AUC mean 不亏、final non-loss rate 至少 0.60、五个 blocks 中至少四个为正、`composite mean - standard error` 至少 0.25，以及 composite 95% CI 下界不低于零。

只要任一条件失败，系统就 Freeze 并部署 exact target-only fallback。当前版本没有“差不多就迁移”的灰色策略，因此正式 held-out 中不会把一个 calibration 证据不足的 raw route 强行上线。

参数来源也要坦诚说明。Paired comparison、confidence interval、强 baseline anchor 是通用统计原则；`0.60`、`0.80` 和 `0.25` 是在开发阶段固定的 calibrated defaults，不是某篇文献规定的普适阈值。它们必须在 held-out 前冻结，并接受 sensitivity analysis。

下一轮应对 final/AUC 权重、block 数量、non-loss threshold 和 risk-adjusted threshold 做 grid 或 bootstrap sweep。更严格的做法是 leave-one-route-out calibration：用其他路线确定 threshold，再看被留出的路线，以减少 pair-specific 调参嫌疑。

### Gate 与统计检验的区别

Gate 是部署规则，回答“要不要使用 source route”。统计检验是结果报告，回答“冻结后的效果有多确定”。二者相关，但不能把 Gate 通过直接等同于论文统计显著。

### 可能追问

**这是不是在挑正结果？**  
模型部署本来就需要 validation。只要 calibration 与 held-out 严格分离、threshold 与 route list 在 held-out 前冻结，这属于合法的模型选择。真正不允许的是看完 held-out 后不断改 Gate，再把同一 held-out 当成最终测试。

**为什么当前没有 Hold？**  
为了让当前证据链清楚，代码只实现 Transfer 与 Freeze。Hold 可以在下一版表示“需要额外 calibration 数据”，但不能把它写成已经完成的功能。

### 过渡

有了 Gate 以后，还需要通过 baseline 和 ablation 证明增益究竟来自哪个模块。

---

## Slide 13｜Baseline 与贡献验证

**建议时间：4 分钟**

### 建议讲稿

这一页是回答“我们的提升到底来自哪里”。目前证据最完整的是两类强 baseline。

第一类是 matched target-only LLM。它与 CARE 使用相同 target 预算和 seeds，但拿不到 source outcome。四条路线相对它提升，三条 exact fallback，因此 source transfer 的净结果是 `4/7` 正向、`3/7` 打平。

第二类是 strongest target-only BO，包括 GP-UCB、GP-EI 或 target acquisition portfolio。完整 CARE deployment 在七条路线中都超过对应的 strongest BO。但再次强调，其中三条依靠的是 target-only LLM fallback，不是 source transfer 本身。

第三类是 raw transfer/no Gate。它强制部署 source route，用于衡量 Gate 的安全贡献。结果出现两条明确负迁移和一条不确定路线，说明 Gate 不是装饰模块；它确实阻止了不可靠 prior 进入正式部署。

Random-transfer 用随机 source 或随机 patch，目的是排除“只要增加一种 prior 形式就会变好”。Similarity-only 只使用 schema 或 descriptor overlap，目的是判断浅层相似度能否替代 LLM Hypothesis。当前这两类只完成了局部路线或历史控制，尚未在同一七路线、同一 split、同一预算下全部完成，所以状态必须写 `PARTIAL`。

Full CARE 2.0 包括 Hypothesis、patch 和 Gate。当前可以证明完整系统有可审计部署和 exact fallback，但还不能说 LLM Hypothesis 的独立贡献已经被完整消融证明。

### 两个代表性结果

- ChemLex -> BH：`+9.20` vs matched LLM，`+8.10` vs strongest BO。
- FreeSolv -> Lipophilicity：`+2.98` vs matched LLM，`+4.99` vs strongest BO。

### 这页最重要的诚实口径

“我们已经证明完整系统在当前冻结套件中有效，并证明 Gate 能阻止负迁移；但 Random-transfer、Similarity-only 和 Without-validation 还需要统一 confirm run，才能完成每个模块的因果归因。”

### 可能追问

**LLM 有没有打过确定性规则？**  
在当前四条部署路线中，完整 source-informed route 超过 matched target-only LLM；但 LLM Hypothesis 是否优于同结构的人工或 similarity-only patch，还没有在统一协议下完全证明。不要回避这一点，这正是下一轮 ablation 的核心。

**7/7 是否说明 CARE 总能变好？**  
不说明。它说明“带安全回退的完整部署”在这七条预设路线中不低于 matched LLM，并高于 strongest BO。原始 source route 仍然会失败。

### 过渡

除了最终质量，我们还关心同样的实验目标能不能用更少轮数达到。

---

## Slide 14｜预算效率与节省实验轮数

**建议时间：3 分钟**

### 建议讲稿

在真实科学任务中，最终最优值只是一个方面。更实际的问题是：达到同样质量需要做多少次 target 实验。

左图 A 计算达到 matched target-only LLM 最终质量时节省的轮数。四条部署路线分别节省：ChemLex -> BH `2.99` 轮，95% CI `[2.05, 3.93]`；Dielectric -> Gap `6.00` 轮，区间 `[5.14, 6.86]`；Gap -> Dielectric `5.03` 轮，区间 `[4.27, 5.79]`；FreeSolv -> Lipophilicity `2.23` 轮，区间 `[1.37, 3.09]`。

左图 B 看更严格的 global top-10 候选，四条路线分别节省 `3.62`、`9.57`、`8.46` 和 `4.80` 轮。左图 C 展示不同 round 的 best-so-far delta，可以看出迁移价值发生在早期还是主要出现在预算末端。

Round 0 是 initial observations，不计作 acquisition round。如果某条轨迹在预算内没有达到比较阈值，按 `budget + 1` 做右删失。Fallback 路线与 matched LLM 完全一致，因此 rounds saved 为零。

这页目前可以支持的结论是：四条通过 Gate 的路线更早达到 matched baseline 的最终质量，说明 source experience 有潜力减少 target 实验成本。

### 不要过度解释边际收益递减

一张按 round 展示的热图不能直接证明“历史经验越多，边际收益越低”。要证明边际收益递减，需要控制 source-history size，例如使用 10%、25%、50%、100% 历史数据分别重跑，再比较线性模型和饱和模型。当前只能说增益在不同轮次分布不均。

### 可能追问

**为什么用达到 baseline 最终质量，而不是全局最优？**  
Matched-final 对应一个现实问题：要达到不迁移方法最终能达到的质量，CARE 少做多少实验。Global top-10 是补充指标，两者一起报告比只看某一个阈值更稳妥。

### 过渡

当前结果说明框架值得继续，但论文级结论还需要一轮冻结协议下的完整模块归因。

---

## Slide 15｜下一轮统一协议与消融实验

**建议时间：3-4 分钟**

### 建议讲稿

下一轮不应该继续根据结果临时挑 pair 或改规则，而是先冻结 Confirmation Protocol。Confirm run 开始前需要固定 route list、calibration/held-out split、candidate pool、initial observations、预算、Gate thresholds、primary metric 和各任务 MDE。

表格中的每一行回答一个不同问题。

`No transfer` 给出完全不使用 source 的 target-only 表现。`Random transfer` 排除“只是多加一个 prior 就会变好”。`Similarity-only` 检查 schema 或 descriptor 相似度是否已经足够，不需要 LLM。`Without Gate` 量化强制迁移造成的负迁移。`Without validation` 检查独立 calibration 是否必要。最后用 Full CARE 2.0 得到完整系统的净效果。

所有方法必须共享同样的 target budget、initial observations、candidate pool 和 held-out seeds；除了被消融的模块，其余代码路径应保持一致。否则结果可能来自不同预算或不同初始难度，而不是模块贡献。

Confirm run 开始后，不能根据 held-out 结果修改 prompt、threshold 或 route。若发现协议问题，需要生成新的开发版本和新的独立 confirm split，并保留旧结果，不能覆盖。

### 这一轮最终要回答的五个问题

1. LLM Hypothesis 是否优于 similarity-only？
2. Source outcome 是否提供超出 schema 的信息？
3. Gate 是否显著降低负迁移率？
4. 增益来自 warm start、kernel geometry 还是持续 prior？
5. 同一协议能否在未参与开发的新 source-target pair 上复现？

### 还需要补的跨领域实验

应把领域内路线与 off-diagonal 路线分层报告。先根据表示重叠程度构造近距离跨领域路线，再测试远距离路线。无论结果正负都保留，用 task distance、descriptor overlap 和 route outcome 分析迁移边界。

### 过渡

最后总结当前已经成立的三条结论，以及我们暂时不能越过的边界。

---

## Slide 16｜结论

**建议时间：2 分钟**

### 建议讲稿

目前可以给出三条结论。

第一，同领域迁移已经得到初步且可复核的支持。反应、材料和分子三个领域内都出现了跨数据集正向路线。七条冻结路线中四条部署了 transfer，并在 100 个 held-out seeds 上超过 matched target-only LLM。

第二，Gate 在当前套件中阻止了负迁移。其余三条 raw route 中，两条明确负向、一条不稳定；它们都被拒绝并精确回退。正式 held-out 部署没有把这些已知不可靠路线上线。

第三，远距离跨领域迁移还没有验证。下一轮需要冻结统一协议，补齐 Random、Similarity-only、Without Gate、Without validation 等消融，并扩展 Reaction、Materials、Molecular 之间的 off-diagonal pairs。

CARE 2.0 当前最扎实的贡献不是“LLM 总能给出更好的实验”，而是建立了一条完整的可信迁移链：LLM 生成可解释 Hypothesis，Compiler 把它变成可执行 patch，Gate 用独立证据决定是否部署，held-out 验证效果，最后把成功路线、失败路线和审计记录沉淀进知识库。

### 建议收尾

“我们希望 CARE 2.0 最终回答的不是某一条经验能不能迁移，而是科学智能体如何系统地提出迁移假设、验证假设、拒绝不可靠迁移，并把这些结果变成下一项任务可复用的知识。”

---

## 会后常见问题与回答

### 1. 这是不是在 calibration 上先试很多次，再挑好的路线？

Calibration 的确用于模型选择，这是部署系统正常需要的步骤。关键约束是 route list、candidate portfolio 和 threshold 必须预先声明；calibration 与 held-out 不重叠；held-out 期间不再改规则。当前协议满足后两项，下一轮还要把所有 manifest、prompt hash 和 split 在 confirm run 前统一冻结。

### 2. Seed 是否相当于训练集和测试集？

不完全相同。Seed 定义一条独立的初始观察与顺序决策轨迹。Calibration seeds 用于选路线，held-out seeds 用于评价冻结路线。候选来自同一个冻结的真实数据池，因此它不是传统意义上重新采样的独立实验数据集。论文中应准确写成 disjoint trajectory seeds，并增加跨数据集、跨 pair 的外部验证。

### 3. Target outcome 是否真的没有泄漏？

算法只在选择一个候选后看到它的 outcome。未选择候选的结果对 LLM、router 和 acquisition 不可见。Held-out 结果不用于 patch 生成或 Gate 调参。需要继续发布 candidate-pool hash、split manifest、prompt record 和 audit，以便外部复核。

### 4. LLM 的输入和输出是什么？

输入是 target 的公开 schema/roles/budget、source 的已观测 outcomes/effect summaries/uncertainty，以及知识库中的成功失败 SkillCard。输出是结构化 Hypothesis 和 `KernelSkillPatch`，包括 role mapping、kernel scale、exploration schedule 和 source-prior strength。LLM 不直接读取 held-out outcome，也不直接给最终 candidate ID。

### 5. LLM 是否真的带来增益？

完整 LLM-generated route 在四条路线中超过 matched target-only LLM，说明 source-informed deployment 有净增益。但 LLM Hypothesis 相对于人工 rule、random patch 和 similarity-only 的独立贡献尚未全部完成统一消融。当前答案应是“有正向证据，但模块级归因仍需 confirm run”，而不是“已经完全证明 LLM 优于所有规则”。

### 6. 为什么不让 LLM 直接选实验？

直接选实验难以校准，也不容易保证预算、候选池和重复性。CARE 让 LLM 提出策略结构，再由 GP 和确定性代码执行候选排序，从而保留 LLM 的语义能力，同时控制随机性和协议偏移。

### 7. Gate 的阈值是否来自文献？

不是全部来自文献。Paired comparison、confidence interval 和多重比较控制有标准统计依据；0.60、0.80 和 0.25 是开发阶段冻结的 empirical defaults。下一轮必须做 sensitivity analysis，不能描述成学界通用阈值。

### 8. 7/7 超过 strongest BO 是否意味着所有迁移都成功？

不意味着。四条实际迁移成功；三条是 Gate 拒绝 source route 后使用 target-only LLM，而 target-only LLM 本身超过 strongest classical BO。正确表述是“完整安全部署 7/7 超过 strongest BO；source transfer 净增益为 4/7，另外 3/7 精确回退”。

### 9. 为什么把 final best 和 AUC 同时放进 Gate？

只看 final best 容易奖励最后一轮偶然命中，只看 AUC 又可能牺牲最终质量。两者同时非负，要求路线既不能在过程上持续变差，也不能在最终结果上亏损。Composite 用于统一排序，但所有分项仍需单独报告。

### 10. 节省轮数是否等于节省真实实验成本？

它是 replay 下的实验轮数代理，说明达到同等 target 质量需要更少查询。实际湿实验成本还受不同候选的材料、时间和失败概率影响，因此后续可以把每个候选的真实成本加入 cost-aware acquisition。

### 11. 目前是真实数据还是 synthetic data？

主结果使用真实公开或真实 wetlab 数据，包括 ChemLex、Buchwald-Hartwig、Matbench 和分子性质数据。Replay 本身是离线顺序重放，不是新开展的湿实验。Synthetic 数据只用于早期 smoke test，不应与当前主证据混在一起。

### 12. CARE 2.0 是否复现了 CARE 1.0？

CARE 2.0 沿用了“可执行 skill、评估和审计”的思路，但研究问题不同。CARE 1.0 更关注单任务内 skill evolution；CARE 2.0 关注历史任务中的 skill 和 outcome 如何迁移到新任务，以及如何检测负迁移。当前不能把 CARE 2.0 简单描述成 CARE 1.0 的复现。

### 13. 知识库中沉淀什么？

知识库应保存 Task Card、source-target role map、Hypothesis、可执行 patch、适用边界、Gate certificate、held-out outcome、accepted/rejected route、reasoning trace 和数据/代码版本。失败路线同样重要，它们告诉后续 router 哪些关系不应迁移。

### 14. 当前最大的不足是什么？

有三个。第一，真正 off-diagonal 跨领域路线仍为空白。第二，LLM、similarity、warm start、kernel 和 continuous prior 的贡献还没有在统一七路线协议下完全拆开。第三，Gate threshold 的敏感性和最终多重比较统计还需要冻结 confirm run。

### 15. 下一步最优先做什么？

先冻结统一 confirm protocol，再完成同预算、同 seeds 的六组 baseline/ablation；同时增加未参与开发的新 source-target pairs。只有这样，后续的正结果才能回答泛化问题，而不是继续证明现有七条路线可以被优化。

## 三分钟版本

CARE 2.0 解决的是如何把历史实验经验迁移到新任务，同时避免负迁移。LLM 读取 source 历史结果、target 的公开 schema 和知识库记录，生成结构化 Hypothesis；Compiler 把它变成 kernel、exploration 和 source-prior patch；Calibration Gate 用 50 个独立 seeds 决定 Transfer 或 Freeze；策略冻结后，再用 100 个不重叠 held-out seeds 评估。

当前冻结套件包含七条真实数据路线，覆盖反应、材料和分子任务。四条路线部署 source transfer，并同时超过 matched target-only LLM 与 strongest target-only BO；三条路线因为 raw transfer 负向或不稳定而精确回退。四条正向路线分别节省约 2.23 到 6.00 轮 target 实验。

因此，当前可以说 CARE 2.0 已经在三个领域中支持同领域跨数据集迁移，并证明 Gate 能阻止当前负迁移。还不能说远距离跨领域迁移已经建立，也不能说 LLM 的独立贡献已经完成全部消融。下一轮将冻结统一协议，补齐 Random-transfer、Similarity-only、Without Gate、Without validation 和完整系统对照，并增加新的 off-diagonal source-target pairs。

## 一句话版本

CARE 2.0 当前证明的不是“LLM 总能迁移”，而是“LLM 可以提出可执行的迁移假设，系统能够用独立证据决定采用还是拒绝，并在多个真实科学任务中获得可复核的质量和实验效率增益”。
