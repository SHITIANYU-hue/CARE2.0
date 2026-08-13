# CARE 2.0 项目现状汇报：逐页中文 Speaker Note

对应 PPT：`CARE2.0_current_status_2026-08-12.pptx`

使用方法：每页先讲“本页核心”，再按“详细讲解”展开；遇到追问时使用“名词解释”；最后用“转场”自然进入下一页。讲解中刻意区分了已经成立的结果、离线证据和仍待验证的结论。

## 第 1 页：封面：CARE 2.0 的定位

**本页核心：** CARE 2.0 的重点不是发明一个永远获胜的优化器，而是建立一套可以验证、拒绝和复用历史实验经验的迁移框架。

**详细讲解：**

开场先把项目定位讲清楚。CARE 2.0 研究的是：过去做过的实验，能不能在一个新实验刚开始、数据很少的时候帮助我们更快选到好条件。这里的 source 是已经完成、结果已知的历史实验，target 是准备开始的新实验。我们希望把 source 中真正有用的经验整理成可执行的 Skill，再用它改变 target 的开局；如果证据不足，就拒绝迁移，回到 target-only 方法。

右侧四个词可以按顺序解释：历史实验提供数据；系统把数据整理成可执行 Skill；Skill影响新实验的决策；新实验的成功和失败再回到知识库。最近最重要的进展，是 v1 在独立验证中失败后，我们没有隐藏失败，而是根据失败 trace 修改出 v2，并在一个冻结后才执行的外部 Suzuki target 上看到明显提升。

**名词解释：** source=历史任务；target=新任务；Skill=带执行逻辑、适用边界和证据记录的策略包。

**转场：** 下一页先给出目前最重要的结果，再逐步解释这个结果是怎么得到的。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-status-and-ppt/CARE2_CURRENT_STATUS_CN.md`

## 第 2 页：最新状态：先给结论

**本页核心：** 固定 v2 已有一个冻结外部正例；随后完成的真实 LLM 组件实验在五个 target 上平均 AUC 提高 2.12，但仍是回顾性证据。

**详细讲解：**

这一页把两条证据分开讲。第一条是严格的外部证据：Frozen v2 在 Suzuki MINLP2 上相对更强 target-only baseline 的 best-so-far AUC 提高 8.66。这里的 AUC 表示搜索过程中多早找到高质量实验。这个结果在 target outcome 揭示前已经冻结，但该切片没有使用 LLM。

第二条是随后完成的真实 LLM 组件实验。我们通过 CommonStack 实际调用 openai/gpt-5.4，让模型在看不到 target outcome 的情况下生成 initial-design hypothesis；按任务结构选择 raw 或 compiled 方案后，五个 target 的平均 AUC 相对 fixed v2 提高 2.12，4/5 不下降。不过 95% 置信区间是 [-0.226, 4.473]，而且这五个 target 以前都被项目使用过，所以这是回顾性组件证据，不是新的外部确认。

**名词解释：** AUC=搜索全过程的效率；Frozen=策略和参数在看 target 结果前已锁定；回顾性组件实验=在已有任务上检查模块行为，不能替代全新 target。

**转场：** 有了结果后，下面回到问题定义：我们到底在优化什么。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-v2-external-confirmation/README.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/README.md`

## 第 3 页：问题定义：有限预算的逐轮实验

**本页核心：** CARE 2.0 不是普通预测任务，而是每轮只能做一个实验、做完才看到结果的序贯决策问题。

**详细讲解：**

左边是算法开始前允许知道的信息：已经完成的 source experiments、target 的变量和候选条件、固定预算，以及尚未揭示的 target outcome。中间是每一轮真实发生的事：算法选择一个候选条件，只看到这个被选条件的结果，然后更新 target model，再做下一轮选择。右边是目标：在相同预算下更早找到高质量候选，同时避免因为错误迁移浪费实验。

这里与普通训练集、测试集的区别是，算法不能一次看到完整 target 表格。它必须像真实实验一样逐轮揭示 outcome。我们真正比较的是 source history 是否改变了 target 的选点顺序，以及这种改变是否让有效实验更早发生。

**名词解释：** candidate=可选择的实验条件；outcome=实验结果，例如产率；budget=最多允许做多少次 target 实验。

**转场：** 接下来需要区分不同强度的证据，避免把离线正结果和真正外部验证混为一谈。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/CARE2_METHOD.md`

## 第 4 页：证据阶梯：三种结果不能混着说

**本页核心：** 系统跑通、多任务离线比较和冻结外部验证是三个不同等级的证据。

**详细讲解：**

这张图从下到上表示证据逐渐变强。第一层是多领域 replay：它证明统一接口、指标、gate 和 audit 可以跨反应、分子、材料数据运行，也能在不适合迁移时精确回退。但这层使用 archived target outcomes 做 calibration，所以属于离线 benchmark。

第二层加入 RGPE、multi-task ICM GP 等传统 transfer BO 方法。它告诉我们经典迁移算法也会出现负迁移，没有一种方法在所有路径都可靠。第三层是最新 frozen external confirmation：策略在看外部 target outcome 之前已经冻结，然后才执行，因此最接近真实部署。

LLM 的贡献还要单独放在第四个问题里做因果对照。即使完整平台包含 LLM，也不能因为某个包含 LLM 的系统取得好结果，就默认增益来自 LLM。

**名词解释：** offline calibration=利用历史 target 结果选择方法；external confirmation=在策略冻结后才使用新 target 结果检验。

**转场：** 下面说明这些更严格的协议，是如何由团队此前的质疑直接推动出来的。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/CARE2_METHOD.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-08-classical-transfer-confirmation/README.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-v2-external-confirmation/README.md`

## 第 5 页：从 review 到代码和实验协议

**本页核心：** 对数据泄漏、baseline、手工规则和可审计性的质疑，已经被转换成具体代码修改与实验约束。

**详细讲解：**

这一页不是在列会议意见，而是在展示意见如何真正改变系统。关于开发测试边界，我们改成 campaign-level 划分，并冻结 selection record；关于 baseline 不够强，我们加入 100-seed random-initial GP-UCB、space filling、RGPE 和 ICM GP；关于规则是不是手工调得太强，我们把规则、配置、hash 和失败路径全部记录；关于结果是否可重放，我们增加 SkillBank、JSONL audit 和 SHA-256 校验。

最新 external target 最重要的作用，是回应“是不是先看了 target 结果，再决定该不该迁移”这个质疑。答案是：外部 target 的 outcome 在策略冻结前没有加载，执行记录也有对应 fingerprint。

**名词解释：** campaign-level split=按完整实验任务划分开发与验证；selection record=冻结的选点规则和参数记录。

**转场：** 协议明确以后，下面看最新 v2 的实际执行链路。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-confirmation/README.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-08-classical-transfer-confirmation/README.md`

## 第 6 页：v2 执行架构

**本页核心：** source 只负责前三个初始实验，之后所有方法都使用完全相同的 target-only GP-UCB。

**详细讲解：**

从左向右讲。首先，每个历史 campaign 单独训练一个 source expert。因为不同 campaign 的 outcome 尺度不同，我们不直接拼接原始数值，而是把每个 expert 的预测转成候选排序，再用中位数形成共识。TransferSkill 根据这个共识选择三个初始实验。

关键控制在黄色分界线之后：三个初始点做完后，source 信息完全退出。随后 10 轮所有方法都运行同一个 target-only GP-UCB，只用已经揭示的 target outcomes 建模。这样，Frozen v2、随机初始化和 space filling 之间唯一的区别就是前三个点怎么选，后续优化器、预算和观测规则完全一致。因此外部结果可以归因于 source-guided initial design，而不是后续偷偷使用了不同算法。

**名词解释：** source expert=在单个历史 campaign 上训练的模型；initial design=优化开始前先做的少量初始实验。

**转场：** 为了避免误解，下一页把完整 CARE 平台和这次最小验证切片分开。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`

## 第 7 页：完整平台与 v2 验证切片

**本页核心：** CARE 2.0 已实现更多模块，但最新外部结果只验证其中最小、最容易归因的一条路径。

**详细讲解：**

左侧是完整 CARE 2.0 平台：LLM 可以读取 source/target schema，生成 role mapping 和 hypothesis；系统也支持 source prior、kernel patch、online router、calibration gate 和 audit。右侧是最新 v2 外部验证：只使用真实 source outcomes 生成 quality-bounded initial design，三个点之后 source 完全退出，而且 LLM 不参与选点或打分。

这样设计不是削弱 CARE，而是为了把因果关系讲清楚。如果一开始把 LLM、kernel、router 和 source prior 全部叠上去，即使结果提高，也不知道是哪一个模块起作用。现在先钉牢 source-outcome initial design 的价值，下一步再在相同执行器上加入 LLM，直接测 LLM 的增量。

**名词解释：** 验证切片=从完整系统中抽出一个可独立检验的最小模块组合；增量=加入某模块后相对固定对照多出来的收益。

**转场：** 下面解释 replay harness 如何模拟真实实验节奏并防止偷看答案。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/CARE2_METHOD.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`

## 第 8 页：Replay Harness 如何运行

**本页核心：** 完整数据表只充当隐藏环境；算法每轮只能看到自己已经选择过的实验结果。

**详细讲解：**

Replay 的做法是把已有真实数据集当作一个可以查询的实验环境，而不是把整张表交给算法。开始时 target outcomes 全部隐藏。每一轮算法选择一个 candidate，环境只揭示这个点的真实 outcome，算法据此更新模型，再选择下一点。每轮的 candidate、score、outcome 和当前 best-so-far 都进入 trace。

所有对照使用相同 candidate pool、相同初始点数量、相同 reveal budget 和相同 seeds。这里 seed 是随机数种子，用来复现实验随机性，不是提前挑选的好实验点。Replay 仍然不能替代新的 wet-lab 实验，但它能严格检查数据泄漏，并评价有限预算下哪种策略更早找到好条件。

**名词解释：** seed=控制随机过程的编号；reveal=执行一个候选后揭示结果；trace=逐轮决策记录。

**转场：** 有了统一 replay 协议，下面看目前接入了哪些真实数据。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/CARE2_METHOD.md`

## 第 9 页：数据集覆盖

**本页核心：** 平台已覆盖反应优化、分子性质和材料性质，但真正冻结的外部正例目前只在反应优化中。

**详细讲解：**

反应优化包括 ChemLex Acid-Amine、Buchwald-Hartwig HTE、Baumgartner C-N 和 Suzuki MINLP；分子性质包括 ESOL、FreeSolv 和 Lipophilicity；材料性质包括 Matbench experimental gap、dielectric 和 phonons。它们让我们可以测试不同 schema、变量类型和 outcome 尺度下的统一接口。

需要特别说明，数据集多不等于每个领域都已证明迁移成功。分子和材料结果主要来自离线 calibration suite，用于验证平台兼容性、方法选择和 fallback；目前最强的冻结外部证据只来自 Suzuki MINLP2。因此汇报时应说“平台已跨领域运行”，而不是“跨领域迁移已经全面成立”。

**名词解释：** schema=变量字段、类型和约束的结构；Matbench/MoleculeNet=公开的材料与分子性质 benchmark 集合。

**转场：** 不同数据能够统一接入后，下一步是把自然语言经验变成可执行 hypothesis。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/overview.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/CARE2_METHOD.md`

## 第 10 页：Hypothesis 的生成与执行

**本页核心：** Hypothesis 只有被编译成明确的选点和停止规则，才真正进入实验。

**详细讲解：**

完整平台给 hypothesis generator 的输入包括 source history、target 的变量角色、候选空间、预算以及 SkillBank 中的成功和失败案例。输出不能只是“这个 source 看起来相似”，而必须结构化为：适用哪些 source、变量如何对齐、什么条件应该优先或惩罚、允许进入什么质量区域、置信度多高，以及在什么情况下应该失败或回退。

结构化输出再被编译成 TransferSkill，用来选择 candidate。target outcome 揭示后，系统更新 evidence status：支持、反驳、暂缓或者拒绝。最新 v2 的 hypothesis 不是 LLM 生成，而是固定规则，目的是先验证从 hypothesis 到执行再到反馈的链路。

**名词解释：** role map=source 和 target 变量的语义对应关系；operator=把 hypothesis 转成选点动作的执行算子。

**转场：** 下一页具体说明一个 Skill 里到底保存什么。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/CARE2_METHOD.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`

## 第 11 页：TransferSkill 的内容

**本页核心：** Skill 是一个可重放、可审计、带失败条件的策略包，而不是一句 prompt。

**详细讲解：**

一个合格的 Skill 至少包含五部分。第一是任务边界：source、target、变量角色以及不能对齐的字段。第二是执行逻辑：它影响 initial design、source prior、kernel 还是 router。第三是安全机制：哪些情况应该 abstain，何时 exact fallback。第四是证据记录：在哪些开发任务和独立任务上运行，使用哪些 seeds，CI、win、non-loss 和 negative cases 是什么。第五是可复现信息：模型记录、配置、commit 和 SHA-256 fingerprint。

因此 SkillBank 不是文档仓库，而是可执行对象与证据的集合。Baumgartner v2 当前应该标注为“在一个外部 target 上得到支持的 candidate skill”，不能标成通用规则。

**名词解释：** abstention=系统主动不迁移；fallback=精确回到预先定义的 target-only 方法；fingerprint=用于确认版本没有被事后修改的哈希。

**转场：** 下面看 v2 如何把多个 source campaign 合成一个稳健共识。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/CARE2_METHOD.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/skill_banks/care2-baumgartner-initial-design/SKILL.md`

## 第 12 页：多个 Source Expert 如何聚合

**本页核心：** 每个 source 单独建模，先转成相对排序，再用中位数聚合，避免量纲和异常 campaign 主导结果。

**详细讲解：**

第一步，每个 source campaign 单独训练 GP，保留各自内部的变量与 outcome 关系。第二步，把每个 expert 对 target candidates 的预测转成 0 到 1 的 rank score。这样我们只比较“这个 source 更偏好哪个候选”，不直接混合不同 campaign 的产率尺度。第三步，对多个 source 的 rank 取中位数，得到 consensus prior。

中位数的意义是稳健：如果某一个历史 campaign 对某些候选给出极端高分，它不会轻易压过其他 source。这里的 prior 不是 target 的真实结果，只是多个 source 对候选顺序的共同意见。

**名词解释：** GP=高斯过程，用少量数据给出预测均值和不确定性；rank score=候选在一个 source 预测中的相对名次。

**转场：** 有了共识排序以后，v2 还增加了一个关键限制：多样性不能跨出 source 支持的质量区域。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`

## 第 13 页：v1 失败如何变成 v2 质量约束

**本页核心：** v2 先限定 source 共识的高质量区域，再在区域内部选择互补实验。

**详细讲解：**

v1 的第一个点取 source 排名最高的候选，第二、第三个点主要追求距离远、覆盖广。问题是“离得远”不代表“质量仍然好”，所以多样性可能把点推入 source 自己也不看好的区域。

v2 保留第一个 source consensus 最优点，但先建立 top-50% admissible region，也就是只允许第二、第三个点从 source 排名前一半的候选中产生；然后在这个质量区域内用 maximin 选择与已选点距离最大的候选。它同时保留质量和覆盖。前三个点选完后 source prior 退出，后面仍是统一的 target-only GP-UCB。

这里的 top-50% 是当前冻结协议中的经验参数，不应称为学界通用常数，后续需要做 threshold sensitivity。

**名词解释：** admissible region=允许选点的候选区域；maximin=选择与现有点最小距离最大的候选，用于扩大覆盖。

**转场：** 前三个点不同以后，后续怎么保证比较公平？下一页解释统一的 GP-UCB。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`

## 第 14 页：统一的 Target-only GP-UCB

**本页核心：** 三种方法在初始点之后使用同一个优化器，因此差异只来自 initial design。

**详细讲解：**

GP-UCB 每轮给每个候选计算“预测均值加探索奖励”。预测均值代表当前看起来有多好，不确定性代表这个点还有多少未知信息，beta 控制探索与利用的平衡。算法选择 UCB 分数最高的候选执行。

公平性有四条：使用相同的 target observations；相同 GP-UCB 实现；相同 10 次 reveal budget；相同 candidate pool 和未选择结果不可见。Frozen v2、random initial 和 space filling 的前三个点不同，但从第四个观测开始完全运行同一套 target-only GP-UCB。因此如果 v2 曲线更高，是因为开局把优化器带到了更有价值的区域。

**名词解释：** UCB=预测均值加不确定性奖励；exploration=探索未知区域；exploitation=利用当前高预测区域。

**转场：** 下面说明我们用哪些指标评价一个策略，不只看最后一轮。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`

## 第 15 页：评价指标

**本页核心：** 主指标看整个搜索过程，辅助指标看最终质量、胜负、负迁移和实验节省。

**详细讲解：**

Primary metric 是 best-so-far AUC。每一轮记录到目前为止观察到的最好 outcome，再对整条曲线求平均或面积。它奖励“尽早找到好条件”，特别适合实验昂贵的场景。Secondary metrics 包括 final best、达到阈值所需轮数、win/non-loss、negative transfer rate 和 gate acceptance rate。

需要区分两种统计单位。对多个独立 campaign 的开发或验证结果，可以在 task level 计算置信区间；同一个 target 上的 100 个 random seeds 只描述随机初始化分布，不能当作 100 个独立科学任务。因此外部 target 的 +8.66 是很大的实际 effect，但暂时不能称为跨任务统计显著。

**名词解释：** best-so-far=截至当前轮找到的最好结果；CI=置信区间；effect size=提升幅度本身，而非只看 p 值。

**转场：** 评价指标确定后，系统需要一个 gate 来决定哪些策略有资格被冻结。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-v2-external-confirmation/README.md`

## 第 16 页：Calibration Gate

**本页核心：** Gate 先检查稳定性和负迁移风险，再从合格策略中选择平均收益最高者。

**详细讲解：**

第一步是资格检查：开发任务上 AUC delta 的 95% CI lower bound 必须大于 0，同时 task non-loss rate 至少为 0.8。前者要求平均收益具有统计支持，后者要求不能靠少数大胜掩盖大量失败。第二步只在 eligible routes 中选择 mean AUC gain 最大的策略，CI lower bound 用作并列时的次级标准。

要诚实说明阈值来源：CI lower > 0 有明确统计含义；non-loss 0.8 是本项目预先冻结的经验门槛，不是普遍公认常数。后续应对 0.7、0.8、0.9 以及 admissible region 阈值做 sensitivity analysis，确认结论不依赖单一超参数。

**名词解释：** gate=迁移准入与路由机制；non-loss=相对 baseline 没有变差的任务比例；eligible=通过资格检查。

**转场：** 有了 gate 以后，还要说明我们到底与哪些强对照比较。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`

## 第 17 页：Baseline Protocol

**本页核心：** Frozen v2 必须同时超过随机初始化和确定性的空间覆盖，对比口径完全一致。

**详细讲解：**

第一列 Frozen v2 的前三个点由 source-guided skill 选择；第二列 random-initial GP-UCB 的前三个点随机选择，并用 100 个 seeds 描述随机性；第三列 space filling 用确定性规则最大化候选空间覆盖。三者之后都运行同一个 target-only GP-UCB、做 10 次 reveal，而且都不能预读 target outcome。

Random baseline 不是每轮都随机。只有前三个初始点随机，后续仍然是强的 GP-UCB。Space filling 也不是弱对照，它能给 GP 一个分散且覆盖面好的初始设计。主比较对象是每个 campaign 上表现更强的 target-only baseline，而不是挑一个最弱对照。

**名词解释：** space filling=优先覆盖参数空间的确定性初始设计；stronger baseline=在给定 target 上两种 target-only 对照中更强者。

**转场：** 下面先回顾 v1 为什么在开发集看起来很好。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`

## 第 18 页：v1 开发结果

**本页核心：** v1 在 9 个开发 campaign 上平均正向并全部 non-loss，因此当时看起来具备冻结资格。

**详细讲解：**

v1 的开发结果是 mean AUC delta +1.321，task-level 95% CI 为 [0.315, 2.326]，下界大于 0；9 个 campaign 中 7 个获胜，9 个都没有输。按照预先定义的 gate，它确实是一个合格候选，而不是随意挑出来的规则。

v1 的主要设置是优先 same-substrate source，其次 same-precatalyst，diversity weight 为 0.85，后续指标是 10 轮 target-only GP-UCB 的 best-so-far AUC。但开发任务几乎都能找到 same-substrate source，所以 fallback 场景没有得到充分检验。这是为什么开发集漂亮仍然不能代替独立验证。

**名词解释：** win=相对 stronger baseline 提升；non-loss=提升或持平；fallback=首选迁移条件不满足时使用的备用策略。

**转场：** 真正的独立任务到来后，v1 的问题暴露出来。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-confirmation/README.md`

## 第 19 页：v1 独立验证失败

**本页核心：** 冻结后的 v1 在 4 个独立 campaign 上平均变差，说明开发收益没有泛化。

**详细讲解：**

冻结后，v1 在 4 个新的 campaign 上 mean AUC delta 为 -1.205，95% CI 为 [-2.927, 0.518]，只有 1 个获胜，虽然 4 个都没有造成极端崩溃，但 final-best delta 为 -1.61。最大失败是 Morpholine + BuBrettPhos。

原因不是后续 GP-UCB 不公平，而是新 substrate 没有 same-substrate source，只能走 same-precatalyst fallback。v1 的多样性机制又没有限制第二、第三个点必须处于 source 高质量区域，因此开局可能被带到覆盖广但质量低的位置。这里最重要的结论是：开发集平均正向不等于冻结策略能泛化。

**名词解释：** independent campaign=未用于规则或超参数选择的新实验任务；final-best delta=预算结束时最好结果之差。

**转场：** 下一页展示我们如何把这个失败转化为 v2，而不是继续在测试任务上调 v1。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-confirmation/README.md`

## 第 20 页：失败驱动的 v2 修订

**本页核心：** v2 修复的是明确的 failure mode：多样性可以扩大覆盖，但不能跨出 source 支持的质量边界。

**详细讲解：**

左侧是 v1 failure trace：第一个点通常位于 source 高排名区域；第二、第三个点受 diversity 加权影响，会远离已选点，却可能进入 source 低质量区域；新 substrate 下 fallback 风险进一步放大。右侧是 v2 的对应修改：第一个点仍取 consensus 最优；建立 source top-50% admissible region；第二、第三个点只在该区域内做 maximin；旧四个验证任务仅用于 post-hoc stress，不重新计作新证据。

这一步体现 CARE 的知识沉淀方式：失败不只是一个负数，而是形成“什么条件下多样性会破坏迁移”的 negative case，并转化为新的 Skill failure condition。

**名词解释：** post-hoc stress=看到结果后用于理解失败和压力测试，不能作为新的独立验证；failure mode=可重复描述的失败机制。

**转场：** 下面用整条证据链说明，哪些结果只是开发，哪个结果真正改变了结论。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-confirmation/README.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`

## 第 21 页：从 v1 到 v2 的证据链

**本页核心：** 只有冻结后外部正向结果能支持泛化；开发结果和 post-hoc 结果不能替代它。

**详细讲解：**

从左到右看五根柱。v1 在 9 个开发任务上 +1.32，但在 4 个独立任务上变成 -1.20；这说明 v1 没有泛化。v2 在 9 个开发任务上 +1.39，说明它可以成为新的冻结候选，但仍不算外部证据。真正改变项目状态的是 v2 在外部 Suzuki MINLP2 上 AUC +8.66。最右侧旧四任务 post-hoc 约 +0.38，只能支持修复方向，不能当成新的 held-out evidence。

汇报时要强调这条逻辑：开发用来选策略，独立验证用来检验策略；独立验证失败后可以分析和修复，但旧验证集从此变成开发信息，不能反复计算成新的泛化证据。

**名词解释：** held-out=冻结前未用于选择策略的数据；post-hoc=看过结果以后进行的分析。

**转场：** 接下来详细介绍真正的外部验证任务。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-confirmation/README.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-v2-external-confirmation/README.md`

## 第 22 页：外部 Suzuki MINLP2 设置

**本页核心：** MINLP1 作为完整 source，MINLP2 在策略冻结前只暴露 schema，不暴露 outcome。

**详细讲解：**

左侧 Baumgartner Suzuki MINLP1 是完成的 source campaign，完整 outcomes 可用于训练 source expert。右侧 MINLP2 是 frozen target，在冻结前只知道候选变量与实验预算，不加载产率结果。两者共享 precatalyst、temperature、residence time 和 loading，因此 source-target 之间有明确的变量对应与反应优化基础。

执行前冻结五类信息：代码 commit、data version、config、selection record 和 skill fingerprint。然后 target 只按 3 个初始点加 10 次 reveal 的预算运行。这个设计防止我们看过 MINLP2 的答案再修改 v2。

**名词解释：** MINLP=混合整数非线性规划形式的反应优化数据；schema-only=只读变量结构，不读结果值。

**转场：** 设置说清楚以后，下一页直接看 Frozen v2 与两个 target-only baseline 的结果。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-v2-external-confirmation/README.md`

## 第 23 页：外部结果：过程与终点同时改善

**本页核心：** Frozen v2 的优势不只体现在最终最好值，也体现在更早进入高产率区域。

**详细讲解：**

左图比较 best-so-far AUC：Frozen v2 为 98.06，space filling 为 89.40，100 次 random 的均值为 88.84。因为 space filling 是两个 target-only 对照中更强者，主结果写成相对 stronger baseline +8.66。右图比较预算结束时的 final best：v2 找到 100% 产率，space filling 为 91.82，random 平均为 94.77。

100 个 random runs 中，只有 15 个 AUC 达到或超过 v2，说明 v2 位于随机初始化分布的高端。这里要避免把 100 seeds 写成 100 个独立任务；它们只说明同一个 target 上随机初始化的波动。最稳妥的结论是“在一个外部 target 上观察到大的实际提升”。

**名词解释：** final best=预算结束时找到的最高 outcome；random mean=同一算法更换随机初始化后的平均结果。

**转场：** 实际 wet-lab 最关心的是少做多少轮，下一页把曲线差异翻译成实验次数。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-v2-external-confirmation/README.md`

## 第 24 页：实验轮数节省

**本页核心：** 在 MINLP2 上，v2 达到 100% 产率所需总观测数比成功随机运行的中位数少 2 次。

**详细讲解：**

Frozen v2 的总观测数是 5：先执行 3 个 source-guided initial points，再做 2 次 target-only GP-UCB reveal，就已经找到 100% 产率。100 个 random-initial runs 中，只有 46 个在总预算内达到 100%；这些成功运行达到 100% 的中位数是 7 次总观测。由此得到“节省 2 次 target observations”。

右侧另一个数字是 15/100：只有 15 个随机运行的全过程 AUC 不低于 v2。Space filling 在预算内甚至没有达到 95%。因此 v2 的价值不是简单多算了模型，而是在相同实验预算下把最有价值的 target 实验提前了。

**名词解释：** 总观测数包含 3 个初始点；节省 2 次是相对成功 random runs 的中位数，不是所有 random runs 的平均。

**转场：** 单个外部结果之外，我们还有一个更广的离线多领域 suite，但它的证据等级较低。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-v2-external-confirmation/README.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-v2-external-confirmation/confirmation/efficiency_summary.json`

## 第 25 页：多领域离线 Suite

**本页核心：** 离线 suite 证明平台能跨数据类型选择迁移或回退，但不能代替全新任务上的零成本决策。

**详细讲解：**

表中 7 条 source-target 路径覆盖反应、分子和材料。系统在 ChemLex 到 Buchwald-Hartwig、dielectric 与 experimental gap 的双向路径、FreeSolv 到 Lipophilicity 上选择 transfer；在 phonons 到 dielectric、ESOL 到 Lipophilicity、Lipophilicity 到 FreeSolv 上选择 fallback。正数表示相对 target-only baseline 的 final-best 和 AUC 改善，0 表示精确回退后与 baseline 完全相同。

限制写在红框里：这套选择使用了 50 个 archived target calibration seeds。也就是说系统利用历史 target outcomes 判断哪条路线更适合，解决的是离线方法选择问题。它不能证明面对一个从未做过的新 wet-lab target，系统可以零成本知道该不该迁移。

**名词解释：** exact fallback=拒绝迁移后与预先定义的 target-only baseline 完全一致；calibration seed=用于离线选方法的随机重复。

**转场：** 为了判断我们的 baseline 是否足够强，下一页加入传统 transfer BO。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-07-24-source-outcome-transfer/README.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/overview.md`

## 第 26 页：传统 Transfer BO 对比

**本页核心：** RGPE、ICM 和混合路由各有优势，但没有一个固定迁移方法在所有任务都可靠。

**详细讲解：**

RGPE 会根据 source 模型在 target 观测上的排序损失给不同模型加权。它在 dielectric 到 experimental gap 上非常强，但在另外多条 final-score 比较中显著变差，说明经典方法也会负迁移。Two-task ICM GP 通过多任务协方差联合建模，整体更保守，但仍不是普遍最优。Hybrid router 在 calibration 信息帮助下选择方法，AUC 在五条路径上为正，支持路由机制的价值。

必须解释深蓝框中的限制：在这五条路径里，CARE 自己的 source-transfer candidates 全部被 gate 拒绝，所以 CARE 的正向结果主要来自 target-only fallback。不能把“系统避免了负迁移”写成“CARE source transfer 全面击败传统方法”。当前平台价值是比较、路由、拒绝和审计。

**名词解释：** RGPE=按目标任务上的排序表现给多个 GP 专家加权；ICM=用任务间协方差联合建模的多任务 GP。

**转场：** 下面回到 LLM：它在完整平台里做什么，以及目前为什么还不能宣称它有独立优势。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-08-classical-transfer-confirmation/README.md`

## 第 27 页：LLM 与 SkillBank

**本页核心：** LLM 负责提出和编译可检验的迁移假设，SkillBank 保存执行证据；最新外部增益并不来自 LLM。

**详细讲解：**

完整平台中，LLM 的输入是 source/target schema、统计摘要、已有 skill 与失败案例；输出是结构化 role map、hypothesis、kernel/prior/routing patch、rationale、confidence 和 failure condition。它更像策略专家或 hypothesis compiler，而不是直接预测每个实验 outcome。

SkillBank 保存 task cards、role maps、frozen TransferSkill、正负案例、被 gate 拒绝的案例、逐轮 candidate-score-outcome trace、commit、config 和模型记录。这样一个新任务可以检索相似经验，同时也能看到经验在哪里失败。

当前结论要克制：此前确实看到过 LLM 的正向信号，但 random-rule warm start 也能提高，部分 CARE 结果与 target-only LLM 相同。最新 v2 external 完全没有使用 LLM，所以它只能证明 source-outcome initial design。下一步必须在同一 v2 executor 上做 fixed v2、LLM hypothesis-only 和 matched random hypothesis 三组对照。

**名词解释：** hypothesis-only=LLM 只生成迁移假设，不接触 target outcome，也不改变后续优化预算；matched random=输出格式和复杂度匹配的随机假设对照。

**转场：** 下面进入本轮新增的真实 LLM 实验：它具体读什么、输出什么，以及是否超过 fixed v2。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/CARE2_METHOD.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/skill_banks`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/README.md`

## 第 28 页：真实 LLM 如何进入执行链

**本页核心：** LLM 不再只是写解释，而是在看不到 target outcome 的情况下冻结一个可执行 initial-design hypothesis。

**详细讲解：**

这一页讲清楚本轮新增的真实 LLM 调用。输入包括完成的 source outcomes、source 层统计、target schema、公开候选条件和 source-prior 排名短名单；明确不包含任何 target outcome。使用 CommonStack 上的 openai/gpt-5.4，模型输出一个 JSON hypothesis：三条 candidate IDs、各自角色、机制解释、置信度、失败条件和是否 abstain。原始 prompt、原始 response、token usage、API timing 和 SHA-256 都保存。

输出有两种执行方式。Raw LLM 直接执行模型选出的三个点；Compiled LLM 把模型提供的语义锚点，与 source 共识最优点和 top-50% 质量区内的 maximin 几何探针组合。三点以后，两种方法都退出 source 和 LLM，统一运行 target-only GP-UCB。这样 LLM 真正改变了实验开局，但后续预算和优化器仍可公平比较。

**名词解释：** semantic anchor=LLM 根据科学机制挑出的重点条件；compiler=把语义建议转换成满足质量和几何约束的执行方案；abstain=LLM 主动拒绝迁移。

**转场：** 下一页用 Suzuki MINLP2 展示 raw LLM 为什么失败，以及安全 compiler 如何把它变成正增益。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/README.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/scripts/run_llm_initial_design_hypothesis.py`

## 第 29 页：Suzuki LLM Case Study

**本页核心：** LLM 给出了正确的反应区域，但直接选三点覆盖不足；安全编译后 AUC 反超 fixed v2。

**详细讲解：**

真实模型在不知道 MINLP2 产率的情况下提出：同底物 source 支持高温、高 loading 的 P2L1 XPhos Cl 区域，温度是主要迁移因子，置信度 0.73。Raw LLM 选了三个都集中在该催化剂家族的点，虽然初始质量不错，但给后续 GP 的几何信息不足，AUC 只有 91.819，低于 fixed v2 的 98.061。

安全 compiler 没有丢掉 LLM。它保留 LLM 的 candidate 024 作为 semantic anchor，加入 source 共识最优 candidate 039，再从 source top-50% 区域中选择 maximin geometry probe 013。这个三点组合 AUC 达到 100，比 fixed v2 高 1.939，二者 final best 都是 100%。这说明合理角色不是让 LLM 替代数值优化，而是让 LLM提出可解释的科学方向，由 compiler 保证可优化性。

**名词解释：** Raw LLM=直接执行三点；Compiled LLM=保留一条 LLM 语义决策并加确定性安全约束；AUC=越早找到好结果越高。

**转场：** 单个 case 还不够，下一页汇总五个 target，看 LLM 增益是否只出现在 Suzuki。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/suzuki_minlp2/evaluation_compiled/summary.json`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/suzuki_minlp2/llm_hypothesis_record.json`

## 第 30 页：五任务 LLM 组件结果

**本页核心：** 任务结构路由在五个 target 上平均 AUC 比 fixed v2 高 2.12，4/5 不下降，但置信区间仍跨零。

**详细讲解：**

五个 target 都是真实 API 调用，共五条主 hypothesis、66,293 tokens。候选路由完全由任务结构决定：只有一个 completed source 时采用 compiled LLM；有多个 source 时采用 raw semantic design。每个 target 的后续 GP、预算和 candidate pool 与 fixed v2 完全一致。

逐任务 AUC delta 是：Suzuki +1.939，Morpholine-AlPhos +5.203，Phenethylamine-AlPhos -0.947，Morpholine-tBuBrettPhos +4.422，preliminary 0。平均 +2.123，win 3/5，non-loss 4/5，final best 5/5 不下降。95% CI 为 [-0.226, 4.473]，仍然跨 0，所以不能写成统计上已经证明 LLM 普遍优于 fixed rule。

还要说明，这条路由是在观察组件行为后形成，五个 target 也都曾被项目使用，因此属于 retrospective ablation。正确下一步是冻结 prompt、compiler 和 route，在全新的 target 上执行。

**名词解释：** 任务结构路由=按 source 数量选择 raw 或 compiled，不读取 target outcome；retrospective ablation=回顾性组件实验，不是新的外部确认。

**转场：** 有了这组结果后，再重新划分哪些结论已经成立、哪些仍然需要外部确认。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/aggregate/suite_summary.json`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/README.md`

## 第 31 页：结论边界

**本页核心：** 已经成立的是可信迁移流程、一个外部固定规则正例和真实 LLM 的正向组件信号；尚未成立的是 LLM 外部普遍优势。

**详细讲解：**

左栏是可以明确说的：我们建立了统一、可审计的 replay harness；固定 v2 在一个真实外部 Suzuki target 上相对强 baseline 的 AUC 提高 8.66；真实 LLM 已经生成并执行结构化 hypothesis；五任务回顾性组件实验平均 AUC 提高 2.12。

中栏是不能说的：任意跨领域稳定迁移、LLM 已在全新 target 普遍优于 fixed rule、task-level CI 已排除零增益，以及 source 在优化全程持续贡献。右栏是项目真正的定位：可信赖的科学知识迁移框架，强调强 baseline、冻结协议、负迁移审计、exact fallback，以及成功和失败共同沉淀。

一句话收束：历史实验可以形成可执行、可拒绝、可验证的 Skill，但每个泛化结论都要由冻结后的新任务来支持。

**名词解释：** 可信迁移不等于每次都迁移；能够识别不该迁移并安全回退，也是系统能力。

**转场：** 最后说明怎样把一个外部正例扩展成论文级泛化证据。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-status-and-ppt/CARE2_CURRENT_STATUS_CN.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/README.md`

## 第 32 页：下一步实验

**本页核心：** 优先冻结 prompt、compiler 和 route，再用真正外部 target 检验 LLM 的独立增量。

**详细讲解：**

P0 第一项是扩大 external targets：冻结同一个 v2 skill family，在更多未参与开发的 C-N、Suzuki 或相邻 reaction campaigns 上执行。只有多个独立 target 才能计算 task-level CI。P0 第二项是冻结统一 Confirmation Protocol，包括任务根列表、预算、primary metric、selection hash 和 skill fingerprint，防止每个结果出来后改变口径。

针对 LLM，还要额外冻结 prompt、模型版本、raw/compiled route、compiler 参数和 trace schema。面对新 target 时不能再根据结果选择哪条路线。确认实验仍然比较 fixed v2、LLM hypothesis-only 和 matched random hypothesis，所有组使用相同 source、candidate pool、initial-design size 和 target budget。之后再研究 skill accumulation 和远距离领域扩展。

**名词解释：** Confirmation Protocol=事先冻结任务、预算、指标和版本的确认性实验协议；skill accumulation=随着完成任务增加，SkillBank 是否带来可重复收益。

**转场：** 最后一页准备回答最容易被问到的四个问题。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-status-and-ppt/CARE2_CURRENT_STATUS_CN.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/README.md`

## 第 33 页：答疑与讨论口径

**本页核心：** 这四个问题决定目前能把结论说到哪一步。

**详细讲解：**

问题一：是不是挑了三个好 seed？不是。前三个点是冻结的算法输出，不是随机 seed；external target 在冻结前没有读取 outcome。问题二：为什么不持续使用 source prior？因为当前实验先隔离 initial-design contribution，持续 prior 需要单独与 warm-start-only 比较，否则归因不清楚。

问题三：LLM 是否已经优于 fixed v2？目前五个既有 target 平均 +2.12，4/5 不下降，但 CI 跨零，所以只能说有正向信号。问题四：这算跨领域吗？目前主要是不同 reaction-optimization datasets 之间的迁移，属于同领域或相邻任务迁移，还不足以证明化学到材料等远距离跨领域泛化。

讨论应收束到三个具体决定：下一批 external targets 是什么；LLM hypothesis-only 对照如何冻结；Confirmation Protocol 和 SkillBank 的版本规则如何确定。

**名词解释：** 最稳妥的对外表述是：CARE 2.0 已获得一个严格外部 source-outcome transfer 正例，并形成了可扩展的验证框架。

**转场：** 结束汇报并进入任务、协议和专家 review 的具体讨论。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-status-and-ppt/CARE2_CURRENT_STATUS_CN.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/README.md`
