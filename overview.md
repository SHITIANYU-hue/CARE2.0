# Overview

## 2026-08-16：Opus 高权限在线控制与冻结化学扩展

这一轮把 LLM 从受限打分器改成了逐轮决策者。每一轮先由 Opus 4.8 提出下一项
实验，再由独立的 Opus critic 审核；最终可以在 target-only GP-UCB 前五名中选择，
不再由 Python 代码替 LLM 固定方向。候选菜单同时给出 GP posterior、UCB、
probability of improvement、expected improvement、source prior 和 rank-fusion
证据。LLM 的参与率和决策权都是 100%，平均每轮有 4.99 个可执行候选，实际覆盖
GP 默认选择的比例为 45.5%。每次 proposal、critique、最终选择和目标结果 reveal
都保存在 trace 中。

开发集覆盖 MoleculeNet 和 Matbench 的 6 条真实路线，在线 Opus 相对“相同 LLM
初始点 + target-only GP-UCB”为 3 胜、1 平、2 负，平均 best-so-far AUC
提升 `+1.8403`。控制器冻结后又跑了 5 条真实化学路线，包括 4 条 Baumgartner
C-N 和 1 条 Reizman 多源 Suzuki；这部分为 3 胜、1 平、1 负，平均 AUC
提升 `+0.0212`。两部分合计 11 条路线，结果为 6 胜、2 平、3 负，平均 AUC
提升 `+1.0135`。因此目前可以说在线控制器在多数已测路线中有正向增量，而且信号
跨越分子性质、材料性质、C-N 和 Suzuki；不能说所有迁移都有效。

完整系统相对固定 source-diverse 初始方案的平均 AUC 只提升 `+0.3867`，并且只有
5 胜、2 平、4 负。主要问题是 outcome-blind LLM 初始设计还不稳定。冻结实验之后
补做了 compiled initial-design 诊断，它修复了最差的一条 C-N 初始路线，但整体在线
结果只有 1 胜、3 平、1 负，因此没有被选为新默认方案。这个诊断明确标为
post-holdout，不能回写成冻结实验结果。

代码、逐轮 trace、汇总表和结论边界见
`experiments/care_replay/results/2026-08-16-opus48-generalization-evidence-v1/`。
开发集、冻结扩展和事后诊断分别保存在对应的三个 `2026-08-16-opus48-*` 目录中，
所有关键 JSON、CSV 和 Markdown 文件都带 SHA-256 校验。

## 2026-07-28：baseline、逐轮效率与 LLM 规则公式补全

按 7/28 评审意见补了一套统一的评估说明。新的汇总不再只列某个方法相对
incumbent 的结果，而是把正式七条 source-target 路径同时对齐到两类主对照：
每条路径最强的 target-only BO，以及预算、初始点和 seed 完全一致的
matched target-only LLM。50 个 calibration seeds 只负责冻结“迁移或精确回退”，
100 个不重叠的 held-out seeds 才用于报告。四条部署迁移的路径同时超过两类
对照；另外三条精确回退，因此相对 matched LLM 的增益为 0，而不是把负迁移
隐藏成正结果。

逐轮部分新增了达到 matched LLM 最终质量和命中全局 top-10 所节省的 target
acquisition rounds，并列出第 1、3、5 轮和最终轮的 best-so-far delta。这样可以
直接讨论“少做多少轮实验”，不再只看 final best。baseline inventory 也补齐了
random、public incumbent、GP-UCB、mixed-kernel GP-EI、target portfolio、LLM
direct prior、LLAMBO-style warm start、matched target-only LLM、matched random
rule 和 CARE source-outcome router，并明确这些历史控制并非全部在七条路径上
逐一重跑。

同时新增 `LLM_RULE_FORMULAS.md`，把 LLM 输出的 kernel patch、role weight、
source-neighbor/additive/interaction prior、target LOO calibration、expert softmax、
transfer mass 和在线 gate 写成与代码一致的公式。图、CSV、公式和校验哈希归档在
`experiments/care_replay/results/2026-07-28-evaluation-completion/`，生成脚本为
`experiments/care_replay/scripts/build_20260728_evaluation_completion.py`。

## 2026-07-27：材料 skill router 独立复核

继续用新的 seed 区间复核 dielectric → experimental band gap。全候选 router
在 calibration 阶段比较了冻结 record 中的 8 个材料 skills 与 GP-UCB、mixed-kernel
GP-EI 和 target acquisition portfolio；使用 10 个 calibration seeds
（86000-86009）和 30 个 held-out seeds（87000-87029）。最终选择了
target acquisition portfolio，所有 LLM skills 都被拒绝，held-out 阶段精确
fallback，没有部署 LLM transfer。

另外对 `counter_transition_metal` 做了单 skill 控制（84000-84009 / 85000-85029）。
它在 held-out 上相对 target portfolio 的均值差为 Final best `+2.3375`、AUC
`+0.9738`、top-10 `+0.0333`，但三个区间都跨 0，不能算确认增益。这个结果说明
材料方向目前更需要增加有效的材料表征或更强的 target-calibrated descriptor，
而不是继续放宽 gate。完整 metrics、selection summary 和 audit trace 在
`experiments/care_replay/results/2026-07-27-materials-replication/`。

为排除 10-seed calibration 过小，又用 30 个 calibration seeds 和 50 个 held-out
seeds 做了全候选 router 检查（88000-88029 / 89000-89049）。这次 calibration
选择了 `counter_transition_metal`；held-out 相对 mixed-kernel GP-EI 的 Final best
为 `+5.11`（95% CI `[-3.78, +14.00]`），AUC `+3.42`
（`[-2.65, +9.49]`），top-10 `+0.08`（`[-0.05, +0.21]`）。点估计为正但
区间仍跨 0，因此这轮只能作为候选正向信号，不能替代已有 500-seed 材料主结果。
归档在同一目录的 `router_30x50/`。

随后把同一冻结 record 和 router 扩到 30 个 calibration seeds 与 100 个 held-out
seeds（90000-90029 / 91000-91099）。calibration 再次选择了
`counter_transition_metal`。在 held-out 上，相对最强的 target-only
mixed-kernel GP-EI，Final best 提升 `+6.52`（95% CI `[+0.13,+12.91]`），
top-10 hit 提升 `+0.16`（`[+0.05,+0.25]`），best-so-far AUC 提升 `+2.59`
（`[-1.60,+6.77]`）。这说明材料 pair 的最终最优值和 top-10 发现率已经出现
可重复的正向证据，但 AUC 仍不显著，而且这里只验证了一个 source-target pair，
不能外推成“所有材料任务都提升”。完整 raw metrics、frozen record 和每 seed
trace 在 `experiments/care_replay/results/2026-07-27-materials-replication/router_30x100/`。

## 2026-07-27：BH semantic skill 独立复核

在已有 Suzuki → Buchwald-Hartwig 500-seed frozen confirmation 之外，补做了一轮
新的独立 smoke replication。复核使用原先已经生成的 `gpt-4o-mini` skill record，
冻结 `high_mw_ligand_effect`，不在 replay 期间调用 LLM，并使用完全不重叠的
10 个 calibration seeds（82000-82009）和 30 个 held-out seeds（83000-83029）。
每个 seed 同时跑 GP-UCB、mixed-kernel GP-EI、target acquisition portfolio、
semantic skill、LLM direct prior 和 LLAMBO-style warm start。

在 held-out seeds 上，semantic skill 相对最强 target-only mixed-kernel GP-EI 的
配对增益为：Final best `+2.5801`（95% CI `[+0.4088, +4.7515]`）、best-so-far
AUC `+3.2158`（`[+0.1176, +6.3140]`）、top-10 hit `+0.1667`
（`[+0.0044, +0.3289]`）。这条结果支持之前反应 HTE 正向信号在新 seed 上仍然
出现，但样本量只有 30，不能替代正式 500-seed 结果。

这次复核也把 LLM 贡献的边界暴露得更清楚：calibration 阶段的 strategy router
选择了 LLM direct-prior 路线；held-out 上 CARE router 与 direct-prior baseline
逐 seed 相同。因此当前能说的是“冻结的语义 skill 在这条跨任务路径上超过了
传统强 BO baseline”，不能说“CARE router 又额外超过了 direct-prior LLM”。完整
metrics、selection summary、冻结 record 和 per-round audit trace 在
`experiments/care_replay/results/2026-07-27-bh-replication-smoke/`。

## 2026-07-25：把“LLM 增益”拆成可检验的 zero-shot 证据

最新讨论指出了三个需要正面处理的方法学问题：如果用 target calibration
挑策略，收益可能主要来自 gate；如果 LLM 同时决定 rule weight、ridge 和
acquisition schedule，就无法把收益解释成科学知识；如果没有同执行器的
random skill null，也不能排除“任意规则结构”本身带来的收益。这个批评对当前
实现是成立的。原有 calibration/held-out 结果仍然保留，但不再把它们单独当作
LLM 泛化证据。

为此新增两条审计路径：

1. `run_zero_shot_semantic_transfer.py` 在 target 上不做 calibration、不按 target
   结果挑 skill，直接报告所有冻结的 LLM skill、matched random null、GP-UCB 和
   mixed-kernel GP-EI。每个策略使用同一组 seed，target outcome 只在正常 online
   acquisition 之后揭示。
2. `generate_llm_semantic_skills.py --proposal-mode hypothesis_only` 只让 LLM
   输出机制假设、公开字段条件、方向、置信度和失败条件；编译器固定 rule magnitude、
   ridge、semantic mass 和 acquisition schedule，并拒绝私有字段。这样可以把
   “LLM 提出的科学假设”与“手工调出来的执行参数”分开。

这两条路径目前是新增的验证协议，尚未替换历史主结果。正式结论需要同时看
zero-shot、matched random null 和 strongest target-only acquisition；若 zero-shot
不赢 random null，只能说执行器有效，不能说 LLM 知识有效。若 hypothesis-only
仍然不赢，则下一步应改进假设表示和可验证的 mechanism library，而不是继续放宽
gate 或增加 target calibration。

首轮结果已经跑完：

- Suzuki -> Buchwald-Hartwig 的 30-seed zero-shot 中，冻结的
  `high_mw_ligand_effect` 相对 GP-UCB 的 composite 为 `+8.0112`
  （95% normal CI `[+2.8732, +13.1492]`），相对 mixed-kernel GP-EI 为
  `+7.0149`（CI `[+1.6881, +12.3418]`）。不过 Final Best 的 CI 仍跨 0，逐 seed
  win rate 为 `43.3%`，所以这是一条条件正信号，不是全面胜出。
- Suzuki -> ChemLex 的 10-seed 扩展中，`counter_hypothesis_branching` 和
  `reagent_effectiveness` 的平均 composite 相对 mixed-kernel GP-EI 分别为
  `+22.35` 和 `+19.18`，但 CI 都跨 0。
- dielectric -> experimental-gap 的材料扩展中，`low_mean_atomic_number`
  的平均 composite 为 `+15.28`，但 `high_chalcogenide_effect` 为 `-23.97`
  且 CI 完全低于 0；这同时显示了候选正信号和负迁移。

因此当前最准确的说法是：zero-shot frozen skill 在多个领域出现了候选增益，
反应 pair 的证据最强，但还没有证明 LLM 在大多数 source-target 上稳定超过
strongest target-only baseline。三组归档分别位于
`results/2026-07-25-zero-shot-suzuki-to-bh-30seed/`、
`results/2026-07-25-zero-shot-suzuki-to-chemlex-10seed/` 和
`results/2026-07-25-zero-shot-dielectric-to-expt-gap-10seed/`。

随后用新的 Common Stack key 实际调用 `openai/gpt-4o-mini`，生成了
`hypothesis_only` 的 Suzuki -> BH record，并做了同规格 30-seed replay。最佳
`ligand_mw_influence` 相对 mixed-kernel GP-EI 的 composite 为 `+4.4166`，但
95% CI 是 `[-2.6207, +11.4540]`，win rate 为 `50%`；另外两个 hypothesis
显著为负。这个结果说明 hypothesis-only 协议确实把“LLM 知识”和“LLM 手调
参数”分开了，但当前 prompt 还没有产生稳定优势。生成 record、每 seed trace
和校验哈希位于
`results/2026-07-25-hypothesis-zero-shot-suzuki-to-bh-30seed/`。

## 2026-07-24：跨领域泛化目标的对照补充

为检验 semantic skill 的收益是否只是“规则特征 + target 在线拟合”的机制红利，新增了
matched random-rule null control。该 null 保留每个 LLM skill 的字段集合、规则数量、
条件阶数、ridge、semantic mass 和 acquisition schedule，只随机化条件取值和规则方向；
随后用同一套 calibration/held-out 协议，并在 calibration 上选择最佳随机路线。

第一轮 Matbench Phonons 结果使用 10 个 calibration seeds、30 个 held-out seeds、3
个随机重复和 5 个 skill。calibration 选出的最佳随机路线在 held-out 上相对 GP-UCB：

| 指标 | Delta |
| --- | ---: |
| Final best | -1.3163 |
| Best-so-far AUC | -0.6295 |
| Final best + AUC | -1.9458 |
| Final best win rate | 36.7% |

这轮结果不能单独证明 LLM 语义知识已经具有因果优势，但说明 matched random rule 没有
复现正向 held-out 行为。第一轮 ESOL null 已完成：held-out Final Best `+0.4924`、AUC
`+0.2297`，但 win rate 只有 `36.7%`，所以不能把它当成稳定泛化优势；ESOL 的
random-rule + warm-start null 则相对 GP-UCB 为 Final Best `+1.1143`、AUC
`+2.1257`，提示初始化本身可以贡献增益。结果、原始
metrics 和选择摘要分别在
`experiments/care_replay/results/random-rule-null-phonons-40seed/` 和
`experiments/care_replay/results/random-rule-null-esol-40seed/`、
`experiments/care_replay/results/random-rule-null-esol-warmstart-40seed/`。ChemLex null
因 RDKit 候选空间计算成本较高，当前本地批量任务已停止，仍需在服务器上完成；已有真实 ChemLex
source-outcome 结果仍保留在正式冻结归档中。

当前项目目标因此固定为：在冻结协议下，让 CARE 2.0 在多个 source-target 组合上
超过强 target-only baseline，并验证跨领域泛化。已有 source-outcome 冻结套件覆盖 7
条真实路径，新增 null control 用来约束“LLM 知识有效”的因果解释；下一步优先补齐
random warm-start、传统 transfer BO baseline 和跨任务 router 的首轮审计已经补齐；
下一步要在服务器完成 ChemLex null，并扩大 source-target 矩阵，继续验证 transfer
是否能在新任务上超过 strongest target-only baseline。

新增的 ESOL random-rule + warm-start null 在 held-out 上相对 GP-UCB 也有正向结果
（Final Best `+1.1143`，AUC `+2.1257`，近似 95% CI 均高于 0），所以不能把所有
GP-UCB 增益归给 LLM 语义。后续主结论必须放在 matched warm-start、strongest
target-only acquisition 和 source-outcome transfer 的三方比较上。

## 2026-07-24：跨任务路由补充

新增 `scripts/cross_task_router.py`，把跨域迁移的第一步从 pair-specific 手调提取
成 schema-only route proposal。它只读取公开的任务族、decision columns 和字段角色：
反应任务按 substrate/condition/catalyst/solvent 等角色对齐，材料和分子任务按共享
descriptor vocabulary 对齐；没有公开对齐时推荐 target-only。它不读取任何 target
outcome，也不绕过 calibration gate。

正式 7-pair 结果接入这层后，4 条候选 transfer 路径通过 gate 并超过 strongest
target-only BO 与 matched target-only LLM，3 条候选路径被 gate 拒绝并精确回退。
因此目前的强结论是“路由后部署不产生负迁移，且多数预设路径有显著 source-outcome
增益”，而不是“所有 source-target 都能直接迁移”。对应的逐路径 route proposal、
部署结果和随机 null 对照已写入
`experiments/care_replay/results/goal-report-2026-07-24/goal_report.md`。

另外完成了一个连续 FreeSolv 扩展（30 calibration / 50 held-out）：
Lipophilicity → FreeSolv continuous 和 ESOL → FreeSolv continuous 的 raw
transfer route 都显著超过传统 target-only BO，但没有显著超过 matched target-only
LLM，因此两条都被 gate 拒绝。这个结果不能算正向 CARE transfer，却说明当前 gate
确实在区分“超过 classical BO”和“真正超过强 LLM baseline”。完整审计在
`experiments/care_replay/results/2026-07-24-freesolv-continuous/`。

随后对连续 FreeSolv 做了 source-extremes 初始设计和 `bound_v2` CI 边界消融。
两种改法都没有在新 held-out 上稳定超过 matched target-only LLM，因此没有被并入
主策略；完整失败对照和 2,060 份 audit 在
`experiments/care_replay/results/2026-07-24-freesolv-routing-ablation/`。这说明当前
路由器宁可放弃不稳定迁移，也不会靠放宽 gate 制造正例。

此外，用冻结的 `gpt-4o-mini` branching skill 做了独立模型对照：target-only
GPT-4o-mini 在连续 FreeSolv 上优于对应 source-outcome route，gate 同样拒绝了
迁移。结果在
`experiments/care_replay/results/2026-07-24-gpt4o-freesolv-branching/`，说明
“换更强模型”与“source transfer 有增量”是两个需要分别验证的问题。

进一步把 DeepSeek source patch 和 GPT-4o-mini target-only skill 组成 model
portfolio 后，source route 仍未超过 target-only skill，calibration 也因 fold
稳定性不足而拒绝部署。完整对照在
`experiments/care_replay/results/2026-07-24-freesolv-model-portfolio/`。

## 2026-07-24：Transfer Portfolio 扩展验证

在单一 transfer 配置之外，新增了一个冻结的 candidate portfolio。每个
source-target pair 在启动前固定候选集合；候选之间只在 calibration seeds 上竞争，
held-out seeds 只执行 calibration 选出的候选，若没有候选同时超过 matched target-only
LLM 和 strongest target-only BO，就精确回退。当前候选覆盖标准 transfer、保守 transfer
和基于 source outcome 的 positive initial design。

在 ChemLex → Buchwald-Hartwig 的真实反应路径上，先做了 5/10 smoke 验证链路，再做
30 calibration / 50 held-out 的正式扩展。portfolio 在 calibration 上自动选择
`source_positive`，held-out 相对 matched target-only GP-UCB 的结果为：

| 指标 | Delta | 近似 95% CI | Win rate |
| --- | ---: | ---: | ---: |
| Final best | +10.2095 | [+7.2982, +13.1208] | 88% |
| Best-so-far AUC | +9.2076 | [+5.5913, +12.8239] | 68% |

这说明 source outcome 不只是提供一个固定规则，还可以帮助系统在不同 transfer
initial design 之间做校准选择；但这仍是一条反应路径上的 30/50 扩展，不能替代
跨分子、材料和反应的全矩阵验证。完整 summary、metrics、690 份 audit 和校验清单在
`experiments/care_replay/results/2026-07-24-transfer-portfolio-bh/`。

随后补了同 seed 的 warm-start-only 归因对照：固定 `router_max_transfer_mass=0`，
保留 `source_positive` 初始设计。它在 50 个 held-out seed 上逐 seed 复现了
portfolio 的 Final best `99.6191` 和 AUC `89.554`。所以这组 +10.2095 / +9.2076
的增益目前应称为 source-informed initial-design transfer，不能再表述成后续
continuous source-outcome adjustment 的独立增益。对照的 530 份 audit 在
`experiments/care_replay/results/2026-07-24-transfer-bh-warmstart-control/`。

同一 portfolio 在 Matbench expt. gap → dielectric 的材料路径上做了 10/20 扩展。
这次没有候选同时通过 matched target-only LLM 与 strongest target-only BO 的联合
校准门槛，因此 20 个 held-out seed 全部精确回退到 target-only。这个结果没有被
改写成正迁移，完整候选 metrics、260 份 audit 和 SHA256 清单在
`experiments/care_replay/results/2026-07-24-transfer-portfolio-materials/`。

在 MoleculeNet FreeSolv → Lipophilicity 上也做了同规格的 10/20 portfolio 扩展。
standard、conservative 和 source-positive 三个候选都没有通过联合校准门槛；因此
held-out 阶段精确回退到 matched target-only，20 个 seed 相对该 fallback 的
Final best 和 AUC delta 都是 0。这个负对照说明 schema 共享并不自动保证
source-outcome transfer，router 的拒绝是必要的。完整候选 metrics、260 份 audit 和
SHA256 清单在
`experiments/care_replay/results/2026-07-24-transfer-portfolio-freesolv-lipophilicity/`。
需要和此前 50/100-seed 的 FreeSolv → Lipophilicity value-prior 结果区分：两者的
冻结协议、target mode 和 seed 配置不同，旧结果仍是分子方向的主正例，本轮则是对
自动 portfolio 泛化边界的独立检验。

三条新 portfolio 路径的统一汇总在
`experiments/care_replay/results/2026-07-24-transfer-portfolio-report/`。按这套
更严格的 candidate-selection protocol，当前是 1/3 路径实际部署 transfer、2/3
路径精确回退。这里的含义不是“跨领域失败”，而是 selector 只在校准证据足够时放行；
但它也意味着目前还不能声称所有领域都能稳定超过 baseline。后续主线应继续增加
不重叠的 source-target pair，并在相同的 calibration/held-out 规则下追求更多强
baseline 正例，而不是放宽 gate。

另外对反向材料路径 Matbench dielectric → experimental band gap 做了一个归因审计。
`source_extremes` portfolio 在 20 个 held-out seed 上达到 Final best/AUC `100/100`，
看起来非常强；但把 source-outcome transfer mass 固定为 0、只保留同一组
source-informed initial probes 后，结果逐 seed 完全相同。因此这条增益应归为
source-informed warm start，而不是持续 transfer，不能计入正向 transfer 数量。完整
portfolio 与 warm-start control 在
`experiments/care_replay/results/2026-07-24-transfer-materials-reverse-initialization/`。

为了避免把不同层次的迁移混成一个数字，最新增加了证据阶梯汇总
`experiments/care_replay/results/2026-07-24-transfer-evidence-report/`。在 5 条
source-schema semantic 路径的 300-seed held-out 验证中，4/5 条相对 matched
target-only LLM 在 Final best 或 AUC 上显著为正，5/5 条相对 strongest target-only
BO 为正；其中 2 条是每轮持续执行的 direct-prior/semantic route，2 条只改变初始
实验点。另一方面，严格 source-outcome portfolio 的 3 条新路径只有 1 条通过，且
被归因为 warm-start，连续 transfer candidate 是 0 条，另外 2 条精确 fallback。
因此目前可以说 CARE 2.0 已经有跨分子、材料和反应任务的可复现 routed LLM
泛化证据，但不能把所有 source-outcome 增益都说成连续迁移，也不能声称所有
source-target pair 都提升。

## 2026-07-24：完整 Source-Outcome Transfer 冻结验证

这一轮补上了此前最关键的缺口：迁移不再停留在 source identity、schema 和字段名，
而是实际读取固定的 source 实验历史及其 measured outcomes。LLM 负责把两个任务的
公开字段编译成 role map 和可执行 skill；系统再从 source outcomes 中估计邻域
prior、单字段 effect、两两 interaction residual、初始探针和 GP kernel geometry。
target outcome 只有在实验点被 reveal 后才能进入在线校准，隐藏结果不会用于初始
设计、路由或候选打分。

实验预先固定 7 条真实数据路径，覆盖分子性质、材料性质和反应 HTE。每条路径使用
50 个 calibration seeds 选择 source transfer 或 matched target-only LLM，
随后冻结策略，在互不重叠的 100 个 held-out seeds 上确认。总 target budget
统一为 15 个实验点。选择规则同时检查 final best、best-so-far AUC、fold
稳定性、non-loss rate、配对 95% 置信区间，并要求 source route 不只超过 matched
target-only LLM，也要通过 strongest target-only BO 的校准比较。

最终 4/7 条路径实际部署 source-outcome transfer，且配对 composite 95% CI
全部显著为正；另外 3 条由 calibration gate 精确回退，所以部署结果逐 seed
等于 matched target-only LLM：

| Source → target | 部署 | Final best delta | AUC delta | 达到 matched LLM 最终值节省轮数 |
| --- | --- | ---: | ---: | ---: |
| ChemLex → Buchwald-Hartwig | transfer | +9.197 | +11.057 | +2.99 `[+2.05, +3.93]` |
| Dielectric → expt. gap | transfer | +32.699 | +48.028 | +6.00 `[+5.14, +6.86]` |
| Expt. gap → dielectric | transfer | +18.065 | +23.723 | +5.03 `[+4.27, +5.79]` |
| Phonons → dielectric | exact fallback | 0 | 0 | 0 |
| ESOL → Lipophilicity | exact fallback | 0 | 0 | 0 |
| FreeSolv → Lipophilicity | transfer | +2.976 | +3.727 | +2.23 `[+1.37, +3.09]` |
| Lipophilicity → FreeSolv | exact fallback | 0 | 0 | 0 |

需要区分“原始迁移有效”和“部署策略不掉点”。ESOL → Lipophilicity 与
Lipophilicity → FreeSolv 的 raw source route 是显著负迁移；Phonons →
dielectric 的 raw composite 均值略正，但 CI 跨 0。它们没有被改写成正结果，
而是完整保留在报告和 audit 中。当前能够成立的结论是：校准后平台在全部预设路径
上避免了负迁移，并在多数路径上确认了真实 source-outcome 增益；不能说任意两个
数据集之间的原始 prior 都能直接迁移。

LLM 输出在 replay 前冻结，held-out 每个 seed 不再调用 API。它提供的是字段角色、
共享 vocabulary、patch 和 acquisition skill；prior 的数值来自真实 source
outcomes，是否启用由 target calibration 决定。因此这轮验证的是“LLM 编译的
可复用迁移 skill + source outcome learning + 安全回退”，而不是每轮让模型凭
自然语言直接猜下一个实验。

完整结果在
`experiments/care_replay/results/2026-07-24-source-outcome-transfer/`，包括
100-seed metrics、逐轮 reasoning traces、每条路径 300 份完整 audit、47 组开发
筛选结果、轮数分析、模型记录、图和 SHA256 校验。

## 2026-07-23：Source-schema 迁移扩展验证

这一轮把“LLM 能不能赢 BO”和“source task 是否真的带来额外信息”拆开验证。
新增 5 条真实数据 source-target 路径，覆盖 MoleculeNet、Matbench、ChemLex
和 Buchwald–Hartwig。每条路径先在 development seeds 上选择 skill identity，
再冻结 skill，用 50 个新 calibration seeds 选择执行方式，最后在 300 个新
held-out seeds 上确认。所有结果都和 matched target-only LLM router 做配对比较。

5 条路径中有 4 条在 `final_best` 或 AUC 上得到显著 source 增量：

| Source → target | Final best delta | AUC delta | Top-10 hit delta |
| --- | ---: | ---: | ---: |
| ESOL → Lipophilicity | +0.602 `[+0.096, +1.108]` | +1.376 `[+0.841, +1.911]` | -0.050（CI 跨 0） |
| Matbench expt. gap → dielectric | +15.265 `[+13.052, +17.478]` | +13.085 `[+11.892, +14.278]` | +0.203 |
| Matbench phonons → dielectric | +6.789 `[+4.799, +8.779]` | +4.416 `[+3.013, +5.819]` | -0.213 |
| ChemLex → Buchwald–Hartwig | +3.971 `[+2.777, +5.164]` | +4.398 `[+3.277, +5.519]` | +0.193 |

Lipophilicity → FreeSolv 是明确负迁移；target-only LLM 在 FreeSolv 上明显更强。
Phonons → dielectric 则是 tradeoff：最终质量和 AUC 提升，但极值命中率下降。
因此主结论是 4/5 路径在 primary metrics 上确认了 source-schema 增量，不是
“所有指标、所有路径都提升”。

和各自最强 BO 比，5 条路径在第 5 轮的 best-so-far 都显著提高；达到 BO
最终值的严格轮数指标在 3/5 条路径上显著节省。Expt. gap → dielectric 平均
节省 3.887 轮，Lipophilicity → FreeSolv 节省 1.003 轮，Phonons →
dielectric 节省 0.650 轮。ChemLex → Buchwald–Hartwig 没有显著缩短这个
严格阈值，但平均提前 1.310 轮命中全局 top-10。

source-schema router 同时在 5/5 条路径上显著优于最强 target-only BO；
target-only LLM 自身在 4 个唯一 target 中的 3 个上优于 BO。这里迁移的是
source identity、任务目标和公开字段映射，不包含 source labels，所以更准确地
说是 semantic/schema transfer，还不是 source outcome transfer。

完整结果、9 次真实 LLM 调用、每 seed raw metrics、2,700 份 held-out router
trace、日志和可复现图在
`experiments/care_replay/results/2026-07-23-source-evidence-extension/`。

## 2026-07-22：多领域 LLM Strategy Router 冻结结果

这一轮解决了一个之前没有拆开的变量：同一份 LLM skill 应该怎样执行。系统现在
不再固定使用 target-calibrated semantic model，而是在 calibration seeds 上
自动比较 target-only BO、semantic calibration、LLM direct prior 和
LLAMBO-style warm-start；通过稳定性门槛后冻结策略，否则回退到 target-only
optimizer。held-out 结果不参与选择。

汇总中的 9 个预设 target 条件里，6 个得到至少一项统计显著正增益，3 个自动
回退。确认正结果覆盖三个领域：

| Target | Router 选择 | Seeds | 强 baseline | Final best delta | AUC delta |
| --- | --- | ---: | --- | ---: | ---: |
| Matbench band gap | calibrated semantic | 500 | target portfolio | +7.0090 | +6.1666 |
| Matbench phonons | LLM warm-start | 500 | target portfolio | +24.4382 | +36.2661 |
| FreeSolv | calibrated semantic | 500 | target portfolio | +3.3023 | +0.8524（CI 跨 0） |
| Lipophilicity | calibrated semantic | 500 | GP-UCB | +1.5660 | +1.2607 |
| ESOL | calibrated semantic | 500 | GP-UCB | +0.4737 | +1.0567 |
| ChemLex Acid-Amine | LLM direct prior | 200 | GP-UCB | +3.1828 | +2.9214 |

ChemLex 使用了真实 updated wetlab record，并把原来的短字符串统计替换为
RDKit reaction representation：酸和胺各自的 12 个归一化数值描述符、官能团
类别和 ring-system 类别。forced semantic execution 在 ChemLex 上是负的，
但 calibration router 正确选择了 direct prior；这说明 LLM 规则本身有用，
问题在执行方式，不需要用 held-out 结果手动挑策略。

Phonons 平均提前 8.534 轮命中全局 top-10，ChemLex 提前 0.325 轮；ESOL
没有显著缩短首次 top-10 命中时间，但第 5 轮 best-so-far 提升 +0.8574。
Suzuki/source-schema -> Buchwald-Hartwig、source-schema -> ChemLex 和
Matbench log bulk modulus 没有得到可靠 LLM 增益，router 保留了 target-only
fallback。

边界也比之前清楚：FreeSolv 是 source-schema transfer；其余正结果主要证明
LLM target-schema generalization。LLM direct prior 和 LLAMBO-style
warm-start 是在同一 replay 协议下的适配版，不是外部系统完整复现。当前
CARE 2.0 的系统优势是 calibration routing、自动 fallback、统一 held-out
协议、逐轮 trace 和知识库沉淀，而不是声称一个固定 CARE acquisition rule
全面击败所有外部方法。

完整结果在
`experiments/care_replay/results/2026-07-22-multidomain-llm-completion/`，
包括 7 次真实 LLM 调用、每 seed metrics、压缩 audit logs、轮数分析、图和
知识库 ingest 所需文件。

## 2026-07-21：LLM 证据来源和实验轮数审计

这一轮把过去混在一起的两个问题拆开了：一是收益到底来自 source task，还是 LLM 只看 target schema 也能写出有用的 skill；二是 LLM 加进来以后，是否能更早找到高价值候选，而不只是最终分数略高。

实验为每个 source-target pair 独立生成三套 skill：`full` 包含源任务结果和统计量，`source_schema_only` 只保留源任务身份、目标和字段映射，`target_only` 完全不告诉 LLM 源任务。三套 skill 使用相同的 calibration/held-out seeds。开发阶段用 30+100 seeds，选中的 skill 冻结后再用 50+500 个新 seeds 确认。LLM 只负责把公开 schema 编译成规则特征和 acquisition schedule；每轮规则系数仍只用已经 reveal 的 target observation 在线拟合，校准不通过就退回 target-only optimizer。

目前有三条 500-seed 结果：

| Target | 结论 | 强 baseline | Final best delta | AUC delta | Top-10 hit delta |
| --- | --- | --- | ---: | ---: | ---: |
| Matbench band gap | target-only LLM 规则划分有效；不使用 LLM 给的系数方向 | target portfolio | +7.0090 | +6.1666 | +0.170 |
| FreeSolv | ESOL 的任务身份和字段映射产生了可确认的 source-schema transfer | target portfolio | +3.3023 | +0.8524（CI 跨 0） | +0.138 |
| Lipophilicity | target-only LLM 规则划分有效；不使用 LLM 给的系数方向 | GP-UCB | +1.5660 | +1.2607 | +0.068 |

Band gap 和 Lipophilicity 都是在 `target_only` 条件下成立的，因此它们证明的是 LLM 能根据陌生 target 的公开 schema 提出可用表征，不能算 source-to-target transfer。FreeSolv 的正结果来自 `source_schema_only`：LLM 知道 source 是 ESOL，并看到 rotatable bond、ring、SMILES length 等字段如何映射到 FreeSolv，但没有看到任何 source outcome。这是目前较干净的一条迁移信号。反过来，源任务 outcome statistics 在四组对照里都没有带来额外收益；Suzuki -> Buchwald-Hartwig 三个条件全部被校准拒绝。

组件 ablation 的结论也很一致。材料和 Lipophilicity 上，把 LLM 给出的初始正负权重清零以后结果更强；但如果把语义规则整个拿掉、只保留 acquisition schedule，收益会明显下降。也就是说，当前 LLM 最有价值的作用不是直接决定方向和权重，而是提出一个可以执行的特征划分，再由 target 数据在线学习系数。

实验轮数方面，材料任务平均提前 1.354 轮命中全局 top-10，FreeSolv 提前 0.478 轮，置信区间均为正。Lipophilicity 没有显著提前首次 top-10 命中，但第 5 轮的 best-so-far 已经比 GP-UCB 高 1.0783。需要保留一个限制：如果阈值定义成“达到同一个 seed 下 baseline 最终找到的值”，三组都没有节省轮数。因此现在可以说 LLM 在部分任务上更早进入高价值区域，不能说它对所有质量阈值都降低了 sample complexity。

完整结果、模型调用、每个 seed 的 metrics、round-efficiency JSON 和压缩 audit log 在 `experiments/care_replay/results/2026-07-21-llm-evidence-causality/`。这一版仍不是和外部 AI Scientist 的同协议 leaderboard；可以主张的是，我们已经在材料和分子性质两个不同领域确认了 LLM schema-to-skill 的泛化能力，并在 FreeSolv 上看到 source-schema transfer，但还没有证明 source outcome transfer 或全领域稳定优势。

## 1. 这轮实验想验证什么

这轮实验不是在复现 CARE 1.0 论文的最终数字。我们做的是 CARE 2.0 的 replay harness 和跨领域 transfer 验证：先把不同领域的数据统一成有限候选池搜索，再看源领域沉淀下来的 skill / prior 能不能在目标领域带来真实收益。

目前结论比最初更清楚：平台接口已经跑通，而且 transfer 不是只有概念验证。最新 paired bootstrap 审计显示，分子性质任务 `FreeSolv -> Lipophilicity` 和反应 HTE 任务 `Suzuki-Miyaura -> Buchwald-Hartwig` 都已经有统计意义上的正向 transfer gain。前者相对 public incumbent 的 final best 提升 +2.7550，95% bootstrap CI 是 [+1.4375, +4.0025]；AUC 提升 +2.4000，CI 是 [+1.3405, +3.4733]。后者 final best 提升 +2.4389，CI 是 [+0.8078, +4.3094]；AUC 提升 +1.1067，CI 是 [+0.1809, +2.2873]。这两条是目前最适合对外讲的“transfer 确实有效”的主结果。

但也要把边界讲清楚：和更强的 target-only GP-UCB 比，当前 transfer 还没有形成统计显著优势。Suzuki -> BH 的 transfer-weighted GP kernel 相对 GP-UCB final best 均值是 +0.3483，但 CI 跨 0；FreeSolv -> Lipophilicity 的 hybrid value-prior 在 5-round low-budget 下 final best 均值是 +0.1462，CI 也跨 0。也就是说，现在已经能证明 CARE 2.0 transfer 在真实任务上有显著正例，但还不能说它稳稳打败所有强优化 baseline。

随后补做的真实 LLM follow-up 说明，LLM proposer 在共享 descriptor 的分子性质方向能给出小幅正收益；但在反应 HTE transfer 上，当前 LLM proposer 还不如确定性 transfer card，LLM auditor 也偏保守。最新的强模型 follow-up 进一步说明，换成 `openai/gpt-5.5` 会改善 LLM proposer 的 final best 和坏干预率，但仍没有超过 deterministic transfer rule；`deepseek/deepseek-v3.2` 在当前长上下文 tool-call 接口下反而不稳。

最新补充后，最适合作为“明显正向 transfer”展示的是 `FreeSolv -> Lipophilicity` 的 shared descriptor value-prior sweep。100 seeds 下，`transfer_value_prior_gate_v1` 在 3/5/10 个 reveal budget 上都稳定超过 incumbent：final best 分别提升 +1.1662、+0.9725、+0.8550，AUC 分别提升 +0.4184、+0.6707、+0.7009，top-10 hit 分别从 0.07/0.11/0.17 提到 0.13/0.18/0.23。这条结果比反应 descriptor transfer 更干净，因为 source 和 target 共享同一套 SMILES-derived descriptor vocabulary，不是在迁移数据集内部编号。

最新又补了一版 acquisition-level scale ensemble，把多个 transfer-weighted GP-UCB scale 的候选排序做 rank averaging，避免只挑一个手写 scale。50 seeds 下，`Suzuki-Miyaura -> Buchwald-Hartwig` 从 GP-UCB 的 final best 91.1145 / AUC 82.9700 提到 92.1629 / 83.6026，增益是 +1.0484 / +0.6326；`Suzuki-Miyaura -> ChemLex` 从 87.7717 / 79.8070 提到 90.2678 / 80.6853，增益是 +2.4961 / +0.8783。这说明 transfer 已经不只是赢 public incumbent，也能在部分真实任务上叠到 strong acquisition baseline 之上。`ChemLex -> Buchwald-Hartwig` 仍是负迁移，这个边界需要诚实保留。

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

按照这个判断，我们又把 LLM 从“候选级打分器”上移成“skill/rule optimizer”。新模式不是让 LLM 直接给候选加分，而是每个 seed 调一次 LLM，让它提出可执行的 transfer rule patch；代码再用 target 已 reveal 的 evidence 和 gate 去执行。第一版只调单字段 role weight，结果仍然不如 deterministic transfer。第二版让 LLM 提出 `ligand-base`、`ligand-aryl_halide` 这类可迁移 role interaction，探索性变强但 bad interventions 偏多。guarded 版本要求 interaction signal 必须和单字段 role-transfer signal 同方向，才允许进入候选分数，已经在 Suzuki -> Buchwald-Hartwig 的 10-seed 检查里首次超过固定 `transfer_gate_v1`。最新又补了 risk-control：`llm_rule_patch_guarded_confirmed_interaction_gate_v1` 把 interaction signal 减半，并在早期之后要求 pair evidence 至少出现 2 次。这个版本目前更适合作为主结果：final best 90.6604 vs 90.0980，AUC 84.2049 vs 82.9136，bad interventions 1.4 vs 1.1。更激进的 damped 版本 final/AUC 更高，91.0730 / 84.4327，但 bad interventions 到 2.2，所以只能当上限诊断，不适合作为主推结果。

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

## 2.7 最新补充：显著性审计和下一步方向

我们这轮又补了一个 paired seed-level significance audit。这个审计不是重新跑更大的实验，而是把已有主结果按同一个 seed 做成 transfer minus baseline 的配对差值，然后用 bootstrap 给 95% CI。它的价值是把“均值看起来更高”和“可以比较有把握地说有正向 transfer”区分开。

结论分三层：

1. 已经显著的 transfer：`FreeSolv -> Lipophilicity` 的 shared descriptor value prior，以及 `Suzuki-Miyaura -> Buchwald-Hartwig` 的 role-level transfer。两者的 final best、AUC、top-10 hit 的 CI 都在 0 以上。这是目前最能支撑 CARE 2.0 跨领域迁移的核心证据。
2. 还不显著但值得继续推的方向：把 transfer 接到 GP-UCB 这类强 target-only optimizer 上。现在 reaction transfer-weighted GP kernel 和 molecule hybrid value prior 的均值都是正的，但 CI 跨 0。说明方向有信号，但还没有到“强 baseline 上稳定胜出”的程度。
3. 风险控制的 tradeoff：新跑的 `hybrid_value_prior_gp_ucb_target_calibrated_gate_v1` 把 MoleculeNet low-budget hybrid 的 bad interventions 从 1.40 降到 0.51，但 final/AUC 增益也变小了。它说明安全 gate 可以降风险，但如果太保守，会把 transfer 的探索收益一起压掉。

这对下一步的启发很直接：如果目标是看到更明显的 transfer 优势，不能只继续收紧 gate。更有价值的路线是让 source skill 进入 acquisition 本身，例如 kernel field weights、descriptor whitelist、early budget allocation、exploration/exploitation schedule；然后用 target calibration 控制方向，而不是完全压低幅度。LLM 也应该参与这些 rule-level 选择，而不是只做候选级加减分。

基于这个判断，代码里又新增并实测了 `llm_rule_patch_prompt_optimized_confirmed_gate_v1`。它仍然走 rule-patch 路线，不让 LLM 直接选 candidate；prompt 明确把目标写成“超过 fixed `transfer_gate_v1` 的 final best / AUC，同时控制 bad interventions”。真实 CommonStack `openai/gpt-5.5` 10-seed 结果说明，结构化调用本身是稳定的，parse error 为 0；但更激进的 prompt 会让 LLM 过度增加 interaction 和 intervention，final best 反而下降。后来加了 risk cap，把 signal cap 限到 0.10、最多一个 interaction、负向信号默认 downweight，bad interventions 从 2.30 降到 1.30，AUC 高于 fixed transfer，但 final best 仍未超过 fixed transfer。当前最强 LLM 版本仍然是 guarded confirmed interaction patch：risk-capped rerun 中 final best 91.1571 / AUC 84.1105，对 fixed `transfer_gate_v1` 的 90.0980 / 82.9136。

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

## 8.1 最新补充：coverage portfolio 和 risk-aware selector

为了回答“能不能让大部分数据集都有 transfer 提升”，我们又补了一轮 coverage 实验，把 target-only surrogate baseline、普通 transfer gate、shared value prior、transfer-weighted GP kernel 和 hybrid GP-UCB transfer 放到同一个 portfolio 里比较。新结果在 `experiments/care_replay/results/2026-07-12-transfer-coverage-portfolio/`。

这轮覆盖 11 个 source-target pair，包含 reaction HTE、ChemLex-style acid-amine、MoleculeNet 分子性质和一个材料 target stress test。每个 pair 用 50 seeds；portfolio 用 seeds 0-24 做 calibration，seeds 25-49 做 held-out evaluation。默认 risk-aware selector 只有在 calibration 上同时超过 fallback `+1.0 final best` 和 `+1.0 AUC` 时才允许 transfer，否则退回 target-only baseline。

结果分三层看：

1. 如果只问“best transfer 是否比 public incumbent 好”，11 个 pair 里有 7 个 final best 为正。
2. 如果问“best transfer 是否比 GP-UCB 好”，11 个 pair 里有 4 个 final best 为正。
3. 如果问“best transfer 是否比最强 target-only baseline 好”，11 个 pair 里有 3 个 final best 为正。

最清楚的 strong-baseline 正例是两条。`FreeSolv -> Lipophilicity` 的 `transfer_value_prior_gate_v1` 相比 GP-UCB final best +1.5850、AUC +1.5082；`Suzuki -> ChemLex` 的 `transfer_weighted_gp_ucb_scale_1` 相比 GP-UCB final best +3.2395、AUC +1.7773。`Suzuki -> Buchwald-Hartwig` 的 weighted-kernel transfer 仍是小正向，final best +0.3001、AUC +0.3162；`BH -> Suzuki` 的 hybrid no-gate final best +0.2847，但 AUC -0.0325，所以只能当诊断，不适合作为主结果。

更重要的是 selector 结论。transfer-only calibration selector 在 held-out 上相对 public incumbent 是 7/11 正向，但相对 GP-UCB 只有 2/11 正向。risk-aware selector 更保守，只在 2 个 pair 上选择 transfer，而这 2 个 pair 在 held-out 上都超过 GP-UCB。这个结果说明 CARE 2.0 不应该包装成“所有 source knowledge 都有用”，而应该强调平台逻辑：source knowledge 先变成 reusable skill candidate，再经过 target calibration / held-out replay / risk gate，只有证据足够时才进入 acquisition。

## 8.2 最新补充：scale ensemble 让 acquisition transfer 更稳

为了继续回应“transfer 能不能在更多数据集上超过 strong baseline”，我们又补了一版 `transfer_weighted_gp_ucb_scale_ensemble_*`。它不是让 LLM 或规则直接选候选，而是把 source transfer card 转成多组 public categorical-kernel weights，再分别跑 GP-UCB acquisition，最后对每组 scale 的候选分数做 normalized rank averaging。直观上，它相当于让多个 transfer 强度投票，减少单个 scale 手调带来的偶然性。

这版结果放在 `experiments/care_replay/results/2026-07-14-scale-ensemble-transfer/`，并重新生成了 portfolio：`experiments/care_replay/results/2026-07-14-transfer-coverage-with-ensembles/`。

三条 50-seed 结果是：

1. `Suzuki-Miyaura -> Buchwald-Hartwig`：GP-UCB 是 final best 91.1145 / AUC 82.9700；scale ensemble 是 92.1629 / 83.6026，提升 +1.0484 / +0.6326。这个结果比之前单 scale 的 +0.3001 / +0.3162 更像一个可讲的 acquisition transfer 正例。
2. `Suzuki-Miyaura -> ChemLex`：GP-UCB 是 87.7717 / 79.8070；scale ensemble 是 90.2678 / 80.6853，提升 +2.4961 / +0.8783。单 scale `scale=1` 的 full-mean 仍然更高，但 ensemble 是一个更稳的可执行策略。
3. `ChemLex -> Buchwald-Hartwig`：scale ensemble 比 GP-UCB 低 -1.8382 final best / -1.3454 AUC。这说明不是所有 ChemLex 形态的 knowledge 都能迁移回 Buchwald-Hartwig，risk gate 仍然必要。

加入 ensemble 后，portfolio 里 policy 数从 134 增到 137。calibration selector 在 held-out 上相对 GP-UCB 为正的 pair 从 3/11 增到 4/11；transfer-only selector 相对 GP-UCB 从 2/11 增到 3/11；相对 best target-only baseline 从 1/11 增到 2/11。严格 risk-aware selector 仍然只放行 2 个 pair，这个保守性目前是合理的，因为负迁移还存在。

这版对 CARE 2.0 的意义是：transfer 不应该只做 additive score patch，也不应该只靠一个固定规则；更自然的形式是“源领域沉淀 skill -> 目标领域校准 -> 进入 acquisition geometry -> selector 决定是否启用”。scale ensemble 是朝这个方向迈的一步。

## 8.3 按最新设计建议补做：hypothesis-only zero-shot transfer audit

前面很多结果使用了 target calibration，适合回答“在 replay 中能否选择一个更好的 route”，但不能直接包装成 zero-shot transfer。根据最新讨论，这一轮把协议收紧：LLM 只输出可证伪的 claim、mechanism、公开字段条件、方向和 failure conditions；编译器固定 rule weight、ridge、prior scale 和 acquisition schedule。target replay 之前不看 target outcome，也不根据 target outcome 选 hypothesis。每个 hypothesis 都完整报告，并加入同样规则数、条件阶数、executor 和 seed schedule 的 matched-random null。

本轮使用真实 LLM `openai/gpt-4o-mini` 生成 hypothesis-only record，之后冻结 record 再回放。结果目录是 `experiments/care_replay/results/2026-07-25-zero-shot-hypothesis-transfer-matrix/`，完整矩阵有 5 个 source-target pair、23 个冻结 hypotheses。每个 pair 的 seed 数和在线预算见下表：

| Source -> Target | Seeds | Hypotheses | 稳定正向 hypothesis | 稳定负向 hypothesis |
| --- | ---: | ---: | ---: | ---: |
| Suzuki-Miyaura -> Buchwald-Hartwig | 30 | 4 | 0 | 2 |
| Suzuki-Miyaura -> ChemLex acid-amine | 10 | 5 | 0 | 0 |
| Matbench Expt Gap -> Matbench Dielectric | 10 | 5 | 0 | 1 |
| MoleculeNet ESOL -> FreeSolv | 10 | 4 | 2 | 0 |
| FreeSolv -> Lipophilicity | 10 | 5 | 0 | 0 |

“稳定正向”要求至少一个 primary metric 相对 mixed-kernel GP-EI 的 normal 95% CI 完全高于 0；不是看某一个 seed，也不是只挑平均值最大的规则。唯一出现 pair-level 稳定正向的是 ESOL -> FreeSolv，其中 `hydrogen_bonding_effect` 的 composite delta 为 +14.9492，95% CI 为 [+1.1688, +28.7295]，`ring_structure_influence` 的 composite delta 为 +12.6694，95% CI 为 [+1.7793, +23.5595]。反应、材料和 FreeSolv -> Lipophilicity 在这套严格 zero-shot 协议下没有稳定正向，BH 和材料各有稳定负向 hypothesis。这些负例也被保留，不能用“最优 hypothesis”掩盖。

这轮的结论比之前更窄但更可信：LLM hypothesis 有跨域泛化的可行性，但目前还不能说“多数数据集都有提升”，更不能说 LLM 已经整体超过强 baseline。当前可复用的平台能力是：LLM 提出结构化科学假说，compiler 固定执行参数，replay harness 做无泄漏验证，matched-random null 判断收益是否只是规则结构造成，knowledge base 保存 claim、mechanism、failure conditions 和结果边界。

对应的知识库卡片在 `knowledge_base/generated_cards/2026-07-25-zero-shot-hypothesis-transfer.json`。其中 mechanism claim 使用 `candidate` 或 `needs_verification` 状态；只有在多 seed、明确 baseline 和边界条件下才记录为 experiment result，避免把一次 replay 的收益直接沉淀成科学事实。

## 9. 现在能得出的结论

第一，代码和实验框架已经从单一 synthetic task 扩到了多个数据集，包括真实 HTE 和真实分子性质数据。这说明 CARE 2.0 的 platform interface 是可行的。

第二，现在已经有真实数据 transfer 正例，但强度要分开讲。分子性质方向，`FreeSolv -> Lipophilicity` 在 50 seeds 下 final best 提升 +2.7550，AUC 提升 +2.4000，top-10 hit 从 4% 到 30%，这是 public-incumbent 设置下最亮眼的 transfer 上限。进一步把 transfer 叠到更强的 GP-UCB 上，100-seed 结果不再是大幅 final-best 碾压，但仍有 early-discovery 增益：10 轮预算 top-10 hit 从 0.11 到 0.21，5 轮和 3 轮低预算下 final best / AUC 都稳定为正。transfer-weighted GP kernel 的 `scale=1.5` 也有小幅正收益，final best +0.0875、AUC +0.2230。反应 HTE 方向，`Suzuki-Miyaura -> Buchwald-Hartwig` 相比 public incumbent 有提升，final best +2.4389，AUC +1.1066。GP-UCB target-only baseline 更强以后，简单 additive hybrid 还没赢 final best；但最新 scale ensemble 已经把 GP-UCB 从 final best 91.1145 / AUC 82.9700 提到 92.1629 / 83.6026。这说明反应方向不是只能赢弱 incumbent，skill 进入 acquisition geometry 后已经能在部分 pair 上超过强 baseline。

第三，结果还不是“所有方向都提升”。BH -> Suzuki 这类反向迁移目前不稳定，ChemLex 和材料方向还需要更强的真实数据与更明确的 transfer map。这个边界反而是有价值的：CARE 2.0 不是盲目把 source knowledge 往 target 上套，而是要识别什么时候能迁移，什么时候应该保守。

## 10. 下一步建议

接下来建议按四个优先级推进。

第一，继续补真实数据。真实 ChemLex、Pfizer 零膨胀数据和材料方向 Matbench / Materials Project 仍然重要。现在 FreeSolv -> Lipophilicity 是更强的 transfer 正例；Suzuki -> BH 是正向但需要进一步优化的反应方向结果。下一步要看这些 transfer 机制能不能继续扩到 ChemLex 和材料 property task。

第二，做 gate calibration。当前最强的 value-prior transfer 能打出明显优势，但 bad interventions 也变多。下一版应该保留它的 top10 hit 和 AUC 优势，同时用 target confirmation、risk-aware gate 或 LLM audit 降低坏 intervention。

第三，继续做 acquisition-level skill optimization。现在 challenger 主要是规则化 factor evidence、shared descriptor prior，以及一版真实 LLM proposer。新 baseline 显示 GP-UCB 在反应 HTE 上很强，简单 additive transfer adjustment 不够；最新 transfer-weighted kernel 说明，把 skill 用来改 GP kernel field weights 是可行方向。下一步应该把 `scale`、posterior mean/uncertainty/exploration weight、candidate filtering 和 gate threshold 放进一个 calibration/search loop。LLM 也应该产出更受约束的 structured proposal、rationale 和 skill artifact，再由 gate 审查，而不是让 LLM 直接决定实验。强模型 follow-up 支持这个判断：`gpt-5.5` 能改善 bounded adjustment，但仍没有超过 deterministic rule；更值得做的是让 LLM 搜 rule，而不是只让它调候选分数。

最新 prompt follow-up 进一步确认了这点。我们按群里反馈把 LLM 从“像 AI 审稿一样评论”改成“实验策略 proposer”：prompt 变短，要求只根据 transfer card 和已揭示 target evidence 提规则；parser 强制校验 prefer/penalize 方向；v3 版本还把 LLM weight 变成建议上限，实际权重由 target support、target effect 和 transfer role weight 重新校准，同一候选命中多条 LLM 规则时取平均信号而不是直接累加。结果是 v3 的 LLM proposer 比直接加权版本更稳，AUC 到 80.8906，bad interventions 降到 0.6；但 final best 只有 86.1182，仍低于 deterministic `transfer_gate_v1` 的 91.5394。1200-token 重跑把 parse error 降到 0，但指标没有变好，说明瓶颈不是 JSON 截断，而是 LLM 选择规则本身还不够强。这个结果不应该包装成 LLM 已经赢了，而应该作为下一步 rule evolution / policy selector 的依据。

第四，补跨域任务。分子方向可以从单属性扩到 LogP/QED/SA 多目标；材料方向可以接 Matbench 或 Materials Project 中能转成 finite-pool replay 的 property task。这样就能更贴近“化学、材料、药物多个领域的新物质发现平台”的 CARE 2.0 目标。

## 11. Transfer 可视化和 reasoning trace

最新的冻结 source-outcome 结果增加了一个独立的可视化归档：
`experiments/care_replay/results/2026-07-25-transfer-visualizations/`。

其中三张图分别回答三个问题：

1. `transfer_role_weight_heatmap.png`：不同 source-target pair 对各个
   transferable role 的平均权重如何。`1.00` 表示中性权重；大于 1 表示
   candidate patch 组合倾向于加强该 role，小于 1 表示倾向于减弱。行前的
   `+`、`-`、`~` 分别表示部署的正向迁移、被 gate 拒绝的负向迁移、以及因
   证据不足而回退的迁移。
2. `transfer_matrix.png`：source 为行、target 为列。绿色表示 held-out
   上实际部署且为正的迁移，红色表示原始迁移为负并被拒绝，灰色表示原始
   信号不稳定并精确回退到 target-only。绿色单元格同时标出 Final best 和
   AUC 的 delta。
3. `transfer_graph.png`：跨反应、材料和分子任务的迁移图谱。箭头方向是
   source -> target，边宽表示 composite held-out signal 的幅度，边颜色
   区分正向部署、负向拒绝和不确定回退。

同时补充了三张用于审计和扩展解读的图：

4. `transfer_evidence_forest.png`：逐条路线的 composite delta 和 95% CI，
   可以直接看出哪些路线显著跨过零点，哪些路线仍然不确定。
5. `transfer_metric_profile.png`：把 raw composite signal、部署后的
   Final-best 增益和 Top-10 rounds saved 并排展示，避免只看单一指标。
6. `transfer_domain_coverage.png`：按 Reaction、Materials、Molecular 汇总
   source domain -> target domain 的覆盖情况。空白格表示当前还没有验证，
   不是把没有实验误写成负迁移。

![CARE 2.0 transfer matrix](experiments/care_replay/results/2026-07-25-transfer-visualizations/transfer_matrix.png)

![CARE 2.0 transfer graph](experiments/care_replay/results/2026-07-25-transfer-visualizations/transfer_graph.png)

![CARE 2.0 transfer evidence forest](experiments/care_replay/results/2026-07-25-transfer-visualizations/transfer_evidence_forest.png)

![CARE 2.0 transfer domain coverage](experiments/care_replay/results/2026-07-25-transfer-visualizations/transfer_domain_coverage.png)

这三张图基于 `headline_results.csv`、7 条 raw summary 和
`run_manifest.json` 自动生成，不是手工绘图，因此重新跑脚本可以复现：

```bash
python3 experiments/care_replay/scripts/build_transfer_visualizations.py \
  --source-outcome-root experiments/care_replay/results/2026-07-24-source-outcome-transfer \
  --output-dir experiments/care_replay/results/2026-07-25-transfer-visualizations
```

LLM trace 也正式纳入归档。`reasoning_trace_index.json` 索引了 18 条 LLM
call record，以及 26 份 source-outcome / zero-shot replay trace。每条记录
可以追溯到：prompt metadata、模型输出、解析后的 skill 或 patch、usage、
每个 seed 的候选选择、revealed outcome 和 acquisition diagnostics。这里的
“reasoning trace”指结构化的可审计决策轨迹，不把隐藏的模型思维链当作实验
结果。完整 source-outcome 的代表性 trace 在 `reasoning_traces/`，逐 seed
审计压缩包在 `audit_archives/`，LLM 调用记录在 `model_calls/`。
