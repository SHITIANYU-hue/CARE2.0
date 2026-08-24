# CARE 2.0 汇报讲稿

配套文件：`CARE2.0_Nature_evidence_update_2026-08-23.pptx`

## 第 1 页：项目定位

CARE 2.0 研究的是旧实验经验能否帮助新任务更快找到好条件。它不是让大模型一次性猜答案，而是把大模型放进一个连续实验循环：模型先读取 source 任务的证据，提出一个可以被实验推翻的假设；target 每得到一个新结果后，模型重新判断这个假设是否还成立，并决定下一次做哪个实验。我们的核心贡献是把知识迁移变成了一个可执行、可审计、可拒绝的在线决策过程。

## 第 2 页：目前最准确的结果

当前证据覆盖 11 条真实数据迁移路线，包括反应优化、分子性质和材料任务。在线 LLM Scientist 相对同开局、同预算的 target-only GP-UCB，平均 best-so-far AUC 增益为 1.01；6 条路线提高、2 条持平、3 条下降。LLM 在每条路线的 10 个在线轮次都参与最终选择。这里必须同时说明证据边界：按路线 bootstrap 得到的 95% 区间为 -0.44 到 2.78，仍然跨过 0，而且每条路线目前只有一条在线 LLM 轨迹。因此，这组结果说明存在路线特异的正向信号，还不能证明普遍跨领域提升。

## 第 3 页：问题如何定义

实验被定义为一个有限预算的 sequential decision problem。迁移前，系统只知道 source 历史、target 的变量和候选空间以及固定预算；target 中尚未执行条件的结果始终隐藏。每轮只能选择一个候选，执行后才揭示它的真实结果。评价重点不是模型的解释是否流畅，而是在相同起点、候选空间和实验次数下，它是否比 target-only GP 更早找到高值条件。

## 第 4 页：系统每轮做什么

每轮先用 target GP、source prior 和覆盖策略建立一个受约束的候选菜单。Proposer 读取现有证据，更新可证伪假设并选择一个候选；Critic 检查支持证据、反对证据和负迁移风险，可以接受提案，也可以退回 GP rank 1，或改选菜单中的其他候选。随后系统只执行一个实验，揭示结果，更新 GP 后验、当前 best、假设状态和下一轮菜单。LLM 在 10 个轮次都拥有实际选择权，不是只写开场说明。

## 第 5 页：Replay harness

Replay harness 用完整历史数据模拟真实的逐轮实验。虽然磁盘上保存了 target 的全部标签，但控制器在第 t 轮只能读取此前已经选择并揭示的结果；未选择候选不会进入 prompt、GP 训练或候选评分。在线 LLM 和对照组从完全相同的三个 target 观测开始，各有 10 次揭示机会。每轮的输入、候选、模型回答、最终选择和真实结果都写入 trace，所以任何一条结论都能回到原始决策检查。

## 第 6 页：Reward 和系统更新

这里的 reward 不是用来训练 Claude 权重的强化学习奖励，而是衡量搜索效率的指标。第 t 轮的当前最好值记为 `b_t=max(y_1,...,y_t)`，长度为 T 的轨迹得分为 `R=(1/T)Σb_t`。高值条件越早被发现，它会在更多后续轮次中贡献较高的 best，因此 AUC 越大。一个新结果会更新 target 观测历史、GP 后验、候选排序、假设状态和 trace；它不会更新 LLM 参数，也不会自动沉淀成永久 skill。

## 第 7 页：主对照为什么公平

最重要的主对照是 same-initial target-only GP-UCB。它与在线 LLM 使用相同的三个初始观测、候选空间和 10 轮预算，也看不到未执行的 target 标签。唯一差别是后续决策是否使用 source evidence 和 LLM proposer/critic，因此可以隔离在线 LLM 增量。我们又从完全相同的 candidate IDs 重放了 RGPE 和 multitask GP：FreeSolv→Lipophilicity 的在线 LLM 比三种 baseline 中最强的 target GP 高 +6.325 AUC；phonons→bulk modulus 比最强的 RGPE 高 +1.922；Suzuki 则比 multisource ICM-BMA 低 1.960。三条路线相对事后最强 baseline 的等权平均为 +2.096，2/3 路线获胜。这里的“最强”是看完轨迹后选出的压力测试，不是预先 calibration 的主结果；每条路线也只有一次 LLM 调用，所以不能报显著性。下一步仍需在新冻结重复协议里预先规定 baseline 选择规则。

## 第 8 页：LLM 的真实输入和输出

每轮输入包括已揭示的 target 历史、当前 best、source 证据、初始假设、GP 的均值和不确定性、expected improvement 以及合格候选菜单。数据集还显式提供原始物理量、单位、replay score 的变换公式和优化方向，但不会提供未执行候选的数值。模型必须结构化输出假设状态、更新后的假设、一个候选 ID、预期结果、改善概率、支持和反对证据、是否继续迁移，以及它与 GP rank 1 的比较。Critic 再做一次独立检查。prompt、原始回答、模型版本、token 用量、候选选择和揭示结果都完整保存，API 密钥不会进入实验文件。

## 第 9 页：FreeSolv 到 Lipophilicity 案例

这是保存下来的真实 trace。该路线的初始 best 为 67.25。第 0 轮的 GP rank 1 是 `lipo_2256`，但 proposer 和 critic 选择了 GP rank 4 的 `lipo_0635`。模型给出的理由是：该候选的 expected improvement 较高，而且卤素丰富、低极性的结构与已观测的高 lipophilicity 分子一致。执行前系统不知道该候选的真实标签；揭示后得到 86.875，第一轮使 best 提高 19.625。整条路线相对同开局 target-only GP 的 AUC 增益为 6.325。这个案例用于说明 LLM 如何覆盖 GP 建议并留下证据链，不用于代表所有路线。

## 第 10 页：路线级结果

图中的每个点都是一条真实 source-target 路线，横轴是在线 LLM 相对同开局 target-only GP 的 best-so-far AUC 差值。正值表示更早找到高值条件，负值表示迁移拖慢搜索。11 条路线中，材料 `expt gap→dielectric` 为 +6.75，`FreeSolv→Lipophilicity` 为 +6.33；材料 `expt gap→mp gap` 为 -2.45。结果说明迁移效果有明显异质性，负迁移不能被平均值掩盖。

## 第 11 页：配对稳定性审计

早期审计调用和第一次冻结重复的总体胜平负都为 6 / 2 / 3，但 11 条配对路线中有 4 条严格翻转正负号：材料 `expt gap→mp gap`、`aniline→phenethylamine/AlPhos`、`aniline→benzamide/tBuXPhos` 和 `aniline→phenethylamine/tBuBrettPhos`。Suzuki 三次观察为 +0.28、+1.38 和 +0.32，方向一致但幅度仍有波动。早期调用发生在确认协议冻结前，因此这不是正式重复检验，而是说明单次随机 LLM trajectory 不能定义一条路线是否稳定可迁移。论文应把本页写成 stochasticity audit，不应把它包装成显著性结果。

## 第 12 页：材料迁移案例

这页展示的是冻结 JSON/JSONL 生成的真实轨迹。`phonons→bulk modulus` 中，LLM 十轮都把 `continue_source_transfer` 设为 false，因为 target 证据不支持把声子峰的结果排序直接搬到体模量。它仍然利用材料组成语义和已揭示的 target 历史，在 4/10 轮选择非 GP rank 1 候选。CARE 最终找到 normalized score 为 77.6127 的 `O8Pt6`，同开局 GP 最终为 73.2886；AUC 增益为 3.240，final 增益为 4.324。这个结果应解释成“负迁移识别后进行 target 自适应”，不能说成 phonon outcome 对 bulk modulus 的正向直接迁移。图中的分数为 `100×log10(K_VRH[GPa])/3`，本身是无量纲归一化分数，不是 GPa。早期 trace 曾有 5 次把这类 replay score 直接写成 GPa；加入 raw quantity、raw unit、score transformation 和 optimization direction 后，新轨迹的 20 个 proposer/critic 响应中该错误为 0。新轨迹相对同开局 GP 的 AUC 增益为 +3.083、final 增益为 +2.746，但两次调用并非配对实验，所以这里能证明的是语义错误被修复，不能用来证明 prompt 修复提高了性能。

## 第 13 页：结论边界

已经完成的是无泄漏 replay、同预算主对照、三个任务家族、逐轮真实 LLM 决策和完整 trace。context-preserving 协议下的分子、材料和 Suzuki 三条单轨迹相对同开局 GP 均为正，但它们仍是 `n=1` development evidence；其中材料 case 还是拒绝 source transfer 后的 target 自适应。尚未完成的是每条路线足量的独立重复、窄且不跨 0 的确认区间、稳定超过 fixed CARE 与 transfer GP，以及 prospective wet-lab 验证。系统会更新运行状态、GP 和假设，但不会训练 LLM 权重，也没有把 trace 自动蒸馏为永久 skill。

## 第 14 页：下一轮实验

重复实验已经启动：Opus 冻结 v1 完成 11/330，另有三条 context-preserving development 轨迹。GLM 仍没有完整轨迹，因此不能报告跨模型效果。同开局 classical stress test 已完成三条路线：分子和材料路线超过 RGPE/ICM，Suzuki 则由 multisource ICM-BMA 领先。下一步要在看结果前固定 baseline 选择规则，再把在线 LLM、RGPE、multitask GP-ICM 和 target-only GP 放进同一个重复协议。除此之外还需要未参与开发的新任务家族、source/critic/gate 消融，以及至少一条 prospective wet-lab campaign。

## 数据来源

- `experiments/care_replay/results/2026-08-22-submission-evidence-audit/route_evidence.csv`
- `experiments/care_replay/results/2026-08-22-submission-evidence-audit/phase_statistics.csv`
- `experiments/care_replay/results/2026-08-16-opus48-bounded-ei-development-v2/molecular_freesolv_to_lipophilicity/llm_trace.jsonl`
- `experiments/care_replay/results/2026-08-16-opus48-bounded-ei-development-v2/molecular_freesolv_to_lipophilicity/summary.json`
- `experiments/care_replay/results/2026-08-23-online-llm-repeated-confirmation-v1/aggregate/repeated_confirmation.json`
- `experiments/care_replay/results/2026-08-23-online-llm-repeated-confirmation-v1/aggregate/repeated_route_effects.svg`
- `experiments/care_replay/results/2026-08-23-online-llm-repeated-confirmation-v2/reizman_cases_123_to_case4/trajectory_2000/summary.json`
- `experiments/care_replay/configs/online_llm_repeated_confirmation_v2.json`
- `experiments/care_replay/results/2026-08-23-llm-trajectory-stability-audit/paired_route_stability.csv`
- `experiments/care_replay/results/2026-08-23-llm-trajectory-stability-audit/paired_route_stability.svg`
- `experiments/care_replay/configs/online_llm_glm53_robustness_v1.json`
- `experiments/care_replay/configs/online_llm_glm53_robustness_v3.json`
- `experiments/care_replay/configs/classical_transfer_benchmark_v1.json`
- `experiments/care_replay/results/2026-08-08-classical-transfer-confirmation/comparison/care_vs_classical.json`
- `experiments/care_replay/results/2026-08-24-online-llm-transport-context-robustness-v3/aggregate/repeated_confirmation.json`
- `experiments/care_replay/results/2026-08-24-online-llm-transport-context-robustness-v3/materials_phonons_to_bulk_modulus/trajectory_5000/summary.json`
- `experiments/care_replay/results/2026-08-24-online-llm-transport-context-robustness-v3/materials_phonons_to_bulk_modulus/trajectory_5000/llm_trace.jsonl`
- `experiments/care_replay/results/2026-08-24-online-llm-transport-context-robustness-v3/materials_phonons_to_bulk_modulus/trajectory_5000/audit/materials_online_trace_audit.json`
- `experiments/care_replay/configs/online_llm_outcome_semantics_robustness_v1.json`
- `experiments/care_replay/results/2026-08-24-online-llm-outcome-semantics-robustness-v1/materials_phonons_to_bulk_modulus/trajectory_6000/summary.json`
- `experiments/care_replay/results/2026-08-24-online-llm-outcome-semantics-robustness-v1/materials_phonons_to_bulk_modulus/trajectory_6000/audit/outcome_semantics_audit.json`
- `experiments/care_replay/configs/online_llm_matched_classical_audit_v1.json`
- `experiments/care_replay/results/2026-08-24-online-llm-matched-classical-audit-v1/aggregate.json`
- `experiments/care_replay/results/2026-08-24-online-llm-matched-classical-audit-v1/comparisons.csv`
- `docs/GLM53_CROSS_MODEL_PROTOCOL.md`
