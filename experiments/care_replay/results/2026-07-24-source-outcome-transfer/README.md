# 完整 source-outcome transfer：冻结验证

这次实验解决的是一个比 source-schema transfer 更严格的问题：系统是否真的读取了
源任务的实验结果，并把其中的 outcome 结构迁移到目标任务，而不是只告诉 LLM
source/target 的名字和字段。

实现上，LLM 先把公开 schema 编译成字段角色映射和可执行 patch。代码再从固定的
source history 中估计邻域 outcome prior、单字段 effect、两两 interaction residual
和 categorical kernel geometry。目标域开始后，prior 只能用已经 reveal 的 target
observation 做 prequential calibration；hidden target outcome 不参与初始设计、路由
或候选打分。每条路径都在 50 个 calibration seeds 上决定使用迁移还是精确回退，
随后冻结，在全新的 100 个 held-out seeds 上报告。

## 结论

预设的 7 条真实 source-target 路径中，4 条选择 source-outcome transfer，并且
`final_best + best_so_far_auc` 的配对 95% 置信区间全部显著为正。其余 3 条在
calibration 阶段没有通过，部署策略逐 seed 复现 matched target-only LLM，因此
部署后没有负迁移。

| Source → target | 部署 | Final best | AUC | 达到 matched LLM 最终值节省轮数 |
| --- | --- | ---: | ---: | ---: |
| ChemLex → Buchwald-Hartwig | transfer | +9.197 | +11.057 | +2.99 `[+2.05, +3.93]` |
| Dielectric → expt. gap | transfer | +32.699 | +48.028 | +6.00 `[+5.14, +6.86]` |
| Expt. gap → dielectric | transfer | +18.065 | +23.723 | +5.03 `[+4.27, +5.79]` |
| Phonons → dielectric | exact fallback | 0 | 0 | 0 |
| ESOL → Lipophilicity | exact fallback | 0 | 0 | 0 |
| FreeSolv → Lipophilicity | transfer | +2.976 | +3.727 | +2.23 `[+1.37, +3.09]` |
| Lipophilicity → FreeSolv | exact fallback | 0 | 0 | 0 |

这里的“所有迁移有效”指部署策略满足 non-negative transfer：4 条路径实际执行并
显著提升，3 条不适用的路径由校准器拒绝后精确回退。不能把后 3 条描述成原始
source route 有收益。原始 route 中，ESOL → Lipophilicity 和
Lipophilicity → FreeSolv 是显著负迁移；Phonons → dielectric 的均值略正但
置信区间跨 0。这些负例完整保留在 `goal/`、`development/` 和 audit 中。

## LLM 做了什么

LLM 不是逐轮读取隐藏答案，也不是直接替优化器选实验。它的一次性输出定义：

- 哪些 source 字段和 target 字段承担相近的科学角色；
- 哪些字段值允许规范化到共享 vocabulary；
- source prior、interaction 和 kernel skill 的候选强度与执行方式；
- matched target-only LLM 的语义 skill 或 acquisition 方案。

这些输出在 replay 前冻结。真正的 prior 数值由 measured source outcomes 估计；
真正是否采用该 prior，则由已 reveal 的 target evidence 和 calibration gate
决定。held-out replay 每个 seed 不再调用模型，因此实验比较的是可复用 skill，
而不是 API 随机性或按 seed 调 prompt。

## 归档内容

- `goal/goal_verification.json`：7 条路径的配对统计和目标判定。
- `headline_results.csv`：知识库和文档使用的主结果表。
- `raw_summaries/`、`raw_metrics/`：每条路径的原始 summary 和逐 seed metrics。
- `round_efficiency/`：达到 matched LLM 最终质量及命中 top-10 的轮数分析。
- `reasoning_traces/`：每条路径 seed 41000 的原始 route、部署 route 和 baseline trace。
- `audit_archives/`：每条路径 calibration 50 + held-out 100 seeds 的完整 source
  route 与 matched baseline audit，共 300 份 JSONL。
- `development/`：47 个开发期策略筛选的 summary 和 metrics，包括未采用的方向。
- `model_calls/`：冻结的 LLM records；不含 API key。
- `figures/`：高分辨率 PNG 和 PDF。
- `SHA256SUMS`：归档文件完整性校验。

复现实验入口是 `scripts/run_source_outcome_suite.py`，冻结配置在
`configs/source_outcome_benchmark.json`。正式结果只使用 `frozen_source_outcome_v1`
这组 calibration/held-out seed 区间；`development/` 只记录路径和策略筛选过程。
