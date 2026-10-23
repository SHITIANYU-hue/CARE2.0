# 投稿补充实验：已完成结果与恢复入口

整理日期：2026-10-23（Asia/Macau）。本次提交发布已完成的 BH 对照、Suzuki 验证、蛋白迁移筛选、AUC 重分析及动作／成本审计，保留无效、中断及负结果。运行记录按仓库规范存放于 `artifacts/`，协议、实现、测试和冻结输入作为源码跟踪。本次发布没有新增模型调用或科学实验。

## 结果及其适用范围

| 投稿事项 | 完成情况及结论 | 报告 |
| --- | --- | --- |
| 实验二：BH 随机规则对照 | 试跑、扩大规模及覆盖匹配完成。历史 LLM 库相对所检验随机库有优势，但未证明稳定优于强 GP 基线；不推广到任意规则库或任务。 | [覆盖匹配](BH_COVERAGE_MATCHED_RULE_CONTROLS_V1.md)、[扩大规模](BH_RANDOM_RULE_CONTROLS_SCALE_V1.md) |
| 实验二：BH 标签匿名化 | 10 对生成完成；标签−匿名化 ΔAUC −0.260，95% 区间 [−1.781,+1.365]。未支持稳定的标签优势，也未证明等效或排除训练记忆。 | [匿名化结果](BH_SEMANTIC_ANONYMIZATION_V2_RESULTS.md) |
| 实验二：Suzuki 第二数据集 | 标签相对 R1、R2、匿名化的三项主比较区间均跨零，未复制 BH 优势。随机比较限于 6 个有效标签父库；R1/R2 实际为同一条件分布的两批抽样。 | [Suzuki 结果](SUZUKI_SECOND_DATASET_V1_RESULTS.md) |
| 实验三：迁移接受测试 | 23 路线、13 个依赖组；16 有益、1 有害、6 不确定。组成匹配 G1 拒绝唯一有害路线，但误拒由 3/16 增至 10/16，整体未改善；不能由 0/1 声称普遍安全。这是固定加性算子的同景观迁移回放。 | [实验三结果](EXPERIMENT3_GATE_VALIDATION_V1_RESULTS.md) |
| 核心 AUC 重分析 | 完成 32 项核心记录、254,160 条逻辑指标行。四个既有任务相对各自原对照的收益保留；统一尺度不构成独立复现，也不扩大原结论。 | [AUC 重分析](AUC_REANALYSIS_V1_RESULTS.md) |
| 实际动作、token 与历史预算审计 | 完成指定三套在线协议及近期生成记录的离线核查。模型是否改变 GP 首选取决于控制器；已记录在线用量至少 2,145,725 token，美元账单及部分失败用量仍未知。历史路线按冻结设置共使用 15 次目标观测。 | [动作与成本审计](OFFLINE_ACTION_COST_BUDGET_AUDIT_V1_RESULTS.md) |

BH/Suzuki 的留出种子来自各自同一反应池，蛋白路线按依赖组统计；不能把这些重复当成新增独立数据集。核心重分析仍只包含冻结的 32 项，没有因预算审计而加入额外历史路线。

跨模型工作、专家盲写、新数据先固定后揭晓及湿实验的完整验证不在本批完成声明中。本次提交不包含工作区中仍在进行的跨模型执行代码或运行目录。阶段性计划见 [AUC 后计划](SUBMISSION_NONRSI_ROADMAP_AFTER_AUC_20261021.md)；其中动作／成本审计的待办状态已被上述完成报告取代。

## 归档与恢复

从仓库根目录执行；恢复工具逐文件校验 SHA-256，并拒绝覆盖不同内容。完整目录索引见 [`artifacts/catalog.json`](../../artifacts/catalog.json)。

```bash
# BH：扩大规模、覆盖匹配及匿名化完整结果
uv run --locked python care.py artifacts restore bh-random-rule-controls-scale-v1-complete bh-coverage-matched-rule-controls-v1-complete bh-semantic-anonymization-v2-generation-complete bh-semantic-anonymization-v2-evaluation-complete

# Suzuki：原冻结与中断记录、续跑冻结、完整生成与评估
uv run --locked python care.py artifacts restore suzuki-second-dataset-v1-protocol-frozen suzuki-second-dataset-v1-generation-interrupted suzuki-continuation-amendment2-protocol-frozen suzuki-continuation-amendment2-generation-complete suzuki-continuation-amendment2-evaluation-complete suzuki-continuation-amendment2-execution-audit

# AUC 认证输入、准备和最终结果；动作／成本审计
uv run --locked python care.py artifacts restore auc-reanalysis-v1-inventory auc-reanalysis-v1-preparation auc-reanalysis-v1-complete offline-action-cost-budget-audit-v1-complete

# 实验三：保留原始内层归档，先恢复，再解包到 care_replay
uv run --locked python care.py artifacts restore experiment3-gate-validation-v1-complete
tar -xzf experiments/care_replay/results/2026-10-20-experiment3-gate-validation-v1/complete_audited_run.tar.gz -C experiments/care_replay
tar -xzf experiments/care_replay/outputs/experiment3_execution_v1/execution_freeze_bundle.tar.gz -C experiments/care_replay
```

实验三内层归档保留历史 `care_replay` 相对路径，外层归档不自动展开。仅在干净恢复目录中执行内层解包；已有不同内容时先核对哈希，避免覆盖后续工作。原始 ProteinGym 测量 ZIP 的获取和权限要求沿用协议，未将模型权重或未完成实验作为结果发布。

## 发布核验

按 `CONTRIBUTING.md` 运行主线契约测试、仓库工具测试、合成 smoke，以及本批新增的规则、匿名化、Suzuki、蛋白和归一化测试。新增归档按仓库验证器核对目录与每个文件哈希，并检查 Git 索引布局和文件大小。

Suzuki 的 `test_seed_history` 是首次冻结前的全工作区扫描。执行完成后的工作区已包含沿用冻结种子的续跑配置及后续工作，该扫描会报告历史种子已注册；发布时另用原冻结审计中的 77 个文件及其 SHA-256 重建当时清单验证，不修改冻结测试、种子或科学方案。具体检查记录随 `completed-results-publication-20261023` 归档保存。

部分冻结文件原为 CRLF；`.gitattributes` 对相关源码、配置、锁和报告保留原始字节，避免 Git 换行转换使已发布 SHA-256 失效。历史归档中的快照保持原样。
