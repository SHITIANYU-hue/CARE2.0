# BH 随机规则对照

这项离线试跑比较冻结 LLM 规则库与两种结构匹配的随机规则库。它检查 R1/R2 生成、校准选择和留出评估能否完整运行，不作为化学知识贡献的确认性检验。没有新模型调用、网络请求或湿实验。

## 固定输入与两种对照

使用 2026-07-21 原始 `suzuki_to_bh_semantic_record.json` 的五条策略，逐字节保留在 `skill_banks/frozen_records/bh_semantic_headline_20260721.json`。其中 `high_mw_ligand_effect` 是既有 BH headline 对应策略。此次重新校准五策略库，而非原论文预选单策略的原样复现。

| 对照 | 保留 | 随机化 |
| --- | --- | --- |
| R1 | 字段、字段复用、规则数、每规则条件数、权重绝对值、全部执行参数 | 合法条件值及权重符号 |
| R2 | R1 的结构约束和执行参数；字段类别与取值数量 | 整个策略的单射字段映射，再随机条件值及符号 |

R2 按配置中的显式类别分组，允许映射回原字段。两值数值分箱、三值数值分箱、二值布尔字段和三值类别字段分开。单元素组只能自映射。值均匀采样，不按候选频率采样；生成过程中不读取测量结果。R1 保持旧函数的随机序列兼容性。

权重和结构匹配不等于规则激活比例匹配。`public_rule_diagnostics.json` 保留每条规则的覆盖数量、比例与激活候选哈希；没有根据效果删除低覆盖或重复规则。BH 的两个 base family 字段可能定义同一分区，字段名不同不自动等于新的表示。

## 相同选择预算和两阶段评估

`configs/bh_random_rule_controls_pilot_v1.json` 固定：

- 完整 BH 池 3,955 候选；每次 5 个相同初始点加 10 次新测量。
- 原始 LLM 库含 5 条策略；R1、R2 各 4 个随机库，每库也是 5 条策略。
- 校准种子 `202610181000–202610181009`，留出种子 `202610182000–202610182019`。
- 三个共享基线：GP-UCB、mixed-kernel GP-EI、target acquisition portfolio。
- 只使用记录中的策略，不额外展开 no-prior/conservative 变体，不混入改变初始点的 warm-start。
- 所有库调用现有 `run_calibrated_llm_semantic_selector.select_skill`：在校准集选择共同目标基线，按原始风险调整 composite、五折稳定性和 non-loss 条件选策略；不满足条件则回退。

每个随机库独立选择。**不把四个随机库合并为二十选一，也不在看到留出结果后挑选最佳随机库。** 所有 48 个模式都使用相同种子，合计 1,440 条轨迹和 14,400 次新测量查表（不是新增物理实验）。

先完成全部校准，再写入 `selection_lock.json`，最后启动留出回放。所有策略的留出效果都保存用于描述随机分布，但不用于重新选择。部署后效果与不经过 gate 的校准赢家效果分别报告。随机规则仍使用历史执行器的 `llm_semantic_` mode 前缀，真实来源由 `library_family` 标明；`selected_llm_skill` 仅对原始 LLM 库成立。

区间复用原有配对 normal 95% 计算，仅描述这些冻结规则在同一 BH 池的留出起点波动。四个随机库共享模板、种子和基线；不能把它们当作独立科学任务。10 个校准和 20 个留出起点不足以支持正式机制结论。扩大规模、独立模型提案、Suzuki 外测与盲专家对照应使用新协议版本。

## 运行

在仓库根目录，使用已有 Python 环境：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
./.venv/Scripts/python.exe experiments/care_replay/scripts/run_matched_random_rule_controls.py --config experiments/care_replay/configs/bh_random_rule_controls_pilot_v1.json --output-dir experiments/care_replay/results/<new-run-id> --workers 4
```

运行目录必须不存在，防止覆盖冻结记录。数据必须已经在本地；主进程的数据加载和工作进程回放禁止 socket 连接，不会在线下载或生成规则。旧 `run_random_rule_control.py` CLI 保持原 R1 实验行为；新 R1/R2 公平流程使用上述入口。

输出包括输入与源码哈希、初始候选、全部冻结规则、公开覆盖诊断、逐阶段指标、逐轮选点和揭晓记录、校准选择锁、汇总及完整性哈希。失败保留已有记录，不报告不完整配对效果。

## 2026 年 10 月 18 日试跑记录

首轮 `2026-10-18-bh-random-rule-controls-pilot-v1` 因代码审查发现历史 portfolio audit 缺少 `public_observed_count` 而在校准选择前停止。该值现在按初始观测数与轮次补齐；真实 BH 的四模式 smoke 验证通过。重跑使用相同规则配置、种子、预算和选择标准，目录为 `2026-10-18-bh-random-rule-controls-pilot-v1-rerun`。原中止记录保留，不覆盖。

### 完成结果

重跑完成全部 1,440 条轨迹，运行约 221 秒，新增模型调用为 0。独立检查重建了 14,400 次选点、揭晓结果、全部指标、校准选择和留出汇总；无重复选点、缺失单元或指标差异，运行期间源码和输入数据哈希不变。

校准选择的共同基线是 `target_acquisition_portfolio`。下表“规则效果”评估每库在校准阶段按风险调整分数选出的未门控赢家；“部署效果”评估同一 canonical selector 应用接受条件后的策略。两者身份都在留出前冻结。所有区间仅为这 20 个留出起点上的条件性 normal 95% 区间，未做多重比较校正。

| 规则库 | 未门控赢家 ΔAUC | 95% 区间 | 接受规则 | 部署后 ΔAUC |
| --- | ---: | --- | --- | ---: |
| 原始 LLM | +3.343 | [-0.677, +7.363] | 否 | 0.000 |
| R1 001 | +1.178 | [-2.609, +4.965] | 是 | +1.178 |
| R1 002 | -2.658 | [-7.597, +2.280] | 否 | 0.000 |
| R1 003 | -0.014 | [-2.953, +2.925] | 否 | 0.000 |
| R1 004 | +1.747 | [-2.533, +6.027] | 否 | 0.000 |
| R2 001 | -2.718 | [-6.082, +0.647] | 否 | 0.000 |
| R2 002 | -3.942 | [-6.785, -1.098] | 否 | 0.000 |
| R2 003 | -8.102 | [-12.267, -3.936] | 否 | 0.000 |
| R2 004 | -4.129 | [-7.795, -0.463] | 否 | 0.000 |

LLM 的未门控校准赢家仍为 `high_mw_ligand_effect`。本轮其留出均值为正，但区间跨零，且校准接受条件未通过。R1 仅一个库通过 gate，留出区间同样跨零；R2 四个库全部回退。本轮不能据此确认 LLM 优于随机规则，也不能把被拒规则的零部署增益当作规则本身无效。

### 检查和归档

- 针对性测试：56 passed，80 subtests passed（R1/R2、两阶段 runner、原语义执行器和原校准选择器）。
- 审计脚本与报告：运行目录内 `audit_replay.py`、`audit_report.json`；审计核对已记录选择及指标，不重新计算全部 acquisition 排名。
- 主结果：运行目录内 `summary.json`、`selection_lock.json`、两阶段 `*_metrics.csv` 与 `*_traces.jsonl`。
- 归档 ID：`bh-random-rule-controls-pilot-v1-complete`；中止记录 ID：`bh-random-rule-controls-pilot-v1-aborted`。
- 恢复：`python care.py artifacts restore bh-random-rule-controls-pilot-v1-complete`。

后续正式实验应冻结新的配置，增加随机库和校准/留出规模，并预先明确主对比与误差控制。当前配置保留为固定的流程试跑，不根据本轮结果修改其阈值或删去负结果。

2026-10-18 的扩大版已完成：R1、R2 各 100 个五策略库，50 个新校准起点和 100 个新留出起点；gate 保持不变。完整设计、结果与局限见 [扩大实验协议与结果](BH_RANDOM_RULE_CONTROLS_SCALE_V1.md)。
