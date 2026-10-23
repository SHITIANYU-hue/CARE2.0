# BH 随机规则扩大实验 v1

本轮检查历史冻结 LLM 规则库相对 R1、R2 随机库的优势能否在更多随机库和新起点下保持。它是已知 BH 测量池上的离线回放；新增模型 API 调用为零。原 pilot 配置和结果保留。

## 运行前固定的设计

- 配置：`configs/bh_random_rule_controls_scale_v1.json`；本地冻结记录：`preregistrations/bh_random_rule_controls_scale_v1.lock.json`。冻结时间与配置、源码、数据、完整库及执行别名的 SHA-256 均保留。这是本地协议冻结，不是外部注册。
- LLM 使用原五策略记录；R1 和 R2 各 100 个独立抽样库，每库同样五个候选，即各 500 次策略抽样、600 条原子条件规则。绝不合并成五百选一，也不挑选最佳随机库作为主比较。
- 原有 R1/R2 生成定义、名义权重与执行参数保持不变。字段/值和符号随机化不使用目标标签；不按激活覆盖率或实际表现拒绝抽样。
- 规则随机种子 `2026101802`；全新校准种子 `202610183000–202610183049`（50 个）；留出种子 `202610184000–202610184099`（100 个）。与 pilot 及 smoke 起点不重叠。
- 每条轨迹同样使用五个初始候选和十次新揭晓；所有库和三个目标基线共享相同起点。共 1,008 个逻辑模式、151,200 条逻辑轨迹和 1,512,000 次逻辑揭晓；不是物理实验数量。
- 每库复用现有 canonical selector。校准集选择共同目标基线；风险调整 composite 阈值 0.25、正增益五折比例阈值 0.8、final non-loss 比例至少 0.55、final 与 AUC 均值非负。没有改动 gate。
- 所有校准完成后写选择锁，再启动留出。既冻结 gate 部署模式，也冻结不经 gate 的校准分数赢家；留出不得重选。

## 重复策略与计算

有限模板空间会产生重复抽样。报告同时给出抽样数与不同执行策略数；不将重复抽样声称为独立的新策略。仅对完全相同的有序条件、权重和全部数值执行参数复用回放，忽略 ID 与解释文本。每个原始 mode 的指标行、库成员和字符串 tie-break 均保留。压缩 trace 记录代表策略，manifest 的 `execution_aliases` 可恢复每个逻辑模式的轨迹。没有按结果去重。

数值线程固定为一；特征缓存最多保留四个输入映射并在每个种子结束清理。这些操作只限制重复计算与内存，不改变选点、gate 或测量预算。真实旧 pilot 起点用于检查别名复用的逐轮等价性；新种子不用于实现调试。

## 预先固定的稳定性分析

主终点为 LLM 库的未门控校准赢家减去每个随机库未门控校准赢家的留出 AUC，先对起点和库等权平均，再分别报告 R1、R2。AUC 沿用执行器的十轮 best-so-far 均值，单位为产率百分点。

使用固定种子 `2026101803` 的 10,000 次 crossed bootstrap：独立重抽库和留出起点，但一次重抽的起点在所有库之间共享，保留共享 LLM 与共同起点带来的依赖。两族各取 97.5% percentile 区间（1.25% 与 98.75% 分位），以 Bonferroni 对两项主比较进行近似 familywise 95% 误差控制。区间条件于本次已冻结的校准选择，不包括重新校准和重新生成 LLM 库的不确定性。

“稳定优势”的预定判据：两族主终点区间下界均严格大于零，而且按种子顺序切成五个连续二十起点块后，两族的五块均差全部大于零。块检查是稳定性描述，不是五次独立验证。同时报告逐库均差的分位数、胜/平/负库比例，完整保存负结果。

gate 后部署模式之间的差为次终点，报告探索性、未调整的 95% crossed bootstrap 区间；不将回退造成的零增益与规则本身效果混淆。final-best 与 top-10 作为描述指标。不能由此推断新化学任务上的泛化、独立模型提案的稳定性、无预训练污染或因果化学知识贡献。

## 复现与审计

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
./.venv/Scripts/python.exe experiments/care_replay/scripts/run_matched_random_rule_controls.py --config experiments/care_replay/configs/bh_random_rule_controls_scale_v1.json --protocol-lock experiments/care_replay/preregistrations/bh_random_rule_controls_scale_v1.lock.json --output-dir experiments/care_replay/results/<new-run-id> --workers 6
./.venv/Scripts/python.exe experiments/care_replay/scripts/audit_matched_rule_run.py --repo . --run experiments/care_replay/results/<new-run-id> --output experiments/care_replay/results/<new-run-id>/audit_report.json
./.venv/Scripts/python.exe experiments/care_replay/scripts/analyze_matched_rule_stability.py --run-dir experiments/care_replay/results/<new-run-id> --output-prefix experiments/care_replay/results/<new-run-id>/stability
```

离线防护禁止 socket 连接；本地数据缺失时失败，不触发在线下载。审计从初始候选和逐轮揭晓重建指标，检查完整单元、无重复选点、选择锁、源码和数据哈希；不重新计算全部 acquisition 排名。

## 2026-10-18 完成结果

本轮达到预设的稳定优势标准：LLM 相对两类随机库的平均留出 AUC 均有正增益，两族调整区间的下界均高于零，各自五个预定起点块的均差也全部为正。结论条件于这一份历史 LLM 库、这次冻结校准及同一 BH 池。

运行目录为 `results/2026-10-18-bh-random-rule-controls-scale-v1`。51.7 分钟内完成 151,200 个逻辑评估单元；精确复用后实际执行 25,950 条轨迹、259,500 次新揭晓记录。没有调用模型 API。每库五选一，测量预算、gate 阈值和数值执行参数保持原样。

### 主比较：校准赢家的留出 AUC

差值均为 LLM 减随机库；每个库先按原校准分数确定赢家，随后固定评估。下表区间为每族 97.5% crossed-bootstrap 区间，对两族主比较作 Bonferroni 调整；库胜平负按其 100 个留出起点的平均差统计，不能解释为逐起点胜率。

| 随机对照 | 平均 ΔAUC | 调整区间 | LLM 胜 / 平 / 负的库数 |
| --- | ---: | --- | --- |
| R1 | +2.5469 | [+1.3748, +3.7457] | 94 / 6 / 0 |
| R2 | +3.8606 | [+2.4612, +5.2994] | 99 / 1 / 0 |

R1 的五块均差依次为 +2.1723、+2.3833、+2.2058、+3.2219、+2.7514；R2 为 +4.3461、+4.0402、+2.5189、+4.0960、+4.3016。打平的六个 R1 库和一个 R2 库均选中了与 LLM headline 完全相同的执行策略；这些抽样照常保留。

### gate 后部署与目标基线

50 个新校准起点选择的共同基线为 GP-UCB；原 gate 接受 LLM 的 `high_mw_ligand_effect`、16/100 个 R1 库和 9/100 个 R2 库。LLM 的未门控赢家与部署策略相同。其余随机库回退到共同基线。

| 部署后比较 | 平均 ΔAUC | 探索性、未调整 95% 区间 |
| --- | ---: | --- |
| LLM − R1 | +1.8575 | [+0.3257, +3.4688] |
| LLM − R2 | +2.1355 | [+0.4726, +3.8943] |

LLM 相对本轮 GP-UCB 基线的 ΔAUC 为 +2.1184（历史配对 normal 95% 区间 [+0.2913, +3.9455]）；final-best 均差为 +0.9821，区间 [-0.7637, +2.7278] 仍跨零。由此最直接支持的是搜索过程的 AUC 平均收益。final-best 与 top-10 的其他对照只作描述。pilot 的校准基线是 portfolio，本轮改为 GP-UCB 是原选择器在新校准集上的结果，不能直接把两个版本对各自基线的增益当成同一个对比。

### 抽样空间与解释边界

| 元数据诊断 | R1 | R2 |
| --- | ---: | ---: |
| 独立库抽样 / 策略抽样 | 100 / 500 | 100 / 500 |
| 不同精确执行策略 | 42 | 155 |
| 按公开激活与参数计的不同策略 | 42 | 120 |
| 含任一原 LLM 精确策略的库 | 69 | 19 |

R1 在当前模板结构下只有 42 种不同执行策略。五百次抽样不等于五百条独立新策略。两族与 LLM 合并共有 170 种精确策略、129 种公开激活等价策略。等价性诊断不参与筛选、删样本或结果加权。

覆盖范围仍是解释限制。例如原 additive 规则覆盖 9.10% 的候选，R1 随机后可覆盖 90.90%，本次平均为 53.27%；headline 的绝对权重加权覆盖原为 28.53%，R1/R2 抽样平均为 41.71% / 41.22%。本轮支持冻结 LLM 内容相对这两类结构和名义权重匹配的随机化对照的条件性优势，不能单独证明覆盖匹配后的化学知识贡献、独立 LLM 提案稳定性或新任务泛化。校准集本身也没有被 bootstrap。

### 验证与归档

- 95 项测试及 114 个子测试通过。独立审计重建全部 151,200 个逻辑评估单元和 25,950 条物理轨迹；零重复查询、零指标差异，校准选择与完整汇总重建一致。
- 配置、源码、数据和选择锁哈希一致；110 个源码快照及两个数据文件、原始 LLM 记录的输入快照均保留，输入路径映射见 `input_snapshot_map.json`。
- `stability.json` / `.txt` 保存主次终点、逐库均差、五块结果及区间；`stability_figure.png` / `.pdf` 为结果图；`audit_report.json` 为审计报告。
- 首次统计调用在文件写完后的终端打印阶段遇到 Windows GBK 无法编码 Unicode 减号。设置 `PYTHONIOENCODING=utf-8` 后使用相同冻结脚本重跑到 `stability_verified`；两份 JSON 和 TXT 均逐字节相同。该事件记录于 `analysis_execution_note.json`，没有改变源码、统计设置或回放结果。
- 归档 ID：`bh-random-rule-controls-scale-v1-complete`。恢复命令：`python care.py artifacts restore bh-random-rule-controls-scale-v1-complete`。运行时原始完整性清单另存为 `execution_integrity.json`；最终 `integrity.json` 及归档索引覆盖报告、图和快照。

结果图：`results/2026-10-18-bh-random-rule-controls-scale-v1/stability_figure.png`。
