# CARE 2.0 干实验与湿实验验证清单

这份是给实验室看的版本。我们的建议不是把所有历史轨迹重做一遍，而是先做三组小面板，验证 CARE 2.0 提出的经验到底能不能在真实重复实验中成立。条件表见 `wetlab_conditions.csv`；表中不放历史测量结果或可检索的原始 candidate ID，实验完成后再用内部的 `internal_provenance.csv` 统一揭盲。

## 我们已经做过哪些干实验

| 干实验 | 用到的数据 | 主要问题 | 目前得到的结论 |
| --- | --- | --- | --- |
| Target-only 基线 | Buchwald-Hartwig、Suzuki、ChemLex、Baumgartner、光催化产氢、MoleculeNet、Matbench | 不迁移时，random、GP-UCB、GP-EI、kNN 等方法能做到什么程度 | GP-UCB/GP-EI 是必须正面对比的强基线，不能只和随机方法比 |
| 同反应空间 warm-start | Baumgartner C-N 与 Suzuki 多个 campaign | 已完成实验能否帮助新 campaign 更快找到好条件 | Suzuki MINLP1 到 MINLP2 有明确的搜索效率信号；C-N 路线有正有负 |
| 多 source 迁移与 abstention | Reizman Suzuki cases 1-3 到 case 4 | 多个历史 campaign 是否一定值得迁移 | 未通过开发集 gate 时回退到 target-only，可避免负迁移，但尚未证明多 source 必然增益 |
| LLM hypothesis 初始设计 | Suzuki MINLP2、Morpholine-AlPhos、Morpholine-tBuBrettPhos 等 | LLM 能否把 source 经验写成可证伪假设，并据此选择首批实验 | LLM 的语义锚点有用，但需要和 source 共识点、几何覆盖点一起执行；让 LLM 单独包办常常不稳 |
| 在线 LLM proposer/critic | 分子、材料、C-N、Suzuki 共 11 条路线 | LLM 能否逐轮改变 GP 候选选择 | 11 条路线为 6 胜、2 平、3 负；说明存在增益，但不是普遍提升 |
| Calibration gate | 同一批在线 LLM 轨迹 | LLM 预测明显失准后是否应继续控制实验 | 固定误差 gate 能减少部分负迁移，定位是权限校准，不是制造正结果 |
| 光催化产氢 pilot | 1,109 条公开真实配方 | LLM 化学常识在哪个环节有用 | 静态规则持续控制每一轮会变差；用规则做首批 warm-start、随后交给 GP 更合理 |
| Semantic skill 与随机规则对照 | 反应、材料、分子性质 | 增益来自科学语义，还是任意规则结构 | 两种情况都出现过；因此正式结果必须同时报告 matched random skill 与 strongest target-only baseline |
| 外部 scientific-agent benchmark | AstaBench DiscoveryBench validation | CARE 的审计流程是否比通用 ReAct 更稳定、更省调用 | 五题配对中 CARE 5/5 正常提交，ReAct 1/5；CARE 少用 42.5% token。官方质量评分器不可用，所以不能据此声称答案更正确 |
| LLM 零经验选点消融 | Lipophilicity、Matbench、Baumgartner Suzuki、光催化产氢 | 不给历史实验、RAG、skill 或 GP 分数时，LLM 靠预训练知识能否选对 | 四个数据集相对菜单随机期望均为正，bootstrap 95% 区间均高于 0；这是一次性选点结果，公开数据的预训练暴露风险仍需保留 |

## 建议先做的湿实验

### P1：Morpholine-AlPhos 三点方向性验证

三点保持同一底物、precatalyst 和 base，主要改变温度、停留时间和 base equivalents。预注册判断是：两组 `100 °C + 约 34 min` 条件应整体优于 `47.5 °C + 约 9.6 min` 条件。这个面板最小，只需 3 个条件，每个条件建议 3 次独立重复。

### P1：3-Chloropyridine Suzuki 三点迁移验证

这一组检验从 MINLP1 沉淀的“高温、高 precatalyst fraction”经验能否在 MINLP2 重现。预注册判断是高 fraction 的 B1/B2 整体优于低 fraction 的 B3。三点同时覆盖 XPhos 与 RuPhos，能区分“连续参数经验有效”还是“催化剂家族决定一切”。每个条件建议 3 次独立重复。

### P2：光催化产氢五点机制面板

这一组专门验证 CARE 提出的可解释假设：无 dye、无 surfactant、低量 sodium silicate 的配方应整体优于含 dye 或含较多 surfactant 的配方。C1-C3 是预测较好的区域，C4-C5 是机制反例。五个条件每个建议 3 次独立重复。配方数值来自公开数据表，但原文变量没有在仓库中统一成实验室单位，开做前必须按上游 SOP 确认加样单位和总体积。

## 实验执行和回传

1. 实验室只看盲码和条件，不看历史结果；同一面板随机化执行顺序。
2. 每个条件使用独立重复，建议 `n=3`；同时保留空白、实验室常用标准条件和仪器校准记录。
3. C-N 与 Suzuki 回传 isolated yield 或经实验室确认的统一 yield 定义，并附原始 LC/GC 数据；光催化回传 HER、采样时间、光源、温度和原始定量数据。
4. 记录试剂批次、实际加样、任何偏离 SOP 的操作及失败实验。失败实验不能从分析中删除。
5. 数据回传后，干实验团队按盲码揭盲，比较方向命中率、重复方差、effect size 和置信区间，不以单个最高值作为成功标准。

## 边界

目前没有“光催化硝基参与的 Buchwald-Hartwig 偶联”这一精确体系的结构化轨迹，因此本清单不能冒充该反应的验证方案。现有光催化数据是产氢配方，现有 C-N 数据是 Baumgartner 流动化学 campaign。若实验室能提供硝基体系的候选空间、SOP 和少量起始数据，我们再单独冻结一版机制相近的 prospective protocol。

表中的 loading 与光催化加样量保留数据源原值。实验室必须先确认单位、浓度定义、设备边界和 EHS 要求；这份清单是验证设计，不替代实验 SOP。

## 交接文件

- 发给实验室：本说明和 `wetlab_conditions.csv`。
- 内部保留：`internal_provenance.csv`，等实验结果回传后再据此揭盲。
- 建议首轮规模：11 个条件，每个条件 3 次独立重复，共 33 次实验；实验室标准条件和空白对照另计。
