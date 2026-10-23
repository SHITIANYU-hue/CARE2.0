# BH 语义标签匿名化：正式运行状态

后续更新：用户已选择另起完整 v2，并批准统一整库回退及 20 次新调用。v1 保留为已停止试验，不恢复或与 v2 拼接；当前执行见 [v2 状态](BH_SEMANTIC_ANONYMIZATION_V2_STATUS.md)。

2026-10-20，WSL `/home/tangchao/projects/care2.0`。当前状态：**按已批准的无效输出停止规则终止，等待后续实验方向决策。**

## 当前结果与费用

| 项目 | 已核实结果 |
|---|---|
| 累计真实模型请求 | 11 次：原运行 1 次 + 批准修订后新增 10 次 |
| 复用 | 首条 `pair_00_labeled` 原样复用，没有重复 POST 或费用 |
| 有效规则库 | 10 个，构成 pair_00–pair_04 共 5 个完整配对 |
| 失败库 | 第 11 槽 `pair_05_anonymous`，未知目标字段／取值 |
| 尚未请求 | 原定 20 槽中的后 9 槽 |
| 累计已确认费用 | **$0.0125145**，其中新增 $0.0112752、原调用 $0.0012393 |
| 未决费用预留 | $0 |
| 校准、留出与 AUC | 均未运行；没有可报告的 AUC 效应 |

用户批准的是：复用首条响应、继续剩余 19 次，但保留无效返回即停止的契约。此次实际在新增第 10 次请求返回无效库后停止；没有重试、补抽、替换或修补。

## 此次停止原因

`skill_003 / rule_004`（`skills[2].rules[0]`）写入条件 `F19=V10`，反解为 `reactant_1_halide_type=chloride`。这是源任务字段，不在目标可执行目录中；冻结提示已有 `target_field=F01` 的映射，目标目录允许的是 `F01=V10`。

只读检查枚举了该响应全部 6 条规则的 7 个条件，仅此一处非法。重新构造配对材料、逆映射、消息与冻结哈希均一致；未发现映射或验证器错误。不能自动将 F19 改成 F01，因为这会修补模型输出，违反已批准契约。

全部 11 份响应重新严格验证；10 份有效解码与保存库完全一致。独立核查确认冻结源码、配置、输入、原失败证据及修订来源不变。证据：`results/2026-10-20-bh-semantic-anonymization-v1-generation-runtime-amendment-v1/post_stop_audit.json`。

## 已批准的运行时修订与首次停止

原运行的首个响应格式有效，包含费用 $0.0012393；冻结执行器强制查询辅助 generation 元数据，在约 1/2/4 秒的三次查询均返回 404 后停止。延迟只读查询已确认同一 ID、费用和非 BYOK 状态。

用户随后明确批准运行时修订：优先使用响应内可靠费用及明确的非 BYOK 标记，缺失时保留辅助查询；原样复用首条结果，仅继续其余请求。修订提案通过 8 项离线测试，方案 SHA256 为 `0ca9404e3ee2a66cceca6f08e22723b57121c334337da2449fa862712b3d2104`。提案文件仍保留批准前快照文字；实际授权与执行源码保存在 `.runtime_amendment` 目录。

此次费用读取已正常工作；新的停止原因是真实无效生成，与首次账单延迟不同。

## 科学设计与证据边界

- OpenRouter `openai/gpt-4o-mini`，temperature 0.45、max_tokens 5000，原定 10 对；每库五策略，规则数 `[2,1,1,1,1]`，总费用上限仍为 $1。
- 原模型、提示词、匿名化映射、生成顺序、五选一预算、gate 和种子均未改变。
- 20 份消息、10 份私有映射、116 份源码及数据／依赖输入已冻结；私有映射未发给模型。
- 校准种子 202610201000–202610201049；留出种子 202610202000–202610202099，均尚未执行。
- 离线检查 317 passed / 1 skipped，映射执行真值比较 155,520 次通过。单个 Windows 专属测试在 Linux 跳过。
- 该消融保留数值、频数和规则空间，检验标签语义的作用，不证明信息论匿名化、预训练独立性或新数据泛化。
- 不能把现有 5 对当作完成版 10 对确认性实验。按照当前协议，不对不完整运行报告部分 AUC。

## 下一步需用户选择

本轮失败及有效输出均保留，不再自动请求模型。可选方向：

1. **另起新版本（建议）**：先明确两组一致的无效输出处理规则、主比较含义和调用预算，离线验证后重新冻结新的完整生成序列。现有输出保留作方法开发证据，不挑选有效库拼成新实验。新 API 调用需明确授权。
2. **仅分析现有 5 对**：单独冻结探索性分析方案，再做本地校准和留出；不再调用 API，但必须标为中途停止后的探索结果，不能替代原定确认性实验。
3. **暂缓匿名化实验**：保留当前证据，转向其他已明确的非 RSI 工作。

本文件仅记录选择边界，不批准或冻结上述任何新方案。

## 冻结与归档

- 配置：[bh_semantic_anonymization_v1.json](configs/bh_semantic_anonymization_v1.json)
- 锁：[bh_semantic_anonymization_v1.lock.json](preregistrations/bh_semantic_anonymization_v1.lock.json)，SHA256 `9ec57bd9570fff81340195de8af1a85eee184569171b2f3935360d8cc534b201`。
- 既有归档：`bh-semantic-anonymization-v1-protocol-frozen`、`bh-semantic-anonymization-v1-generation-stopped`、`bh-semantic-anonymization-v1-runtime-amendment-proposal`。
- 本次运行：`results/2026-10-20-bh-semantic-anonymization-v1-generation-runtime-amendment-v1/`；真实修订源码、授权与新旧计数位于同名 `.runtime_amendment/` 旁侧目录。
- 本次归档 ID：`bh-semantic-anonymization-v1-runtime-amendment-invalid-stopped` 与 `bh-semantic-anonymization-v1-runtime-amendment-provenance`；两者须一并保留，不能只保存有效库。
- 首次冻结曾在 API 调用及正式锁之前因相对路径处理失败而中止，已保留 `outputs/bh_semantic_anonymization_v1_freeze_attempt1_aborted/`；路径问题已在正式冻结前修复。
