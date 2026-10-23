# BH 语义标签匿名化 v2：已完成

更新：2026-10-20。完整结果见 [实验报告](BH_SEMANTIC_ANONYMIZATION_V2_RESULTS.md)。运行环境为 WSL `/home/tangchao/projects/care2.0`。

20 次全新生成、50 个校准种子、100 个留出种子全部完成，独立统计与轨迹审计 PASS。无效库按用户批准的整库回退方案纳入全部 10 对主比较；没有修补、补抽或复用 v1 输出。

- 有效库：有标签 10/10，匿名 7/10；匿名 pair_00、pair_02、pair_07 整库回退。
- 主 AUC 差（有标签 − 匿名）：**−0.260，95% CI [−1.781, 1.365]**；五个种子块 3/5 为正，未达到预定稳定优势标准。
- Gate 后次要差：**−0.788，95% CI [−2.537, 0.825]**；两组分别 2/10、3/10 库通过 gate。
- 主结论合并生成合法性与策略效用；不能解释为纯化学知识因果效应，也不能由区间跨 0 宣称等效。
- 新费用 $0.0218514；含 v1 $0.0125145 后累计 $0.0343659，低于批准的 $1。校准、留出和审计无模型 API 调用。

生产冻结前 141 项离线测试通过。原运行完成校准与选择锁后环境中断，原留出文件为 0 字节；原目录完整归档。外部恢复入口经 8 项离线测试与独立审查，在新目录复用原校准及原选择，仅执行同一 100 个留出种子。未修改冻结配置、源码、规则、gate 或种子。13,200 条逻辑轨迹和 按相同执行配置合并后保存的 12,150 条轨迹完整核对。

主要位置：

- 配置：`configs/bh_semantic_anonymization_v2.json`
- 正式锁：`preregistrations/bh_semantic_anonymization_v2.lock.json`，SHA256 `4b35575db7a47ac36e2cace37cb65224c6596fdc8e7c9f5eb592595862cbe516`
- 选择锁 SHA256：`8d5b211b2c723bf15e8135dccfedca51e37e0a5dfde4b75bf8d724a1be2ae31f`
- 恢复锁 SHA256：`336e59d812b625d80b3ede72e39e4bcdfc15287031e0fec959adc93fbce0f613`
- 生成：`results/2026-10-20-bh-semantic-anonymization-v2-generation/`
- 完整评估：`results/2026-10-20-bh-semantic-anonymization-v2-evaluation-recovery-1/`
- 原中断评估：`results/2026-10-20-bh-semantic-anonymization-v2-evaluation/`
- 审计与执行日志：`outputs/bh_semantic_anonymization_v2_execution/`
- 报告与图：`results/2026-10-20-bh-semantic-anonymization-v2-report/`

已打包验证：草案、正式冻结、完整生成、原中断评估、恢复运行入口与完整评估。报告及审计证据另行归档。下一步专家盲写、第二数据集或新 API 实验均需另行明确方案。
