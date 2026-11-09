# 投稿补充实验交接说明

本分支 `codex/bh-nonrsi-wsl-20261020` 汇集已发布的非 RSI 投稿补充代码、结果和可恢复证据。先读 [11 月跨模型结果](../experiments/care_replay/SUBMISSION_COMPLETED_RESULTS_20261109.md)，再读 [10 月完成索引](../experiments/care_replay/SUBMISSION_COMPLETED_RESULTS_20261023.md)。只查看结论无需调用模型或重新跑科学实验。

## 按原投稿计划查看

| 原计划事项 | 实际工作及结果 | 入口 |
|---|---|---|
| 前置重分析 | 32 项核心 AUC 记录完成统一尺度；动作、token 和历史预算审计完成，账单仍有未知项 | [AUC](../experiments/care_replay/AUC_REANALYSIS_V1_RESULTS.md)、[动作成本](../experiments/care_replay/OFFLINE_ACTION_COST_BUDGET_AUDIT_V1_RESULTS.md) |
| 实验一跨模型 | 新非 RSI 子实验完成四固定任务、八迁移路线、660 生成槽。分子/带隙均值方向较一致，BH 未稳定复制；BH→Suzuki 源信息增量最一致。未完成原多次修订与等效检验 | [11 月结果](../experiments/care_replay/SUBMISSION_COMPLETED_RESULTS_20261109.md) |
| 实验二随机规则 | 历史 BH 库相对所检验覆盖匹配随机库有优势；不能推广为普遍超过强 BO 或所有新生成库 | [覆盖匹配](../experiments/care_replay/BH_COVERAGE_MATCHED_RULE_CONTROLS_V1.md) |
| 实验二匿名化 | 10 对新生成完成，未支持语义标签稳定优势；不是等效或排除记忆证据 | [匿名化](../experiments/care_replay/BH_SEMANTIC_ANONYMIZATION_V2_RESULTS.md) |
| 实验二 Suzuki | 旧标签/随机/匿名比较均未达到稳定优势标准；与新 F−T 信息增量是不同问题 | [Suzuki](../experiments/care_replay/SUZUKI_SECOND_DATASET_V1_RESULTS.md) |
| 实验三接受测试 | 调整后的固定加性算子四臂完成，23 路线/13 依赖组。组成匹配 gate 误拒从 3/16 增到 10/16，整体未改善；只有一条有害路线 | [gate](../experiments/care_replay/EXPERIMENT3_GATE_VALIDATION_V1_RESULTS.md) |
| 专家、实验四新数据、湿实验 | 没有完成的新增结果，不由公开回放替代 | [原计划进度](../experiments/care_replay/SUBMISSION_NONRSI_ROADMAP_AFTER_AUC_20261021.md) |

## 新旧实验的完成状态

11 月 8 日完成的是一次生成固定规则的跨模型实验，严格主分析及事后外层代码块补充都已算完。Haiku 严格格式失败按冻结流程回退，属于已完成的无效生成终态，不是等着重新生成的缺口。补充仅恢复已有响应的可执行部分，不新增模型调用。

另有旧在线 proposer–critic 三套协议，原计划合计 840 条轨迹，保存记录为 18 完成、16 失败、806 未见终态。这是另一工作线，不是本次 660 槽新实验的剩余任务。旧失败含费用、余额、密钥、人工中断和校验失败；部分耗尽重试或属于终态。审计还发现旧源码/路径/锁的恢复问题。本轮保留其失败和未完成事实，未覆盖旧结果或将其列为完成。

原 TrpB 覆盖解释的同版本输入复核、原多轮修订等效设计仍有缺口；Amylase 旧执行已无效，不计为确认。未跟踪的 trace-skill 设计材料不属于此次已完成实验发布。原始投稿计划 Word 未包含在仓库中，现有计划和本表提供计划对应关系。

## 在同事环境查看与恢复

需要查看全部提交历史时不要使用浅克隆。已有仓库可执行：

```bash
git fetch origin
git switch --track origin/codex/bh-nonrsi-wsl-20261020
```

若本地分支已存在，切换到该分支后快进更新。阅读上述 Markdown 不需安装依赖或恢复大结果。仅归档恢复工具使用标准库；安装 uv 后可用 `uv run --no-project --python 3.12.3 python care.py artifacts ...`。检查逐库统计、响应和冻结链时，从仓库根一次恢复全部 14 个证据分包：

```bash
uv run --no-project --python 3.12.3 python care.py artifacts restore \
  nonrsi-publication-20261109-history-part01 \
  nonrsi-publication-20261109-history-part02 \
  nonrsi-publication-20261109-history-part03 \
  nonrsi-publication-20261109-strict-control \
  nonrsi-publication-20261109-strict-molecular-materials \
  nonrsi-publication-20261109-strict-bh \
  nonrsi-publication-20261109-strict-suzuki \
  nonrsi-publication-20261109-fence-control \
  nonrsi-publication-20261109-fence-molecular-materials \
  nonrsi-publication-20261109-fence-bh-calibration \
  nonrsi-publication-20261109-fence-bh-heldout \
  nonrsi-publication-20261109-fence-suzuki-calibration \
  nonrsi-publication-20261109-fence-suzuki-heldout \
  nonrsi-publication-20261109-dependencies
```

每包源内容不超过 90 MiB，完整原路径与逐文件 SHA-256 保留。history 包保存初冻、最初执行、恢复修订 v1–v9 与离线完成控制；strict 包保存 v10 最终生成、校准、留出和分析；fence 包单独保存事后解析派生与评估；dependencies 包保存冻结链引用的依赖快照。已有同内容文件跳过，异内容拒绝覆盖。不要只恢复最终 evaluation 而漏掉上游锁、账本和依赖。

发布核验记录另用小包恢复：

```bash
uv run --no-project --python 3.12.3 python care.py artifacts restore nonrsi-publication-20261109-verification
```

可在干净目录中用 `--destination /tmp/care2-handoff-evidence` 检查归档恢复；要执行完整核验，需把证据恢复到含本分支源码的仓库中。10 月实验按其索引另行恢复。实验三有原始内层归档，需要按索引展开；原始 ProteinGym ZIP 不在仓库，从源数据重跑另需取得协议指定版本并核对哈希。

## 源码与复核

- 固定规则生成、校准、统计：[nonrsi_cross_model_v1](../experiments/care_replay/protocols/nonrsi_cross_model_v1/)
- 历史恢复实现：[resume_v1](../experiments/care_replay/protocols/nonrsi_cross_model_resume_v1/) 至 [resume_v10](../experiments/care_replay/protocols/nonrsi_cross_model_resume_v10/)
- 11 月 8 日离线完成：[offline_resume_v1](../experiments/care_replay/protocols/nonrsi_cross_model_offline_resume_v1/)
- 外层代码块补充：[fence_supplement_v1](../experiments/care_replay/protocols/nonrsi_fence_supplement_v1/)
- 发布分包工具：[publish_nonrsi_archives.py](../tools/publish_nonrsi_archives.py)
- 网络禁用、只读科学证据核验：[verify_nonrsi_handoff.py](../tools/verify_nonrsi_handoff.py)

完整核验强制要求冻结的 Python 3.12.3、NumPy 2.5.3、RDKit 2026.3.6 与 setuptools 84.0.0；默认 `uv sync` 不足以建立该环境。使用 Linux 或 WSL，在仓库之外创建独立环境，不修改已有研究环境或原冻结文件：

```bash
uv venv --python 3.12.3 /tmp/care2-nonrsi-frozen-env
uv pip install --python /tmp/care2-nonrsi-frozen-env/bin/python \
  -r configs/nonrsi_handoff_requirements.txt
/tmp/care2-nonrsi-frozen-env/bin/python tools/verify_nonrsi_handoff.py \
  --output /tmp/care2-handoff-verification
/tmp/care2-nonrsi-frozen-env/bin/python care.py artifacts verify --all
/tmp/care2-nonrsi-frozen-env/bin/python care.py check
```

第一次建环境可能需要下载 Python 和依赖；这是软件安装，不是模型调用。若当前冻结环境已经满足以上版本，可直接使用它的解释器。普通开发环境说明另见 [ENVIRONMENT.md](ENVIRONMENT.md)。

核验脚本复用冻结链、账本、检查点及派生内容的校验，不加载密钥、不发模型请求、不写回原实验目录。模型调用的历史授权及停止文件属于审计证据，不是交接时重新调用 API 的指令。执行身份依据各阶段冻结文件哈希；归档的 source_commit 是发布来源快照，不代表实验当时已在该提交中运行。

## 解释结果时保留的区别

1. 固定规则有效，不等于源实验反馈有效；T 不包含源反馈。
2. F−T 是整体源信息包，F−S 才更接近新增源结果摘要的增量，仍不是正确源绑定的独有机制证明。
3. 候选技能有收益，不等于 gate 后实际部署有收益。
4. 新跨模型分析是描述性复现；Haiku 去代码块是事后敏感性分析，不能合成“三模型确认性成功”。
5. 同池多起点和多次生成不等于新增独立科学数据集；离线目标校准成本与每轨迹搜索预算分开报告。
