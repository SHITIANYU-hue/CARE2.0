# 仓库整理与复现

## 基准与保全范围

整理分支：`codex/repository-reorganization`。

基准提交：**`4c6f19f8cf62b25558f966604ce0b7644578f889`（2026-09-01，`eval: add official AstaBench HMS scores`）**。当前整理分支从这一提交继续，保留目录整理、归档工具、主线冻结输入及截止点以内的代码和实验资料。

**本次范围调整：按要求排除 RSI，以 `4c6f19f8` 为基线，其后实验不纳入当前分支。** 原始研究分支仍保留对应历史；当前代码、配置、测试、文档和归档目录均以此截止点为准。

首次整理已核对全部 7 个远程分支。分支关系审计见 [项目主线](PROJECT_MAINLINE.md)；本次沿用该审计作为历史来源说明，不把远程最新树作为当前功能范围。`llm-traces@d95253a5c476a0879c5c45652f0f5a049037d4af` 的 17 件早期独有证据继续保存在 `llm-traces-2026-07-16` 归档中。

| 项目 | 数量 / 尺寸 |
| --- | ---: |
| 基准提交原始文件 | 25,326 |
| 基准提交总字节 | 1,466,811,989 |
| 已封存的基准历史文件 | 24,975 |
| 另一分支独有证据 | 17 |
| 归档内原始文件合计 | 24,992 |
| tar.gz 归档数 | 138 |
| 归档总字节 | 575,763,319（约 549.09 MiB） |

归档逐项统计见 [catalog.json](../artifacts/catalog.json) 及 [归档目录](../artifacts/README.md)。整理后共 **511 个受跟踪文件**，包含源码、文档、运行输入和 138 个归档。

原始研究分支与 Git 历史保留；整理分支重建在上述基线上。当前文件树变简洁不等于历史对象变小。需要较浅的工作副本时，可克隆此分支并使用 `--depth 1`，但归档本身仍需下载。

## 内容放置

| 原位置 / 内容 | 新位置 / 处理 |
| --- | --- |
| `experiments/care_replay/results/**` | `artifacts/replay-results/*.tar.gz` |
| `experiments/care_replay/outputs/**` | `artifacts/legacy-outputs/*.tar.gz` |
| `experiments/astabench/results/**` | `artifacts/astabench-results/*.tar.gz` |
| 截止点以内的 PPT 和相关讲稿 | `artifacts/presentations/*.tar.gz`、`docs/` |
| `llm-traces` 的 17 件独有证据 | `artifacts/branch-evidence/llm-traces-2026-07-16.tar.gz` |
| 原根 README、overview、飞书实验报告 | `docs/history/` |
| 主线必需的 5 件 target-only frozen records | `experiments/care_replay/skill_banks/frozen_records/` |
| 配置、预注册、技能、知识卡片、公共输入数据 | 继续以原始文本/输入文件跟踪 |

主要脚本、导入关系和原有实验相对路径继续有效。`care.py` 提供统一入口，`docs/PROJECT_MAINLINE.md` 提供脚本职责地图。此次没有混入大规模 Python 包重构或算法改变。

冻结记录从原始 Git blobs 按字节提取，配置只更新定位路径；[PROVENANCE.md](../experiments/care_replay/skill_banks/frozen_records/PROVENANCE.md) 记录出处及 SHA-256。同名相同内容的恢复副本可以被旧报告解析器识别；不同内容仍拒绝歧义解析。

## 归档完整性

归档直接从截止点对应的 Git 原始对象生成，不依赖 Windows 是否成功检出长路径；独有 trace 证据按其来源分支生成。所有原始文件名、目录、内容和可执行 mode 均记录在 tar 中；成员 mtime/uid/gid 固定以便确定性生成。每个 tar.gz 另含 `CARE_ARCHIVE_MANIFEST.json`，登记原始 commit、path、Git blob、字节数、SHA-256。

`artifacts/catalog.json` 是小型目录元数据，不是裸运行日志。它保存每个归档的 SHA-256、文件数、尺寸和来源。单包按最多 90 MiB 原始内容分组，生成结果均低于 100 MiB。

```bash
python care.py artifacts verify --all
python care.py artifacts list --category replay-results
```

校验会检查外层 SHA、tar 路径、普通文件类型、重复成员、逐文件 SHA、数量和尺寸。恢复前会预检全部所选归档和目标路径；拒绝路径逃逸、链接、不同内容覆盖及跨包版本冲突。

## 恢复历史实验

```bash
# 补充的 live LLM trace 分支
python care.py artifacts restore llm-traces-2026-07-16

# 典型的主线历史结果
python care.py artifacts restore replay-2026-07-23-source-evidence-extension

# 独立目录复核，避免与现有运行混淆
python care.py artifacts restore replay-2026-08-24-online-llm-matched-classical-audit-v1 --destination review-copy
```

恢复目录包含原始完整仓库相对路径，而不是直接把内容放进目标目录根。部分历史归档带原有内层 tar.gz；工具保留它们，不递归展开。分包实验必须恢复全部 `partNN`；ID 见 [归档目录](../artifacts/README.md)。

`restore --all` 会重建所收录的历史文件树，所需未压缩空间可从 catalog 的成员尺寸统计。不同快照可能含同路径不同版本；此时工具会拒绝全部恢复，应选择具体实验或分别指定目标目录。日常开发只恢复所需内容。

其他冻结配置仍保留历史出处与 hash。运行在线 LLM 实验或重建历史图表之前，先按配置中的 `results/...` 路径恢复对应实验包。私有数据和外部 API 凭据不包含在归档中。

## 验证入口

```bash
python -m pip install -r requirements-dev.txt

# 在未恢复任何历史结果的工作树内也能执行
python care.py smoke
python care.py kb-build
python care.py kb-query gate --limit 2
python -m pytest experiments/care_replay/tests/test_transfer_skill.py experiments/care_replay/tests/test_source_outcome_transfer.py experiments/care_replay/tests/test_cross_task_router.py experiments/care_replay/tests/test_hidden_target_noninterference.py tests

# 完整研究测试还读取历史证据，需先恢复以下 3 个包
python care.py artifacts restore replay-2026-08-10-baumgartner-warmstart-v2-external-confirmation replay-2026-08-15-opus5-generalization-study replay-2026-08-24-online-llm-matched-classical-audit-v1
python -m pytest

# 提交前先暂存，再检查索引
python care.py check
git diff --cached --check
```

AstaBench 保持独立的 `experiments/astabench/pyproject.toml` 与 `uv.lock`，在该环境运行其测试。完整 LLM 在线实验和多种子科学复现与仓库验收测试分别执行，不使用 smoke 的缩减预算替代原始结果。

## 后续维护

新增实验按 [CONTRIBUTING.md](../CONTRIBUTING.md) 先写源码和协议，再写被忽略的结果目录，最后用 `python care.py pack` 封存。归档工具保留源文件，不自动删除或上传；确认校验完成后再由维护者清理本地临时输出。

## 本次验收（2026-09-15）

138 个历史 tar.gz 已完整校验通过，覆盖外层 SHA-256、24,992 个原始成员的 SHA-256、路径、数量和尺寸。

源码范围已逐路径对照基准 Git tree：24,975 个封存路径和 17 个独有 trace 路径全部与原 blob 对应，保留源码及输入均可追溯。主线配置的源/目标冻结记录可用，5 个迁出的目标记录与基准原始字节一致。

未恢复任何历史结果时，主线及工具测试 **56 passed、2 skipped、30 subtests passed**；最小 synthetic smoke 通过，知识库成功构建 189 张卡并可查询。恢复上述 3 包的 102 个原始文件后，完整离线回归为 **310 passed、2 skipped、33 subtests passed**。两项跳过均因本机 Windows 没有创建真实符号链接的权限；junction/reparse 检查已通过。测试完成后已再次清理临时恢复的历史文件。

验收范围包括：归档外层与逐成员校验、原始 Git blob 映射、主线冻结输入可用性，以及恢复上述 3 包后的 replay、knowledge_base 和仓库工具测试。AstaBench 的适配器测试需要独立 inspect-ai/astabench 环境，应另行记录；未启动新的付费 LLM 实验。

GitHub Actions 工作流验证布局、归档、无需恢复的主线和恢复历史证据后的测试。远程 CI 的状态以推送后实际运行结果为准。
