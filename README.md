# CARE 2.0

**从源实验的证据编译可执行迁移技能，在目标任务中校准、门控，并留下可审计的决策记录。**

本整理分支 `tangchao/repository-reorganization` 以 **2026-09-01 的 `4c6f19f8`** 为代码和实验截止点，保留目录整理、归档工具及更早的独有分支证据。全部 7 个远程分支的历史关系见主线报告。

## 先看这里

- [环境与依赖](docs/ENVIRONMENT.md)：uv 安装、可选功能和统一运行方式。
- [项目主线与全部分支关系](docs/PROJECT_MAINLINE.md)：研究问题、算法链路、证据层级和扩展实验。
- [主方法定义](experiments/care_replay/CARE2_METHOD.md)：冻结 source-outcome transfer 的执行契约。
- [实验与产物目录](artifacts/README.md)：138 个按实验组织的 tar.gz、校验与恢复命令。
- [整理记录与复现步骤](docs/REORGANIZATION.md)：目录变化、保留范围、测试和历史恢复。
- [开发与归档约定](CONTRIBUTING.md)：新增代码、配置和实验产物放在哪里。

## 主线

```text
固定源实验结果 → 冻结 LLM/语义技能 → 可执行 TransferSkill
             → 目标观测校准与 gate → 独立留出评测 / 基线回退
```

当前主线是有限候选池的离线实验回放。在线 LLM 选点、零经验初始化和外部 AstaBench 评测保留为独立研究轨道，分别报告证据和调用成本。正结果、负结果及调用日志保存在归档中。

## 目录

| 路径 | 内容 |
| --- | --- |
| `care.py` | 统一导航入口：smoke、pair、suite、知识库、归档 |
| `experiments/care_replay/scripts/` | 主算法、数据适配器和各阶段实验脚本 |
| `experiments/care_replay/configs/` | 可执行配置与冻结协议 JSON |
| `experiments/care_replay/skill_banks/` | 可执行技能及主线依赖的冻结模型记录 |
| `experiments/care_replay/data/` | 公共回放数据、描述符和来源说明 |
| `experiments/care_replay/tests/` | 算法、门控、数据和审计测试 |
| `experiments/astabench/` | AstaBench 外部评测适配器，作为 uv workspace 成员 |
| `knowledge_base/` | 知识卡片、检索与受控更新 |
| `artifacts/` | 历史结果、旧 outputs、PPT 与独有分支证据的归档 |
| `docs/` | 主线说明、论文材料、讲稿和历史说明 |
| `tools/`、`tests/` | 归档工具、仓库检查及工具测试 |
| `task_tracker/` | 原有协作事项与任务模板 |

## 快速运行

在仓库根目录使用 uv。项目要求 Python 3.11+，`.python-version` 固定使用 3.12；根目录的 `uv.lock` 统一锁定主线和 AstaBench 依赖。默认安装 NumPy、Matplotlib 和开发测试所需的 pytest。

```bash
uv sync --locked
uv run --locked python care.py --help
uv run --locked python care.py smoke
uv run --locked python care.py pair --help
uv run --locked python care.py suite --help
uv run --locked python care.py kb-build
uv run --locked python care.py kb-query gate --limit 5
```

`smoke` 默认只跑 2 个 synthetic seeds、3 轮，不调用模型 API。真实迁移 suite 使用已冻结的模型记录；7 对任务的全部默认记录都已保留在源码树内。完整多种子评测并非快速检查，参数见主方法定义。

可选功能包括 `descriptors`（RDKit）、`embeddings`（Sentence Transformers 与 CPU PyTorch）和 `astabench`。安装全部可选功能：

```bash
uv sync --locked --all-extras
```

使用可选功能时，运行命令也带上对应的 `--extra <名称>` 或 `--all-extras`。默认 `uv run --locked pytest` 检查主线；AstaBench 单独从仓库根目录运行：

```bash
uv run --locked --package care2-astabench pytest experiments/astabench/tests
```

完整环境说明见 [ENVIRONMENT.md](docs/ENVIRONMENT.md)。

## 查找、恢复和新增实验结果

```bash
uv run --locked python care.py artifacts list
uv run --locked python care.py artifacts verify --all
uv run --locked python care.py artifacts restore llm-traces-2026-07-16
uv run --locked python care.py pack experiments/care_replay/results/my-run --id my-run
```

恢复保留原始仓库相对路径；已有同内容文件跳过，不同内容拒绝覆盖。大实验有 `part01` 等分包，完整恢复需选择全部分包。现有内层 tar.gz 保持原样；逐文件校验清单封装在每个归档内，`artifacts/catalog.json` 仅作为可查询的目录索引。

日志和结果目录默认被 Git 忽略；配置、技能及知识库卡片继续按 JSON 跟踪。原始研究分支与 Git 历史保留；本分支只纳入截止点以内的研究内容，未执行新的付费模型调用。
