# uv 环境与同事部署

整个仓库使用根目录的 `pyproject.toml` 和唯一的 `uv.lock`。AstaBench 是同一 uv workspace 中的成员；所有功能共用锁定的依赖版本和根目录 `.venv/`。

## 第一次部署（WSL / Linux）

本地工作目录为 `/home/tangchao/projects/care2.0`，分支为 `tangchao/repository-reorganization`。同事在分支发布后可执行：

```bash
# 首次安装 uv；已有 uv >= 0.12.9 可跳过。
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone --branch tangchao/repository-reorganization https://github.com/SHITIANYU-hue/CARE2.0.git
cd CARE2.0
uv sync --locked
uv run --locked python care.py smoke
```

`.python-version` 选择 Python 3.12。uv 会按需下载解释器并建立环境，无须手动激活 venv。项目声明 Python >= 3.11；验收和 CI 使用 3.12。无需安装系统 pip、Conda 或 CUDA。

WSL 建议将仓库放在 Linux 的 `~/projects/`，所有命令都在 WSL 终端执行。

## 按功能选择依赖

| 用途 | 安装命令 | 运行示例（仓库根目录） |
| --- | --- | --- |
| 主线、知识库、绘图与测试 | `uv sync --locked` | `uv run --locked python care.py smoke` |
| 仅运行主线，省略 pytest | `uv sync --locked --no-dev` | `uv run --locked --no-dev python care.py smoke` |
| RDKit 描述符重建 | `uv sync --locked --extra descriptors` | `uv run --locked --extra descriptors python -c "from rdkit import Chem; print(Chem.MolFromSmiles('CCO').GetNumAtoms())"` |
| 本地语义向量 | `uv sync --locked --extra embeddings` | `uv run --locked --extra embeddings python knowledge_base/build_embeddings.py --help` |
| AstaBench | `uv sync --locked --extra astabench` | `uv run --locked --package care2-astabench pytest experiments/astabench/tests` |
| 全部功能 | `uv sync --locked --all-extras` | `uv run --locked --all-extras pytest experiments/astabench/tests` |

`uv run` 也会同步环境：运行可选功能时需带对应的 `--extra`、`--all-extras` 或 `--package`。单纯 `uv run --locked` 只保证默认依赖可用；不要依赖某次手动安装留下的包。共享环境可在这些用途之间重新同步，锁文件不变。

默认环境含 NumPy、Matplotlib 和 pytest。Linux / Windows 的 embeddings 选择 CPU PyTorch，避免安装 CUDA 运行库；macOS 使用 PyPI 的 PyTorch wheel。本地语义模型的权重在首次实际使用时另行下载。已有知识库的 hashed 向量和 HTTP API 模式不需要 embeddings extra。

AstaBench 保留原有 `astabench==0.5.3`、`inspect-ai==0.3.203`、`matplotlib==3.10.5` 等约束。数据集许可、HF_TOKEN 和模型 API 凭据按 [AstaBench 说明](../experiments/astabench/README.md) 设置；安装依赖和离线测试不会发起付费模型调用。

## 验证主线和全部功能

```bash
uv sync --locked
uv run --locked python care.py check
uv run --locked python care.py smoke
uv run --locked python care.py kb-build
uv run --locked python care.py kb-query gate --limit 2

# 部分历史回归测试依赖这三个可校验的结果包。
uv run --locked python care.py artifacts restore \
  replay-2026-08-10-baumgartner-warmstart-v2-external-confirmation \
  replay-2026-08-15-opus5-generalization-study \
  replay-2026-08-24-online-llm-matched-classical-audit-v1
uv run --locked pytest

uv sync --locked --all-extras
uv run --locked --all-extras python -c "import rdkit, sentence_transformers, torch, care_astabench_solver"
uv run --locked --all-extras pytest experiments/astabench/tests
```

归档完整性可单独检查：`uv run --locked python care.py artifacts verify --all`。图形环境不可用时可设置 `MPLBACKEND=Agg`。

## 本次部署验收（2026-09-15）

在本机 WSL2 Ubuntu 24.04、Python 3.12.3、uv 0.12.9 下完成：

- 在新建的临时环境中执行 `uv sync --locked`，随后完整主线测试 **311 passed、1 skipped**；唯一跳过是 Linux 不适用的 Windows junction 测试。
- 主目录环境执行 `uv sync --locked --all-extras`，RDKit、Sentence Transformers、CPU PyTorch 和 AstaBench 导入成功；`uv pip check` 确认已安装依赖相容。
- AstaBench 的 **12 项离线测试全通过**。
- synthetic smoke、189 张知识卡片构建/查询，以及 **138 个归档的完整校验**通过。
- `uv lock --check` 通过。CI 配置已同步为 uv，远程 CI 结果以推送后的实际运行为准。

验收未下载语义模型权重，未运行付费模型或新的完整科学实验。日志保存在本地 `.git/uv-validation/`，不进入源码目录。

## 团队开发与完整实验复现的范围

该分支可以作为主线开发、测试及扩展功能开发的起点。七对默认主线任务所需的冻结模型记录和数据已跟踪，不依赖维护者本机的未提交文件。

2026-09-16 在 WSL 中从提交 `ff29aa93` 建立独立浅克隆和新的 `.venv` 后，默认安装、无历史结果的核心测试（57 passed、1 skipped）、恢复三个证据包后的完整主线测试（311 passed、1 skipped）、全部可选依赖导入和 AstaBench 离线测试（12 passed）均通过。克隆目录的受跟踪文件没有变化。

完整实验还需要以下输入或外部条件：

- **其他历史报告和实验**：按配置恢复 `artifacts/` 中对应的结果包；默认完整回归所需的三个包见上文。
- **额外公共数据任务**：`flip2_trpb_one_to_many.csv.gz`、`matbench_mp_gap.json.gz`、`matbench_mp_e_form.json.gz` 未随 Git 提交，相关任务首次运行时由 `run_synthetic_suzuki.py` 联网下载。当前下载器未校验内容 SHA，因此不能承诺上游文件未来变化时仍逐字节复现；它们不是上述七对默认主线测试的前提。
- **在线 LLM 实验**：使用调用方自己的 API 凭据和额度，按照对应实验说明配置 `CARE_LLM_API_KEY`、`COMMONSTACK_API_KEY` 等。
- **AstaBench 实际评测**：需要数据集访问许可、Hugging Face 凭据及所用模型的 API 凭据；复现官方 ReAct 对照还需按实验协议获取外部 `allenai/agent-baselines` 仓库的指定提交。本仓库的 uv 环境包含 CARE 适配器。
- **本地语义模型**：首次使用另外下载权重；uv 锁定的是 Python 依赖，不包含模型权重和外部服务。

当前部署验收覆盖 WSL2 Ubuntu 24.04 / Python 3.12；其他操作系统、Python 版本、完整科学实验和外部服务应按实际使用场景再验收。推送后的 GitHub Actions 通过后，可将该提交作为团队已验证的开发基线。

## 添加和升级依赖

```bash
uv add PACKAGE
uv add --dev PACKAGE
uv add --optional descriptors PACKAGE
uv add --package care2-astabench PACKAGE
# 有意升级某个已锁定的包：
uv lock --upgrade-package PACKAGE
uv sync --locked
```

提交相应的 `pyproject.toml` 和根 `uv.lock`。不要提交 `.venv/`，也不要在 AstaBench 下新建第二份锁文件。CI 使用 `uv sync --locked`，依赖声明与锁文件不一致时会失败。

原有两份根 `requirements*.txt` 已由 uv 元数据取代。如外部平台只接受 requirements 格式，临时导出：

```bash
uv export --locked --no-dev --no-emit-project --format requirements.txt --output /tmp/care2-requirements.txt
```

带 `--all-extras` 的导出包含本地 workspace 成员，使用时需保留仓库源码；直接使用 `uv sync` 是首选部署方式。

参考：[uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)、[依赖管理](https://docs.astral.sh/uv/concepts/projects/dependencies/)、[CPU PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)。
