# CARE Codex EvE 实验框架

本分支从 tangchao 的 `a0a8c309252ce66e979a6ae421ec3278362203a4` 开始，目标是让三个方法共用任务、结果揭示和指标，不重写 CARE 的科学算法。模型权重固定；本地验证不调用模型。

## 选定的 CARE 版本

复用 `source_outcome_benchmark.json` 中的 **full_source_outcome** 配置、冻结 source/target 模型记录，以及 `run_llm_transfer_router` 中的数值专家路由和逐轮门控。主循环抽成 `router_decisions`，每次选点后等待外部环境提供测量结果。原入口继续调用同一个决策循环。

这与“在线 proposer–critic”和后续 RSI 权重训练不同。默认方法名 **care_router** 表示原始迁移路由：不自动运行离线校准，不冒充已经通过校准的完整选择器。每轮记录 `deployment_mode`、`calibration_status`。

若要评价已经通过离线校准的完整部署策略，先使用原有 `care.py pair/suite` 取得 canonical summary，再导出不含 held-out 结果的选择文件：

```bash
uv run care.py benchmark freeze-care --summary PATH_TO_CANONICAL_SUMMARY.json --output runs/selection.json
```

在对应单任务配置的 CARE method 中设置 `selection_file`。它包含实际执行参数、source 数量/seed、目标 anchor、记录指纹、校准种子及成本；运行时恢复这些参数，不只读取一个通过/拒绝标志。校准与测试种子分离，预算需一致，初始化使用 `care`。每个 source-target pair 使用自己的选择文件。拒绝迁移时执行该 summary 指定的冻结目标策略，而不是改成任意 GP。

## 公共接口

| 层 | 职责 |
| --- | --- |
| `care_harness.environment` | 加载现有数据适配器；分离公开候选和结果表；维护初始观测、选点和预算 |
| `care_harness.runner` | 向方法发公开状态，接收选点，保存决策后揭示结果，统一轨迹和指标 |
| `care_harness.agents` | CARE、目标 GP、Codex CLI、冻结 EvE solver、随机对照 |
| `care_harness.eve` | 调用官方 EvE 搜索；开发集评估服务；冻结策略后部署 |

方法只实现 `select(view) -> {candidate_id, diagnostics?, usage?}`。公开状态含候选条件、已完成来源实验、已揭示目标结果、当前轮次、剩余预算和 seed。已有 CARE 数据适配器把任务结果映射成越大越好的 utility；`task.direction` 明确为 `maximize`，不能把原始需最小化的数值直接接入。

数据层按字段白名单导出 legacy metadata，排除 measured property、产率等隐藏结果及原始行号。程序只在 `reveal(candidate_id)` 时取出一个真实结果。pool extrema 和真实排名只用于结束后的评测。

这实现了数据/执行接口分离。**本地 Codex CLI 与普通 Python solver 不是同一用户主机文件的强隔离边界**；它们的工作目录只放公开数据，但正式实验应在不挂载原始目标表和历史结果的 worker 环境中执行。不能把“通过了公开视图测试”解释为完整文件系统隔离。

## 安装与运行

根环境仍沿用 tangchao 的 uv workspace，不新增第三方运行依赖。最低 uv 放宽为 0.11；默认可按环境需要解析锁文件，不要求每次 `--locked`：

```bash
uv sync
# 校内镜像：这会按镜像重新解析依赖，允许更新 uv.lock。
UV_DEFAULT_INDEX=https://mirrors.osa.moe/pypi/web/simple/ uv sync
```

已有环境也可直接 `.venv/bin/python care.py ...`，无需每次同步。要精确使用现有锁文件可 `uv sync --frozen`；冻结锁中的下载地址不一定改走镜像。没有更改用户全局包源。

所有命令在仓库根执行，配置中的 solver、selection_file 路径相对当前目录。

```bash
# 无模型的完整接线检查：随机、GP、未进化的 EvE 种子策略。
uv run care.py benchmark run --config configs/baselines/smoke.json --output runs/smoke-01

# 查看三条真实路线的预算/任务数，不加载模型或启动 agent。
uv run care.py benchmark run --config configs/baselines/pilot.json --dry-run

# 冻结 CARE 记录 + GP，仅一个配对 seed，不需要 API。
uv run care.py benchmark run --config configs/baselines/pilot.json --seeds 41000 --output runs/care-gp-01

# Codex 需要可用的 CLI 登录/后端额度；研究运行应明确设置 model。
uv run care.py benchmark run --config configs/baselines/pilot.json --methods target_gp,care_router,codex --output runs/codex-01
```

`pilot.json` 包含 ESOL→Lipophilicity、ChemLex→Buchwald–Hartwig、dielectric→band-gap。继承每条原始路线的总预算：分子路线为 5+10，后两条为 2+13；不要误把它们都称为 5+10。默认只运行 `target_gp,care_router`，不会意外消耗模型额度。

`initialization: care` 把 CARE 在公开信息上产生的起始候选固定为每种方法的共同起点，比较后续选点增量；这不衡量各 Agent 自主初始化的端到端收益。`random` 使用同一 seed 的随机起点，用于独立的同起点对照。更改初始化就是更改实验协议，应分开报告。

## Codex 和 EvE

Codex 调用真实 `codex exec --json --output-schema`。每次实验选择开启新会话，分析代码与笔记在该 campaign 的独立临时目录中持续保留。每轮保存公开输入、提示、命令、原始事件和响应，汇总实际返回的 tokens。CLI 失败时报告失败，不替换成 GP。默认例子固定 `gpt-5.4`、medium reasoning；可在 JSON 中调整，EvE 应使用同一模型设置，结果标记具体配置。

Codex 适配器忽略个人配置，使用显式实验参数；默认沿用 CLI 认证。自定义 provider 可在 `command` 数组中显式加入 CLI 配置。未继承个人 MCP、网页搜索等额外信息源。[官方非交互接口](https://learn.chatgpt.com/docs/non-interactive-mode)

EvE 使用 [官方实现](https://github.com/scaling-group/eve)，固定 commit `979065fd2cf1b3be9caf70596478fe27df7815ad`。演化循环、solver/guidance population、采样和 Elo 都由上游负责，本仓库只提供应用和评估接口：

```bash
# 先按 care_harness/eve/README.md 安装上游并 checkout 固定提交。
uv run care.py eve prepare --eve-repo /path/to/eve --workspace runs/eve-development \
  --development configs/baselines/eve-development.json --model gpt-5.4 --iterations 5 --workers 2

# 此步才启动模型搜索，等待额度可用后运行。
uv run care.py eve launch --workspace runs/eve-development

# 在开发结果上选定 solver 后冻结。
uv run care.py eve export --solver /path/to/selected/solver.py \
  --workspace runs/eve-development --output runs/eve-frozen
uv run care.py benchmark run --config configs/baselines/pilot.json \
  --methods target_gp,care_router,codex,eve --output runs/four-method-01
```

完整上游安装、hook/auth 配置、策略函数接口见 [EvE 接入说明](../care_harness/eve/README.md)。`prepare` 不调用模型。默认开发集为 ESOL→FreeSolv 与 phonons→dielectric，与 pilot 的目标结果表分离；开发任务既不能把测试目标当 target，也不能把其带结果的表当 source。不同种子不能替代这条任务边界。

`eve_seed` 仅用于接线检查，不是 EvE 搜索结果。真正 baseline 必须先完成官方演化并保存导出 provenance。开发评估次数、初始和新增观察成本保存在 EvE workspace，模型调用成本由上游 telemetry 保存；它们与目标 replay 成本分别报告。CARE 冻结记录的历史生成/校准成本也不等于当次零模型调用，应单列。

## 输出与验证

每次运行生成 `manifest.json`（配置、提交、数据/策略指纹）、`metrics.csv`、`summary.json`，以及逐 campaign 的 `initial.json`、逐轮决策、`trace.jsonl` 和 `result.json`。Codex 原始工作目录保留并写入结果的 `agent_workspace`；没有自动删除日志。失败包含已完成轮次和已取得的 usage；在模型直接报错且没有返回 usage 时，成本信息以其原始 trace 为准。

主指标 `best_so_far_auc` 是每次新增结果后最佳 utility 的均值；`normalized_best_so_far_auc` 用完整候选池 min/max 归一到 0–1，仅由评测器计算。汇总按同 task/seed 配对，并列出失败数；不会把失败填成基线成绩。小 seed 的接线结果不作为论文效果结论。

```bash
# Mac 的临时路径用真实目录，避免原归档工具误拒绝 /var 系统链接。
TMPDIR=/private/tmp uv run pytest tests/test_harness_*.py
```

验证包含隐藏标签不改变公共输入、预算和指标手算、旧 CARE 选点序列回归、冻结校准记录与执行参数、Codex 假进程端到端，以及官方 EvE 评估协议。没有增加 CI，也没有改动旧历史归档。

2026-09-21 本地验证：原锁文件安装成功，`uv lock --check --offline` 通过；恢复既有回归所需的三个归档后，完整主线测试 **353 passed、1 skipped**（平台相关）。与整理前 router 独立比对的三个种子中，完整轨迹和指标一致。三条 pilot 路线各完成一个种子、三轮 CARE/GP 接线验证；这些缩减预算结果只验证执行。官方 EvE Hydra 配置及本地评估服务真实往返通过，Codex 使用假 CLI 测试。未运行真实模型调用或 EvE 演化搜索；校园镜像在当前网络的临时依赖下载曾遇到 TLS EOF，因此镜像配置方式已提供，但未声称在此网络验证成功。
