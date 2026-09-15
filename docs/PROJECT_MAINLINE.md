# CARE 2.0 项目主线与研究地图

本文依据 2026-09-15 核对的全部 7 个远程分支及 Git 历史整理。本分支的研究基线和截止点是 **`4c6f19f8cf62b25558f966604ce0b7644578f889`（2026-09-01，新增官方 AstaBench HMS 评分）**。远程分支的后续提交仅用于说明历史关系，不代表已纳入本分支；实验结论以截止点以内的冻结协议和原始记录为准。

## 1. 项目到底在解决什么问题

CARE 2.0 的主问题是：**把已完成实验中的证据编译成可执行的策略，在新任务中有条件地复用；当证据不支持迁移时，退回可靠的目标任务基线，并留下可核查的决策记录。**

当前最清楚的主线实现是有限候选池中的 source-outcome transfer：固定源实验历史，利用公开的目标描述符和冻结的 LLM 输出构造先验，再用已经揭示的目标观测校准。它覆盖反应优化、分子性质和材料性质的离线回放。

项目已有明确的主方法定义：[CARE2_METHOD.md](../experiments/care_replay/CARE2_METHOD.md)。整理工作沿用这份定义，不把每个新实验都改称 CARE 2.0 的主算法。`run_care2.py` 是该定义对应的公开入口。

## 2. 全部分支的关系

下表保留首次整理时的完整分支审计。“独有提交”表示该分支相对于当时的 `exploration-aware-llm@c064a0cf` 的独有提交数量；比较依据为 `git rev-list --left-right --count`。这是历史来源表，本分支的截止点仍为 `4c6f19f8`。

| 分支 | 最后提交 | 日期 | 独有提交 | 定位 |
| --- | --- | --- | ---: | --- |
| `main` | `b5aab263` | 2026-06-28 | 0 | 最早公开回放框架和说明，已落后于实际研究进展 |
| `carry1.0` | `b5aab263` | 2026-06-28 | 0 | 与 `main` 完全相同的快照 |
| `carry2.0` | `9c0a25c3` | 2026-07-04 | 0 | 数据扩展、迁移消融、GP 对照、加权核和预算扫描 |
| `codex/target-calibrated-transfer` | `4e89b7db` | 2026-07-11 | 0 | 目标校准、描述符、LLM 规则补丁和显著性审计 |
| `target-calibrated-transfer` | `7072fadd` | 2026-07-15 | 0 | 冻结技能、路由器、在线证据评估 |
| `llm-traces` | `d95253a5` | 2026-07-17 | 1 | 从前一阶段分出，保存 7 月 16 日 live LLM 调用证据 |
| `exploration-aware-llm` | `c064a0cf` | 2026-09-15 | — | 历史开发来源；仅取其中截至 `4c6f19f8` 的研究内容 |

```text
main = carry1.0 (b5aab263)
  └─ carry2.0 (9c0a25c3)
      └─ codex/target-calibrated-transfer (4e89b7db)
          └─ target-calibrated-transfer (7072fadd)
              ├─ llm-traces (d95253a5)
              └─ exploration-aware-llm 的历史
                  └─ 4c6f19f8 (2026-09-01，本分支截止点)
                      ├─ tangchao/repository-reorganization
                      └─ 后续历史 → c064a0cf（不纳入本分支）
```

审计时的 `c064a0cf` 分别比 `main`、`carry2.0`、`codex/target-calibrated-transfer`、`target-calibrated-transfer` 多 169、149、134、126 个提交。前述分支均已包含在其祖先历史中，无需逐个重新合并。这些数量描述远程历史，不表示本分支保留了 9 月 1 日之后的新增功能。

### `llm-traces` 已补保全的内容

独有提交 `d95253a5` 只新增 17 个文件，没有新增独有算法代码。17 个路径在 `c064a0cf` 中均不存在，不能把这条分支当成“已经全部合入”。它们位于原来的 `experiments/care_replay/outputs/`：

- 3 份说明：`API_LIVE_RUN_2026-07-16.md`、`API_LIVE_RUN_2026-07-16_10SEED.md`、`LLM_TRACE_README_2026-07-16.md`。
- `runs/transfer_generalization_live_gpt56sol_20260716*`：两个批次各自的 prompt/response traces、metadata、proposals 和 seed 0 审计记录，共 8 件。
- `tables/transfer_generalization_live_gpt56sol_20260716*`：两个批次各自的 aggregate、comparison 和 runs CSV，共 6 件。

此次整理已将这 17 件文件保存在 [llm-traces-2026-07-16 归档](../artifacts/branch-evidence/llm-traces-2026-07-16.tar.gz)，catalog ID 为 `llm-traces-2026-07-16`。可通过 `tools/artifacts.py` 校验并恢复原路径。

这一资产证明真实 API 调用进入了迁移链路。10-seed 批次包含 60 次成功 LLM 调用、30 组配对回放；各迁移对的 final-best 区间均包含零。因此它是可执行性和可审计性证据，不能升级为稳定的普遍性能优势。归档必须保留正负结果、模型标识、实际调用数量和原始提交出处。

## 3. 主算法的执行链

```text
固定源实验候选与测量结果
  → 提取源角色效应和目标公开描述符的映射
  → 冻结 LLM 生成的数值补丁及其理由
  → 编译带版本和指纹的 TransferSkill
  → 构造初始设计、邻域/加性/交互先验及核几何
  → 每轮只用已揭示目标观测校准专家
  → 将迁移候选与 target-only GP 候选比较，执行在线 gate
  → 校准种子选择迁移或 exact fallback
  → 在独立 held-out seeds 上冻结评价
  → 保存指标、完整审计和紧凑 canonical trace
```

在线 gate 和离线选择是两种不同的检查。在线 gate 使用当时已经揭示的观测；离线校准选择会在额外回放轨迹中揭示目标标签，必须计入评估成本。后者不能直接声称是新湿实验中零成本可用的安全门。

| 责任 | 现有代码路径，相对于 `experiments/care_replay/scripts/` |
| --- | --- |
| 公开入口与多任务调度 | `run_care2.py`、`run_source_outcome_suite.py` |
| 单迁移对的规范算法与校准选择 | `run_calibrated_source_outcome_transfer.py` |
| 可执行技能、证据边界、指纹和紧凑轨迹 | `transfer_skill.py` |
| 源测量历史和角色证据 | `run_transfer_ablation.py` |
| 专家路由、逐轮 gate、目标候选选择 | `run_llm_transfer_router.py` |
| 目标 GP、加权核和基线 | `run_surrogate_baselines.py`、`run_transfer_weighted_kernel.py` |
| 数据适配、任务对象和通用回放基础 | `run_synthetic_suzuki.py` |
| LLM 补丁、语义技能、策略路由 | `run_llm_kernel_skill_evolution.py`、`llm_semantic_skills.py`、`cross_task_router.py` |

这些文件相互导入。`run_synthetic_suzuki.py` 等名称带有历史痕迹，但已经承担真实数据和核心回放依赖，不能仅凭名字将它们移进“废弃实验”。先保留执行路径、建立模块地图和验证入口，再逐步做包结构重构。

## 4. 研究演进与各条线的边界

| 阶段 | 形成的能力 | 代表提交/入口 | 在整理后的位置 |
| --- | --- | --- | --- |
| 6 月：最小 CARE 回路 | incumbent/challenger/gate、合成和公开候选池、审计和 KB 原型 | `b5aab263`，`run_synthetic_suzuki.py` | 通用回放基础；合成数据用于 smoke |
| 7 月初：可比较的迁移 | 角色和描述符映射、GP-UCB/EI 对照、加权核、目标校准 | `9c0a25c3`、`4e89b7db`、`7072fadd` | 基线与迁移组件 |
| 7 月中下旬：冻结 LLM 技能 | exploration-aware 策略、语义技能、source-schema 和 source-outcome 对照 | `2af3f67d`、`c1b976ec`、`89dfdeeb`、`c745337d` | 主线方法和确认性回放协议 |
| 8 月：主方法契约和机制检查 | TransferSkill、warmstart/data-only 控制、经典 transfer BO | `8ad217ee`、`8a80f6e1`、`03ebcccf` | 主线接口、消融和复现证据 |
| 8 月：跨任务技能复用 | 多源 Reizman/ Baumgartner、SkillBank、ask/tell、source-only abstention | `f0b38dd3`、`d740fe28`，`skill_bank.py`、`wetlab_protocol.py` | 面向新任务部署的独立评估轨道 |
| 8 月：在线 scientist | 初始假设、proposer/critic、逐轮选择、受限 authority | `run_online_llm_scientist.py`、`run_online_llm_suite.py` | 探索模块，单独报告调用成本和实际动作影响 |
| 8—9 月：外部 agent 评估 | 冻结 AstaBench solver 和官方 HMS 评分 | `3234e78b`、`4c6f19f8`，`experiments/astabench/` | 外部基准适配，不改变有限池主方法定义 |

### 三种“技能”应明确区分

1. **Pair-local TransferSkill**：一对源/目标任务的可执行策略，包含参数、gate、证据和 hash；本身不证明持续累积能力。
2. **Persistent SkillBank**：多任务证据、适用条件、拒绝条件和来源索引，位于 `skill_banks/`，由 `skill_bank.py` 等执行；必须维持 development/evaluation task 分离。
3. **知识库卡片**：`knowledge_base/seed_cards.json`、`generated_cards/` 及检索/摄取代码，是可检索知识和研究结论的记录层。它们不等同于已经通过部署验证的策略。

## 5. 证据层级和必须保留的负结果

| 证据层级 | 能回答的问题 | 不能据此声称 |
| --- | --- | --- |
| 合成 smoke / 单元测试 | 接口、编译、gate 和审计是否工作 | 真实科学任务效果 |
| 公开测量数据上的有限池回放 | 固定候选池、预算和起点下的配对差异 | 新进行的湿实验、任意任务泛化 |
| 先校准后冻结、held-out seeds | 声明迁移对上冻结确定性策略的回放表现 | 独立 LLM 生成的稳定性、新任务泛化 |
| Task-disjoint / prospective 协议 | 已声明任务边界下的迁移和拒绝能力 | 所有任务族均能迁移 |
| 独立 live LLM 生成重复 | 随机模型调用、动作改变、成本和失败率 | 把许多 replay seeds 当成许多独立模型样本 |

当前需显著展示，而不是在清理时删除的结果：

- **迁移会失败。** 7 月的 source-schema/source-outcome 档案保留了被拒绝路线和 target-only exact fallback。主方法应同时展示 raw transfer 和部署后的选择结果，避免只展示经过选择的赢家。
- **在线 gate 的回溯改善没有自动泛化。** [整理前 README](history/README-before-reorganization.md) 记录 round-three、MAE threshold 5 gate 在 11 路线回溯中改善均值，但在 6 条 route-disjoint 路线上，使相对 target GP 的均值由 `-1.0558` 变为 `-1.9677`。相关原始证据在 `2026-08-24-online-llm-route-disjoint-fixed-threshold5-gate-audit-v1`。
- **在线参与不等于实际贡献。** 六路线 one-round bounded-authority live pilot 中，6 个 proposer/critic 决策都选择 GP-UCB rank 1，在线增量 AUC 为 `0.0000`，共用了 `99,243` tokens。这一模块需要实际动作改变和受益证据。

## 6. 文件整理原则：JSON 不都是日志

保留源代码、测试、协议配置、可执行冻结技能、数据 schema、数据来源和知识卡片；把批量日志、逐 seed 记录、大型结果快照、演示文件放入带索引与校验的归档。归档后的原始路径应可还原，论文/报告的指标必须能追溯到原始证据。

### 主线运行的冻结输入

`configs/source_outcome_benchmark.json` 的 `llm_record` 位于 `configs/source_outcome_llm/`。此次整理已将其依赖的以下 5 件 `target_llm_record` 从历史 results 提取到 `skill_banks/frozen_records/`，并更新主线配置中的路径：

```text
skill_banks/frozen_records/
  molecular_esol_to_lipophilicity_target_only.json
  molecular_lipophilicity_to_freesolv_target_only.json
  materials_dielectric_shared_target_only.json
  reaction_chemlex_to_buchwald_hartwig_target_only.json
  materials_target_only.json
```

这些冻结模型记录由主方法读取和校验 hash，不是调试日志。文件逐字节保留了原始 Git blob，来源提交、原路径和 SHA-256 记录在 [PROVENANCE.md](../experiments/care_replay/skill_banks/frozen_records/PROVENANCE.md)。因此默认主线 suite 无需恢复历史输出即可读取这 5 份冻结输入。历史 `results/` 本身通过 `artifacts/` 归档按需恢复。

截止点以内的 configs 仍有 `results/` 或 `outputs/` 路径引用，包含文件及目录，不全部都是直接运行依赖。主要分为：

- source-outcome 确认所需的 target-only frozen records；
- online controller 复用的 `initial_record.json`；
- 从已有 summary/aggregate 重建的报告和审计；
- 实验结果目录引用。

必须按“执行输入 / 证据引用 / 输出位置”分类，而不能按 `.json` 后缀统一打包后直接删除。历史 snapshot 内的 config、模型响应和 hash 也应保持原字节，避免为了新目录而改写原始证据。

### 可重建文件与知识资产

- `outputs/runs/`、`outputs/tables/` 是旧 runner 默认生成位置，可在新运行时重建；历史文件需先完整归档。
- `knowledge_base/seed_cards.json`、`generated_cards/` 和 `skill_banks/` 是知识输入，不应被日志忽略规则误伤。
- PPT、讲稿、LaTeX 和稿件定位属于传播与论文资产，需有独立索引；它们不是算法入口，也不应该决定代码模块的归属。
- 归档的 README、配置、原始 prompt/response、运行参数、失败尝试、逐 seed 指标、hash 和数据 provenance 共同构成证据，不能只留一张汇总表。

## 7. 阅读、运行与验证入口

推荐阅读顺序：本文件 → [主方法契约](../experiments/care_replay/CARE2_METHOD.md) → `run_care2.py` → `run_calibrated_source_outcome_transfer.py` → 声明协议及其结果归档。针对湿实验部署继续阅读 `MULTISOURCE_WETLAB_PROTOCOL.md`、`BAUMGARTNER_WARMSTART_PROTOCOL.md` 和 SkillBank。

以下命令均从仓库根目录运行；先用 `uv sync --locked` 安装 Python 3.12 的锁定环境。项目要求 Python 3.11+，主项目和 AstaBench workspace 成员共用根 `uv.lock`。可选描述符、语义检索和外部评测的安装方式见 [环境与依赖](ENVIRONMENT.md)。以下是维护时的分层检查入口，不代表本文已经执行全部实验。

```bash
# 无模型调用的最小合成回放
uv run --locked python care.py smoke

# 主线参数入口，不启动外部模型
uv run --locked python care.py pair --help
uv run --locked python care.py suite --help

# 不依赖真实数据的技能契约测试
uv run --locked pytest experiments/care_replay/tests/test_transfer_skill.py

# SkillBank 和 ask/tell 契约：需要 data/raw 中的 Reizman 数据
uv run --locked pytest experiments/care_replay/tests/test_skill_bank.py experiments/care_replay/tests/test_wetlab_protocol.py
```

更完整的主线回归应覆盖 `test_source_outcome_transfer.py`、`test_calibrated_frozen_llm_selector.py`、`test_cross_task_router.py` 和 `test_classical_transfer_baselines.py`。其中部分测试读取公开真实数据，需要先具备对应 data/raw。涉及结果重建或在线模型的测试应按模块单独运行，不能与离线 smoke 混为同一验收结论。

正式 suite 使用 `configs/source_outcome_benchmark.json` 和已恢复/保留的冻结输入；完整的校准及 held-out seed 数、initial/reveal 预算由协议决定。smoke 的缩减预算只能验证可执行性，不用于替代论文结果。

## 8. 整理后的主线判断

项目并非 7 套互不兼容的实现，而是**一条长期增长的回放与迁移主线，加一条已在此次整理中归档保全的早期 trace 证据分支**。整理前的混乱主要来自：根 README 停留在较早叙事、通用库放在历史命名的 scripts 中、冻结运行输入与批量输出混放、论文资料与代码同处根目录，以及技能编译与在线选择的结论没有统一导航。

本分支以 2026-09-01 为清晰截止点，明确一个 canonical method，保留在线 LLM、零经验初始化、AstaBench 和知识库等已有轨道，归档该范围内的证据并保留运行依赖。后续若做 Python 包结构重构，应逐模块迁移并保留兼容入口，让目录清理与算法变更分别可审阅。
