# CARE 2.0 实验进展汇总

版本日期：2026-07-03

用途：团队内部同步；可直接复制到飞书文档继续编辑。

## 0. 先说结论

这轮工作主要是在验证 CARE 2.0 能不能从单一反应优化系统，往“通用新物质发现平台”推进。现在的结论可以分成三层：

第一，平台接口已经跑通。我们把反应 HTE、分子性质、材料性质、ChemLex 形态任务都统一成了 offline finite-pool replay。每个任务都可以用同一套流程跑：`incumbent baseline -> challenger/skill -> gate -> audit log -> metrics`。

第二，transfer 已经有比较清楚的正例。最新 50-seed server sweep 里，有两组结果最值得讲：

| Transfer 方向 | 最强模式 | Final best 提升 | AUC 提升 | Top-10 hit 变化 | 读法 |
| --- | --- | ---: | ---: | ---: | --- |
| FreeSolv -> Lipophilicity | `transfer_value_prior_gate_v1` | +2.7550 | +2.4000 | 4% -> 30% | 共享分子 descriptor 空间下，transfer 优势最明显 |
| Suzuki-Miyaura -> Buchwald-Hartwig | `transfer_gate_v1` | +2.4389 | +1.1066 | 16% -> 24% | 反应 HTE 中，role-level transfer 有稳定正向收益 |

第三，系统还不是“所有任务都提升”。BH -> Suzuki 这类反向迁移目前仍不稳定；Matbench 上当前 composition-only observation model 不够强；LLM audit 已经能接入并输出可解析结果，但目前偏保守。这个边界很重要：CARE 2.0 不是盲目迁移 source knowledge，而是要学会什么时候迁移、迁移多少、什么时候由 gate 拦下来。

## 1. 实验框架

我们把每个数据集都改写成一个有限候选池搜索问题。数据集中已经有候选点和真实结果，但 replay 过程中系统不能提前看未选择候选的目标值。每一轮只能基于已经 reveal 的 observations 和 public features 选择下一个候选点，然后再 reveal 真实结果。

核心角色如下：

| 组件 | 作用 |
| --- | --- |
| `no_care_random` | 不使用 CARE，随机选择未观测候选，是最低基线 |
| `incumbent` | 只基于已公开观测做稳健选择，是主要 target-only baseline |
| `challenger / skill` | 根据 factor evidence、skill prior 或 transfer card 调整候选排序 |
| `gate` | 审查 challenger 是否可以覆盖 incumbent，控制风险 |
| `audit log` | 记录每轮选择、gate certificate、hypothesis snapshot 和 LLM 响应 |

主要指标：

| 指标 | 含义 |
| --- | --- |
| `final_best` | replay 结束时找到的最好结果 |
| `best_so_far_auc` | best-so-far 曲线面积；越高说明越早找到好点 |
| `simple_regret` | 距离全局最优还有多远 |
| `top10_hit` | 是否进入全局 top 10% 区域 |
| `intervention_count` | gate 授权 challenger 覆盖 incumbent 的次数 |
| `bad_intervention_count` | intervention 后结果比 incumbent 差的次数 |
| `llm_call_count` | 实际 LLM 调用次数 |
| `llm_parse_error_count` | LLM 输出解析失败次数 |

## 2. 数据集覆盖情况

目前已经接入的数据集/任务如下：

| 数据集 | 类型 | 候选数 | 作用 |
| --- | ---: | ---: | --- |
| `synthetic_suzuki_i` | synthetic reaction replay | 448 | Suzuki-like smoke test，验证 replay/skill/gate 接口 |
| `synthetic_chemlex_i` | synthetic ChemLex-style replay | 1728 | 验证 acid-amine/ChemLex 任务形态 |
| `synthetic_materials_i` | synthetic materials replay | 336 | 验证材料配方/工艺任务形态 |
| `real_buchwald_hartwig` | public real HTE | 3955 | Dreher-Doyle Buchwald-Hartwig yield replay |
| `real_suzuki_miyaura` | public real HTE | 5760 | Perera Suzuki-Miyaura yield replay |
| `real_chemlex_acidamine` | real wetlab ChemLex | 11669 | ChemLex Acid-Amine wetlab conversion replay |
| `real_moleculenet_esol` | molecular property | 1128 | ESOL solubility finite-pool replay |
| `real_moleculenet_freesolv` | molecular property | 642 | FreeSolv hydration free energy replay |
| `real_moleculenet_lipophilicity` | molecular property | 4200 | Lipophilicity replay |
| `real_matbench_expt_gap` | materials property | 4604 | Matbench experimental band gap replay |

## 3. 第一批实验：基础 replay 和 smoke test

这批实验的目标不是证明科学性能，而是确认系统能在不同任务形态上稳定跑通。

设置：

- Seeds：30
- Initial observations：5
- Reveal budget：10
- Modes：`no_care_random`, `incumbent`, `no_gate`, `gate_v1`, `gate_v2`

关键结果：

| 数据集 | 最好 CARE2 模式 | Final best 相比 incumbent | AUC 相比 incumbent | 结论 |
| --- | --- | ---: | ---: | --- |
| synthetic_suzuki_i | `gate_v1/gate_v2` | +4.5563 | +6.1074 | positive synthetic control |
| synthetic_chemlex_i | `gate_v1/gate_v2` | +6.9028 | +9.7056 | ChemLex 形态任务可被 skill/gate 利用 |
| synthetic_materials_i | `gate_v1/gate_v2` | +2.4809 | +3.5795 | 材料形态接口可用 |
| real_buchwald_hartwig | `gate_v1/gate_v2` | +0.0000 | +0.0000 | 单任务保守 gate 基本持平 |
| real_suzuki_miyaura | `gate_v1/gate_v2` | +0.0000 | -0.0886 | 单任务保守 gate 略弱 |
| real_moleculenet_esol | `gate_v1` | +0.4262 | +0.0640 | 小幅正信号 |
| real_moleculenet_freesolv | `no_gate` | +0.3511 | +0.0352 | 有 challenger 信号，但 gate 较保守 |
| real_moleculenet_lipophilicity | `gate_v1/gate_v2` | -0.0584 | -0.0058 | 单任务下略负 |
| real_matbench_expt_gap | `no_gate` | +0.3250 | +0.0650 | 诊断结果，random baseline 反而更强 |

解读：

这一批证明的是“平台接口可迁移”，不是“所有真实任务都已经提升”。Synthetic 任务提升明显，真实 HTE 单任务基本中性，MoleculeNet 有小信号，Matbench 暴露出 observation model 太弱的问题。

## 4. 材料方向实验

材料方向分两步：先跑 synthetic materials，再接真实 Matbench experimental band gap。

### 4.1 Synthetic materials

| Mode | Final best | AUC | Top-10 hit | Bad interventions |
| --- | ---: | ---: | ---: | ---: |
| `no_care_random` | 90.2415 | 88.4020 | 0.4333 | 0.0000 |
| `incumbent` | 93.3509 | 90.4718 | 0.7000 | 0.0000 |
| `gate_v1` | 95.8318 | 94.0513 | 0.9333 | 0.7000 |

结论：材料形态的 finite-pool replay、skill adjustment、gate、audit 都能工作。

### 4.2 Real Matbench experimental band gap

| Mode | Final best | AUC | Top-10 hit | Bad interventions |
| --- | ---: | ---: | ---: | ---: |
| `no_care_random` | 48.7417 | 43.7104 | 0.0000 | 0.0000 |
| `incumbent` | 44.8000 | 41.3633 | 0.0000 | 0.0000 |
| `no_gate` | 45.1250 | 41.4283 | 0.0000 | 0.0667 |
| `gate_v1` | 44.8000 | 41.3633 | 0.0000 | 0.0333 |

结论：真实材料数据能跑，但当前只用 composition bin 的 public observation model 不够强。这个结果应该当作诊断，不应该包装成 CARE 已经提升真实材料发现。

下一步材料方向应该补 matminer-style descriptors、composition embeddings 或 pretrained materials surrogate。

## 5. 真实 ChemLex 实验

我们接入了 ChemLex Acid-Amine wetlab updated record。这个数据是真实 wetlab conversion，不是 synthetic proxy。

### 5.1 Non-LLM ChemLex replay

| Mode | Final best | AUC | Top-10 hit | Bad interventions |
| --- | ---: | ---: | ---: | ---: |
| `no_care_random` | 90.3810 | 82.8549 | 0.0333 | 0.0000 |
| `incumbent` | 82.0203 | 75.7267 | 0.0000 | 0.0000 |
| `gate_v1` | 82.1300 | 76.0638 | 0.0000 | 0.1333 |

### 5.2 LLM ChemLex replay

| Mode | Final best | AUC | Top-10 hit | LLM calls | Bad interventions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `no_care_random` | 89.6800 | 72.7893 | 0.0000 | 0.0000 | 0.0000 |
| `incumbent` | 72.0440 | 70.1367 | 0.0000 | 0.0000 | 0.0000 |
| `llm_no_gate` | 72.0440 | 70.1367 | 0.0000 | 3.0000 | 0.2000 |
| `llm_gate_v1` | 72.0440 | 70.1367 | 0.0000 | 3.0000 | 0.2000 |

解读：

ChemLex 真实数据现在能跑，但当前 policy 不强，random baseline 在这组设置里很高。这说明真实 ChemLex 需要单独分析数据 split、候选分布和 observation model，不能直接拿 synthetic ChemLex 的强提升来讲真实 wetlab 提升。

## 6. LLM-in-the-loop 实验

我们做了两类 LLM 实验：一类是让 LLM 直接给 factor-level adjustment；另一类是让 LLM 做 transfer gate 的 auditor。

### 6.1 九数据集 LLM generalization sweep

设置：

- Seeds：5
- Rounds：6
- LLM endpoint：OpenAI-compatible CommonStack endpoint
- Modes：`no_care_random`, `incumbent`, `llm_no_gate`, `llm_gate_v1`

代表结果：

| 数据集 | LLM gate final 相比 incumbent | LLM gate AUC 相比 incumbent | 读法 |
| --- | ---: | ---: | --- |
| real_buchwald_hartwig | +0.1850 | +0.0616 | 小幅正向 |
| real_matbench_expt_gap | +2.8000 | +0.9833 | LLM 能提出有意义调整，但整体材料模型仍弱 |
| real_moleculenet_lipophilicity | +1.4000 | +0.6875 | 分子性质上有小正信号 |
| real_suzuki_miyaura | +0.0000 | +0.0000 | 持平 |
| synthetic_materials_i | +0.1165 | +0.0522 | 小幅正向 |

结论：LLM 路径已经接通，audit log 能记录 raw response、parsed policy、token usage、parse error 等。但直接让 LLM 生成 factor adjustment 不稳定，还不是主力结果。

### 6.2 LLM transfer proposal

BH -> Suzuki 的 LLM transfer 5-seed 结果：

| Mode | Final best | AUC | LLM calls | Bad interventions | 读法 |
| --- | ---: | ---: | ---: | ---: | --- |
| `incumbent` | 89.7358 | 86.6963 | 0.0000 | 0.0000 | target-only baseline |
| `transfer_strict_gate_v1` | 89.7358 | 86.6963 | 0.0000 | 0.4000 | deterministic strict transfer |
| `llm_transfer_gate_v1` | 87.2325 | 85.1085 | 7.0000 | 2.8000 | LLM proposal 太宽，伤害较多 |
| `llm_transfer_strict_gate_v1` | 87.9673 | 85.1761 | 7.0000 | 1.4000 | strict 后更安全但仍弱 |

结论：LLM 作为直接 proposer 目前比 deterministic strict gate 更噪。后续更适合让 LLM 做 auditor、解释器或 reranker，而不是直接接管 challenger。

### 6.3 LLM audit transfer

BH -> Suzuki 的 LLM-audited transfer 5-seed 结果：

| Mode | Final best | AUC | LLM calls | Bad interventions | 读法 |
| --- | ---: | ---: | ---: | ---: | --- |
| `incumbent` | 89.7358 | 86.6963 | 0.0000 | 0.0000 | target-only baseline |
| `transfer_gate_v1` | 87.6407 | 85.1042 | 0.0000 | 1.4000 | plain transfer 有负迁移 |
| `llm_audit_transfer_gate_v1` | 89.7358 | 86.6963 | 3.2000 | 0.0000 | LLM audit 拦掉坏 transfer，恢复 baseline |
| `llm_audit_transfer_strict_gate_v1` | 89.7358 | 86.6963 | 0.6000 | 0.0000 | 更安全，但偏保守 |

结论：LLM auditor 已经能真实调用且无 parse error，能作为 safety filter，但现在过于保守，还没有保留正向 transfer gain。

## 7. Transfer 实验主线

Transfer 是 CARE 2.0 当前最重要的实验方向。我们按几个阶段推进。

### 7.1 BH -> Suzuki：第一版 transfer，负向诊断

第一版用 BH source evidence 构建 transfer card，再迁移到 Suzuki target。

| Mode | Final best | AUC | Bad interventions | 读法 |
| --- | ---: | ---: | ---: | --- |
| `incumbent` | 92.6085 | 87.5533 | 0.0000 | target-only baseline |
| `transfer_gate_v1` | 91.3879 | 86.4187 | 1.6333 | transfer 太宽，低于 incumbent |
| `transfer_plus_local_gate_v1` | 91.6643 | 86.6158 | 1.0333 | 加 local skill 仍低于 incumbent |

结论：这是一个有价值的 negative control。它说明 role-level transfer 不能随便搬，方向和 gate 都很重要。

### 7.2 BH -> Suzuki：server-side strict transfer 小正例

随后在服务器上做了 strict transfer 参数优化。

| Mode | Final best | AUC | Bad interventions | 读法 |
| --- | ---: | ---: | ---: | --- |
| `incumbent` | 92.6085 | 87.5533 | 0.0000 | baseline |
| `transfer_gate_v1` | 91.3879 | 86.4187 | 1.6333 | plain transfer 仍伤害 |
| `transfer_strict_gate_v1` | 92.8770 | 87.7002 | 0.8333 | strict 后出现小幅正收益 |
| `transfer_strict_plus_local_gate_v1` | 92.8273 | 87.5630 | 0.8667 | 与 strict 接近 |

结论：BH -> Suzuki 不是完全不可能，但需要 stricter transfer gate。过窄的 `min_positive_roles = 3` 又会掉下去，说明 gate calibration 很关键。

### 7.3 Multi-domain transfer feasibility

我们把 transfer role map 扩展到多个方向：

- reaction HTE：BH, Suzuki, ChemLex
- molecular property：ESOL, FreeSolv, Lipophilicity
- proxy：synthetic ChemLex -> real ChemLex, synthetic materials -> Matbench

代表结果：

| Source -> Target | Seeds | Best transfer mode | Final delta | AUC delta | 读法 |
| --- | ---: | --- | ---: | ---: | --- |
| Suzuki -> Buchwald-Hartwig | 30 | `transfer_gate_v1` | +2.9907 | +0.8135 | 第一条强反应 transfer 正例 |
| Suzuki -> ChemLex | 10 | `transfer_strict_gate_v1` | +4.4380 | +0.4367 | 有正信号，但 random baseline 很强 |
| ChemLex -> Buchwald-Hartwig | 10 | `transfer_no_gate` | +1.4691 | -0.2507 | final gain 有，AUC 不稳 |
| FreeSolv -> Lipophilicity | 20 | `transfer_strict_gate_v1` | +0.3499 | +0.0556 | 分子 transfer 小正信号 |
| ESOL -> Lipophilicity | 20 | `transfer_gate_v1` | +0.4062 | +0.1193 | 分子 transfer 小正信号 |
| BH -> Suzuki | 10 | `transfer_gate_v1` | -1.3260 | -0.8504 | negative direction |

结论：transfer 不是单一 pair 的特殊现象，但方向性很强。正向结果和负向结果都要保留，因为它们一起说明系统在区分“可迁移”和“不可迁移”。

### 7.4 最新 transfer advantage sweep：明星结果

这轮是目前最值得重点展示的结果。我们新增了 `transfer_value_prior_*` 模式：当 source 和 target 共享同一套 public descriptor vocabulary 时，允许迁移保守 value-level prior。这个模式只用于共享描述符空间，例如 MoleculeNet 的 SMILES-derived bins；不会用于 HTE 的 dataset-local label。

#### FreeSolv -> Lipophilicity

| Mode | Final best | Delta final | AUC | Delta AUC | Top-10 hit | Bad interventions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_care_random` | 85.7650 | -1.5425 | 84.2917 | -1.2743 | 0.0200 | 0.0000 |
| `incumbent` | 87.3075 | 0.0000 | 85.5660 | 0.0000 | 0.0400 | 0.0000 |
| `transfer_value_prior_gate_v1` | 90.0625 | +2.7550 | 87.9660 | +2.4000 | 0.3000 | 3.4600 |
| `transfer_value_prior_strict_gate_v1` | 88.1625 | +0.8550 | 86.1820 | +0.6160 | 0.1600 | 1.9000 |

读法：这是当前最强的 transfer advantage。它不只是 final best 更高，AUC 也明显更高，top-10 hit 从 4% 提到 30%。这说明 transfer prior 让 replay 更早进入目标任务的高价值区域。

#### Suzuki -> Buchwald-Hartwig

| Mode | Final best | Delta final | AUC | Delta AUC | Top-10 hit | Bad interventions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_care_random` | 82.1728 | -4.4749 | 77.4347 | -2.5366 | 0.0600 | 0.0000 |
| `incumbent` | 86.6477 | 0.0000 | 79.9713 | 0.0000 | 0.1600 | 0.0000 |
| `transfer_gate_v1` | 89.0866 | +2.4389 | 81.0779 | +1.1066 | 0.2400 | 0.8600 |
| `transfer_strict_gate_v1` | 87.6158 | +0.9681 | 80.2528 | +0.2815 | 0.1600 | 0.1800 |

读法：这是反应 HTE 方向最稳的正例。即使不直接迁移具体 factor values，只迁移 role-level evidence，也能比 target-only incumbent 更好。

## 8. 知识库和 embedding 原型

除了 replay 实验，我们还搭了 CARE 2.0 知识库原型。这个知识库不是会议纪要，而是 CARE 系统用的 structured memory，用来沉淀 task、dataset、mechanism、skill、hypothesis。

已经做的部分：

| 模块 | 作用 |
| --- | --- |
| `seed_cards.json` | 初始 task/dataset/skill/mechanism cards |
| SQLite + FTS | 支持关键词检索 |
| `build_embeddings.py` | 支持本地 hashed embedding，也预留 OpenAI-compatible embedding endpoint |
| `query_embeddings.py` | 支持语义检索 |
| `awesome_resources.md` | 整理 AI4Chem、LLM4EDA、materials-aware LLM、molecular discovery 资源 |

这部分和实验的关系是：之后每次实验产生的成功/失败 hypothesis、transfer card、gate decision、可复用 skill 都应该沉淀到知识库里。CARE 2.0 要变成通用平台，不能只靠一次性脚本跑实验，而要能积累 reusable skills。

## 9. 当前总体判断

现在可以比较稳地这么讲：

1. CARE 2.0 的 replay platform 已经跑通，覆盖 synthetic reaction、real HTE、real molecular property、real materials property、real ChemLex。
2. Transfer 已经有两条真实数据正例：`FreeSolv -> Lipophilicity` 和 `Suzuki -> Buchwald-Hartwig`。
3. 分子共享 descriptor 空间下，value-prior transfer 的优势最明显；反应 HTE 下，role-level transfer 也能产生稳定 gain。
4. LLM 路径已经真实调用并可审计，但目前更适合做 auditor / critic，而不是直接 proposer。
5. 材料方向和 ChemLex 方向已经接入，但要出强结果还需要更强 observation model 和更好的数据-specific analysis。
6. Gate calibration 是下一步核心：我们要保留 transfer 的优势，同时降低 bad interventions。

## 10. 下一步建议

按优先级，建议接下来做四件事。

### 10.1 继续做 gate calibration

目标是把 `transfer_value_prior_gate_v1` 的大优势保留下来，同时降低 bad interventions。可以考虑：

- 要求 target-side confirmation 后再授权强 transfer。
- 对 source prior 做 uncertainty-aware weighting。
- 把 LLM audit 改成结构化 checklist，而不是简单 approve/reject。
- 针对不同 transfer 类型设置不同 gate：role-level HTE transfer 和 shared-descriptor molecular transfer 不应该用完全同一套阈值。

### 10.2 补真实数据和更强 observation model

重点数据：

- ChemLex：继续分析 wetlab split 和 candidate 分布。
- Pfizer zero-inflated reaction data：如果能拿到，是很好的 CARE 1.0/2.0 连接点。
- Matbench / Materials Project：材料方向需要 descriptors 或 surrogate，不然现在的 composition bin 太弱。

### 10.3 强化 LLM 的位置

现在不建议让 LLM 直接决定实验点。更合理的位置是：

- 给 transfer card 做解释和风险审查；
- 对 candidate shortlist 做 reranking；
- 根据 audit log 总结失败原因；
- 把成功/失败经验写回 skill library。

### 10.4 把实验结果沉淀进知识库

建议把以下内容结构化进入 CARE 2.0 knowledge base：

- 每个 dataset 的 task card；
- 每次 transfer 的 source-target role map；
- 成功 transfer cases；
- negative transfer cases；
- gate rejected / approved 的典型 audit examples；
- 可复用 skill/prior 及其适用边界。

## 11. GitHub 中对应文件

主要文件位置：

| 内容 | 路径 |
| --- | --- |
| 总览 | `overview.md` |
| replay 主脚本 | `experiments/care_replay/scripts/run_synthetic_suzuki.py` |
| transfer 主脚本 | `experiments/care_replay/scripts/run_transfer_ablation.py` |
| 50-seed transfer advantage | `experiments/care_replay/results/2026-07-03-transfer-advantage-sweep/` |
| multi-domain transfer feasibility | `experiments/care_replay/results/2026-07-03-multidomain-transfer-feasibility/` |
| LLM audit transfer | `experiments/care_replay/results/2026-07-03-llm-audit-transfer-5seed/` |
| LLM generalization sweep | `experiments/care_replay/results/2026-06-30-llm-commonstack-5seed/` |
| real ChemLex | `experiments/care_replay/results/2026-06-30-real-chemlex/` |
| materials baseline | `experiments/care_replay/results/2026-06-29-materials-baselines/` |
| knowledge base | `knowledge_base/` |

## 12. 给团队同步时可以用的一段话

我们现在不是只把 CARE 1.0 的单个实验复现了一遍，而是在搭 CARE 2.0 的跨领域 replay 和 transfer 框架。所有任务都被统一成 finite-pool search，然后用同一套 incumbent、challenger、gate 和 audit 机制做比较。最关键的新结果是两条真实数据 transfer：FreeSolv 到 Lipophilicity 在 50 seeds 下 final best 提升 2.755、AUC 提升 2.400，top-10 hit 从 4% 到 30%；Suzuki 到 Buchwald-Hartwig 也在 50 seeds 下 final best 提升 2.439、AUC 提升 1.107。这说明 transfer 不是只有概念，已经能在合适的 source-target pair 上带来真实收益。下一步主要不是继续堆 synthetic，而是做 gate calibration、补真实数据、让 LLM 做更可靠的 audit/rerank，并把成功和失败的 transfer cases 沉淀进 CARE 2.0 知识库。
