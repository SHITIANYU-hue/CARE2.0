# CARE 2.0 汇报讲稿

配套文件：`CARE2.0_Nature_evidence_update_2026-08-25_v7.pptx`

## 第 1 页：项目定位

CARE 2.0 研究的是旧实验经验能否帮助新任务更快找到好条件。当前最清楚的主线不是让大模型长期接管数值优化，而是让它把 source evidence 和 target 的公开 schema 编译成可检查、可执行、可拒绝的迁移 skill。Strict compiler 检查字段和值，calibration split 决定 skill 是否部署；冻结后由 target optimizer 在真实预算内执行。这样，LLM 提供语义和假设，GP 提供稳定的数值优化，失败 skill 也会留下完整审计记录。在线 proposer/critic 仍然存在，但首批真实 pilot 没有改变 GP rank 1，因此目前是探索模块，不是论文主贡献。

## 第 2 页：目前最准确的结果

冻结协议下完成的第一批证据覆盖 11 条真实数据迁移路线，包括反应优化、分子性质和材料任务。在线 LLM Scientist 相对同开局、同预算的 target-only GP-UCB，平均 best-so-far AUC 增益为 1.75；6 条路线提高、2 条持平、3 条下降。按路线 bootstrap 得到的区间为 0.41 到 3.31，但符号检验并不显著，而且每条路线仍只有一条在线 LLM 轨迹。为了检查路线选择偏差，我们又纳入仓库中 6 条更早的负向或持平路线；完整 17 路线回顾的平均增益降到 0.76，区间为 -0.50 到 2.09。这里的准确结论是：冻结 11 路线组合上有平均正向信号，但所有历史路线合并后仍不能证明普遍提升。

## 第 3 页：问题如何定义

实验被定义为一个有限预算的 sequential decision problem。迁移前，系统只知道 source 历史、target 的变量和候选空间以及固定预算；target 中尚未执行条件的结果始终隐藏。每轮只能选择一个候选，执行后才揭示它的真实结果。评价重点不是模型的解释是否流畅，而是在相同起点、候选空间和实验次数下，它是否比 target-only GP 更早找到高值条件。

## 第 4 页：系统每轮做什么

这页展示的是在线 proposer/critic 控制器。每轮先用 target GP、source prior 和覆盖策略建立受约束的候选菜单；Proposer 更新可证伪假设并选择候选，Critic 可以接受、退回 GP rank 1 或改选。历史完整在线版本允许 LLM 在 10 轮都有最终选择权，但分离路线表现不稳。最新 bounded pilot 只保留第一轮 LLM 决策，后九轮交回 target GP；首批六条真实调用中，LLM 六次都选择 GP rank 1。它输出了结构化判断，却没有产生 action change，因此本模块仍需重新开发。

## 第 5 页：Replay harness

Replay harness 用完整历史数据模拟真实的逐轮实验。虽然磁盘上保存了 target 的全部标签，但控制器在第 t 轮只能读取此前已经选择并揭示的结果；未选择候选不会进入 prompt、GP 训练或候选评分。在线 LLM 和对照组从完全相同的三个 target 观测开始，各有 10 次揭示机会。每轮的输入、候选、模型回答、最终选择和真实结果都写入 trace，所以任何一条结论都能回到原始决策检查。

## 第 6 页：Reward 和系统更新

这里的 reward 不是用来训练 Claude 权重的强化学习奖励，而是衡量搜索效率的指标。第 t 轮的当前最好值记为 `b_t=max(y_1,...,y_t)`，长度为 T 的轨迹得分为 `R=(1/T)Σb_t`。高值条件越早被发现，它会在更多后续轮次中贡献较高的 best，因此 AUC 越大。一个新结果会更新 target 观测历史、GP 后验、候选排序、假设状态和 trace；它不会更新 LLM 参数，也不会自动沉淀成永久 skill。

## 第 7 页：主对照为什么公平

最重要的主对照是 same-initial target-only GP-UCB。它与在线 LLM 使用相同的三个初始观测、候选空间和 10 轮预算，也看不到未执行的 target 标签。唯一差别是后续决策是否使用 source evidence 和 LLM proposer/critic，因此可以隔离在线 LLM 增量。我们先在三个代表性案例中重放了 RGPE 和 multitask GP：分子和材料路线超过所有已跑 baseline，Suzuki 则输给 multisource ICM-BMA。随后把同开局对照扩展到冻结的全部 11 条路线。LLM 相对 target GP 的平均增益是 +1.753，相对固定 RGPE 对照是 +3.229，但相对每条路线事后最强 baseline 的平均值是 -0.550，只取得 3 胜、1 平、7 负。因此，LLM 的优势目前主要是相对 no-transfer 和部分迁移方法，不能说普遍超过最强 optimizer。

## 第 8 页：LLM 的真实输入和输出

每轮输入包括已揭示的 target 历史、当前 best、source 证据、初始假设、GP 的均值和不确定性、expected improvement 以及合格候选菜单。数据集还显式提供原始物理量、单位、replay score 的变换公式和优化方向，但不会提供未执行候选的数值。模型必须结构化输出假设状态、更新后的假设、一个候选 ID、预期结果、改善概率、支持和反对证据、是否继续迁移，以及它与 GP rank 1 的比较。Critic 再做一次独立检查。prompt、原始回答、模型版本、token 用量、候选选择和揭示结果都完整保存，API 密钥不会进入实验文件。

## 第 9 页：FreeSolv 到 Lipophilicity 案例

这是保存下来的真实 trace。该路线的初始 best 为 67.25。第 0 轮的 GP rank 1 是 `lipo_2256`，但 proposer 和 critic 选择了 GP rank 4 的 `lipo_0635`。模型给出的理由是：该候选的 expected improvement 较高，而且卤素丰富、低极性的结构与已观测的高 lipophilicity 分子一致。执行前系统不知道该候选的真实标签；揭示后得到 86.875，第一轮使 best 提高 19.625。整条路线相对同开局 target-only GP 的 AUC 增益为 6.325。这个案例用于说明 LLM 如何覆盖 GP 建议并留下证据链，不用于代表所有路线。

## 第 10 页：Controller 选择稳定性

这页检验的不是哪个阈值在已知结果上最高，而是“一轮 LLM、随后交回 GP-UCB”是否会因为删掉一条开发路线就变成另一套策略。左图使用 11 条开发路线做 leave-one-route-out。每一折先拿掉一条路线，只在剩余 10 条路线上，按照冻结的词典序规则重新比较 115 个 controller：先最小化负迁移路线数，再最大化路线等权平均 AUC 和最差路线 AUC，最后才偏好参数更少、交回更早的策略。11 折中有 10 折重新选中 `bounded_authority_r1`，也就是只允许 LLM 做第一轮实验选择，随后用全部已积累观测交回 target-only GP-UCB；另 1 折选择了两轮 hard-abstention gate。因此，主策略的选择稳定率是 90.9%，不是由单独一条开发路线偶然推出来的。

把每一折重新选出的策略应用到当时被拿掉的那条路线，平均 AUC 增量为 +1.141，结果为 2 胜、8 平、1 负。右图展示完全不参与策略选择的 6 条 route-disjoint 既有轨迹：一轮授权相对同开局 target-only GP-UCB 为 2 胜、3 平、1 负，平均 +0.612 AUC；正向路线是 Lipophilicity 到 FreeSolv 和 phonons 到 perovskites，负向路线是 bulk 到 shear，AUC 差为 -1.326。

这项结果说明，一轮 bounded authority 是一个相对稳定、低复杂度的 controller 选择，也把 full online LLM 在这 6 条路线上的 2 条负迁移减少到 1 条。但它仍不能支持“安全迁移已经解决”：route-bootstrap 区间跨 0，轨迹是回顾性的，而且 bulk 到 shear 仍然失败。下一步必须把这套 controller 完整冻结，在新的 prospective task family 上验证。

## 第 11 页：配对稳定性审计

早期审计调用和第一次冻结重复的总体胜平负都为 6 / 2 / 3，但 11 条配对路线中有 4 条严格翻转正负号：材料 `expt gap→mp gap`、`aniline→phenethylamine/AlPhos`、`aniline→benzamide/tBuXPhos` 和 `aniline→phenethylamine/tBuBrettPhos`。Suzuki 三次观察为 +0.28、+1.38 和 +0.32，方向一致但幅度仍有波动。早期调用发生在确认协议冻结前，因此这不是正式重复检验，而是说明单次随机 LLM trajectory 不能定义一条路线是否稳定可迁移。论文应把本页写成 stochasticity audit，不应把它包装成显著性结果。

## 第 12 页：材料迁移案例

这页展示的是冻结 JSON/JSONL 生成的真实轨迹。`phonons→bulk modulus` 中，LLM 十轮都把 `continue_source_transfer` 设为 false，因为 target 证据不支持把声子峰的结果排序直接搬到体模量。它仍然利用材料组成语义和已揭示的 target 历史，在 4/10 轮选择非 GP rank 1 候选。CARE 最终找到 normalized score 为 77.6127 的 `O8Pt6`，同开局 GP 最终为 73.2886；AUC 增益为 3.240，final 增益为 4.324。这个结果应解释成“负迁移识别后进行 target 自适应”，不能说成 phonon outcome 对 bulk modulus 的正向直接迁移。图中的分数为 `100×log10(K_VRH[GPa])/3`，本身是无量纲归一化分数，不是 GPa。早期 trace 曾有 5 次把这类 replay score 直接写成 GPa；加入 raw quantity、raw unit、score transformation 和 optimization direction 后，新轨迹的 20 个 proposer/critic 响应中该错误为 0。新轨迹相对同开局 GP 的 AUC 增益为 +3.083、final 增益为 +2.746，但两次调用并非配对实验，所以这里能证明的是语义错误被修复，不能用来证明 prompt 修复提高了性能。

## 第 13 页：结论边界

这页重新排列证据层级。当前最扎实的 LLM 证据不是在线选点，而是 frozen semantic-skill compiler：LLM 读取 source evidence 和 target 的公开 schema，生成规则特征、先验方向和 acquisition schedule；strict compiler 检查字段和值，calibration split 决定是否部署，held-out replay 阶段不再调用 LLM。相对 strongest target-only anchor，FreeSolv、Buchwald-Hartwig 和 Matbench experimental band gap 的 AUC 分别提高 +1.384、+2.859 和 +5.930。schedule-only 与 no-prior 的同 seed 消融说明 semantic representation 和 prior direction 都有独立贡献；ChemLex 的负结果被完整保留。相反，首批六条真实在线轨迹全部选择 GP rank 1，在线增量为 0。因此论文主线应聚焦“LLM 把经验编译成可执行 skill，再由校准和 GP 安全执行”；在线 proposer/critic 暂时只能作为待验证的 hypothesis revision 和 abstention 模块。

## 第 14 页：Hidden-label 非干预审计

这一页先回答最基础的审计问题：系统做第 t 轮决策时，会不会因为代码实现而接触到尚未揭示的 target 标签。我们在 17 条真实路线中各抽取 10 个归档决策状态，共检查 170 个状态。每次都保持已经揭示的 target 观测、候选 ID、公开属性、source evidence 和协议参数不变，只对尚未揭示的 target 标签做确定性置换，再用生产代码重建 source prior、candidate menu、menu diagnostics 和完整 LLM prompt。

左图显示每条路线、每一轮实际被置换的隐藏标签数量，颜色使用对数尺度。材料任务候选池最大，单个状态可以改动数十万甚至上百万个隐藏值；反应 HTE 的候选池较小，但同样逐状态执行。右图汇总每条路线的干预规模，并标出所有路线均为 10/10 通过。总计 2,725,689 个 state-level 隐藏标签位置被改动，170/170 个状态的 source prior、候选菜单、诊断量和当前代码生成的 prompt 仍完全相同。归档 prompt 的 SHA-256 也单独验证通过。

需要主动说明一个版本问题：170 个归档状态都早于后来加入的公开 outcome semantics 字段。110 个状态只差这一项；另外 60 个状态来自更早的六条路线，还早于当前 eligibility 和 safety 字段。我们把它记为 code-version drift，而不是把历史 prompt 与当前 prompt 的逐字节重构当作主要判据。真正的主要判据是：对未揭示标签的干预不能改变生产决策状态。

这页不能被讲成“LLM 有效”。它只支持一个实现层面的结论：在已测试代码和路线中，未揭示 target labels 不会进入决策状态。LLM 的科学推理是否正确、source outcome 是否真正贡献因果信息、迁移是否提高搜索效率，仍分别由后续的 source-outcome 置换检验、匹配 baseline 和前瞻性确认回答。

## 第 15 页：Source-outcome 因果检验

这一页专门回答一个更严格的问题：迁移收益究竟来自 source 中真实的“特征—实验结果”对应关系，还是仅仅来自 target schema、候选覆盖或初始化策略。我们冻结 LLM 编译出的 role map、target anchor、候选空间、预算和全部 target seeds，只在 source 内随机打乱 measured outcomes 与 source candidates 的对应关系。每条路线使用 100 个 target seeds 和 19 个 outcome permutations，共执行 300 个路线级任务；并且把 sequential transfer mass 设为 0，使检验只针对 source-informed initial design。

ChemLex 到 Buchwald-Hartwig 的真实绑定相对 target-only AUC 提高 9.672，95% 配对 target-seed 区间为 7.405 到 11.939；相对置换均值提高 20.347，但随机化检验 p=0.15。材料 dielectric 到 experimental band gap 相对 target-only 提高 48.954，区间为 44.050 到 53.857；相对置换均值提高 47.766，p=0.10。Lipophilicity 到 FreeSolv 相对 target-only反而下降 1.027，区间为 -1.228 到 -0.826；相对置换均值提高 0.451，p=0.20。

这里必须区分两类不确定性：图中的区间是在既定置换集合下、跨 target seeds 计算的配对区间，回答“同一组 outcome assignment 在不同 target 初始种子下是否稳定”；随机化 p 值则在 source outcome assignments 之间比较真实绑定的排名，回答“真实绑定是否优于随机绑定”。因此，即使某个配对区间很窄，也不等于通过了 source-outcome 因果检验。三条路线的 p 值都大于 0.05，所以当前只能说反应和材料 warm start 相对 target-only 有稳定收益，不能声称系统已经证明学到了正确的 source feature-outcome association。分子路线对 target-only 仍为负。这是一个有价值的 falsification 结果：它排除了过度归因，并把下一步证据目标明确为预注册的新路线、更多独立 outcome assignments，以及 prospective wet-lab confirmation。

## 第 16 页：LLM 自报概率校准

这一页回答一个很实际的问题：LLM 每轮给出的“所选候选会刷新当前最好结果”的概率，能不能作为可信的自我判断。我们使用完整归档中的 170 次真实决策，覆盖 17 条 source-target 路线。每一轮都保存了最终 proposer-critic 给出的 improvement probability。为了避免把候选选择能力和概率估计能力混在一起，我们拿生产 GP 对同一个已执行候选给出的 improvement probability 作比较。最终有 27 / 170 轮真正刷新了 pre-round best，真实发生率为 0.159。

左图是 reliability diagram。LLM 的平均预测概率为 0.219，比真实发生率高 0.060；同候选 GP 的平均概率为 0.182，高 0.023。LLM 的 Brier score 是 0.133，GP 是 0.134，两者几乎一样。Brier score 越低，说明概率预测越接近真实结果。右图给出每条路线的 LLM 减 GP Brier 差：负值表示 LLM 更好，正值表示 GP 更好。17 条路线中，LLM 在 7 条更好，GP 在 10 条更好。

将各路线等权后，LLM 减 GP 的 Brier 差为 -0.0016，route-bootstrap 95% 区间为 [-0.0132, +0.0090]，跨过 0。ROC-AUC 的等权差为 +0.0308，95% 区间 [-0.0581, +0.1373]，同样跨 0。因此当前不能说 LLM 的概率比 GP 更稳定，也不能把原始 LLM confidence 直接作为 transfer gate。这个结果不是候选选择对比，也不是前瞻性 gate 验证；它只是对现有归档的回顾性概率审计。

这项负结果为下一步给出了明确做法：只在 development routes 上拟合 calibration map，把模型概率校准为实际改善概率；再冻结 calibration map 和 abstention threshold，最后在新的 disjoint routes 上做前瞻验证。这样，系统会在证据不足时主动交回 GP，而不是直接相信模型自己的置信度。

## 第 17 页：在线 LLM 首批真实调用

左图的橙色刻线表示在线 LLM 相对同开局 target-only GP 的 best-so-far AUC 差值，六条路线全部为 0；蓝色条表示包含 LLM initial design 的 complete system 相对固定 initial design，三条正向、一条持平、两条负向，平均 -0.0087。右图直接检查行为：六次都选择 GP rank 1，非 GP override 为 0，critic 改选为 0；其中三次在文字上保留 source transfer，但没有改变候选。六条成功轨迹共使用 99,243 tokens；单轮 authority limit 在后续九轮交回 GP，合计避免 108 次名义 proposer/critic 调用。这个结果证明 bounded authority 能节省调用并避免长期 LLM 控制，但不能证明在线 LLM 提高性能。当前需要优化的是 decision mechanism，而不是继续增加相同调用次数。

## 第 18 页：外部任务族压力测试

这一页是本轮新增的结论边界。我们使用官方 FLIP2 Hydrophobic Core 数据，把 P01053 和 P0A9X9 两个蛋白的历史实验作为 source，把未见过的 P06241 蛋白骨架作为 target。输入只包含七个核心位点的残基组成、体积和多样性等公开特征；target 的实验稳定性在候选被执行前始终隐藏。协议在看正式结果前冻结，并在 100 个 held-out seeds 上用相同的 3 个初始观测和 12 轮预算比较方法。

左图给出传统迁移相对 target-only GP-UCB 的配对 AUC 差。Multisource RGPE 下降 16.96，95% 区间为 [-18.41, -15.51]；ICM-BMA 下降 8.13，区间为 [-9.87, -6.39]；固定多源 skill prior 下降 23.01，区间为 [-25.66, -20.36]。三个区间都完全低于 0，说明在这个新蛋白骨架上，直接搬运 source 的数值排序会产生稳定而明显的负迁移。这个负结果也说明，任务都属于蛋白稳定性并不等于它们可以安全共享 outcome prior。

右图是一次真实 Claude Opus 在线轨迹，不与左侧 100-seed 统计量混合。LLM 最初提出“低芳香、富含 Val/Ile 的 core 应该稳定”并据此给出初始设计；前三个 target 观测为 28.59、34.55 和 67.76，第一条在线候选揭示为 55.01，低于预期的 60。看到这些 target 证据后，proposer 和 critic 把解释改为 methionine-rich、低多样性 core，并明确设置 `continue_source_transfer=false`，选择 GP rank 1，把剩余 11 轮交还给 target-only GP。与完全相同开局的 GP 相比，在线决策增量为 0，且避免了 22 次后续 proposer/critic 调用。这支持“能识别失配并撤回迁移”，但不是正向性能证据。

完整系统还包含 LLM 给出的前三个初始实验，因此必须单独与固定初始设计比较。它的 AUC 低 7.19，最终最好值低 5.06。换句话说，LLM 的最初科学假设在这个任务上是错的；系统的价值是及时止损、留下可审计的失败假设和修正理由，而不是把失败轨迹包装成提升。当前论文可以据此声称 CARE 2.0 具有可拒绝、可撤回和可追溯的迁移机制，但仍不能声称在外部任务族上已经取得普遍正向增益。

## 第 19 页：无 target outcome 的负迁移 gate

这一页继续追问上一个负结果：系统能不能在读取 P06241 的真实 outcome 之前就拒绝这次迁移。我们只使用已经完成的两个蛋白任务，把 P01053 和 P0A9X9 轮流当作伪目标，每个方向运行 50 个配对 seeds。P06241 被明确禁止进入 calibration routes，也没有参与方法选择。gate 的准入规则是：一种迁移方法必须在两个 source-only 方向上都满足“相对 target-only GP-UCB 的 95% 置信区间下界大于 0”，否则就不允许迁移，直接回退到 target-only GP-UCB。

矩阵左边两列是 gate 能看到的全部证据。RGPE 在 P01053 到 P0A9X9 和反向路线上的 AUC 差分别为 -6.74 和 -5.57；ICM-BMA 分别为 -3.36 和 -8.16；固定 skill prior 分别为 -12.70 和 -27.29。六个置信区间都完全位于 0 以下，所以三种迁移方法都没有通过准入。这里不是用 target 小样本估计一次相似度，而是用已完成 source family 内部的伪目标轮换检查一种迁移机制是否具有可复用性。

如果没有 gate，只按两条 source-only 路线的平均表现选一个看起来最不差的方法，会选 ICM-BMA。矩阵第三列只用于部署后的外部审计：ICM-BMA 在 P06241 上实际比 target-only GP-UCB 低 8.13 AUC。gate 则部署 target-only GP-UCB，因此该列的部署增量为 0，也就是避免了 8.13 AUC 的损失。这个结果把“发现负迁移”和“处理负迁移”区分开：前一页证明迁移确实失败，这一页说明历史 source evidence 可以产生一个不读取 target outcome 的拒绝决定。

汇报时必须主动说清楚证据边界。虽然 gate 的计算过程没有使用 P06241 outcome，但这条 gate 是在我们已经看过 FLIP2 外部结果之后设计的，所以它属于 retrospective mechanism audit，而不是 prospective safety confirmation。我们随后按这条要求做了第二个任务族，结果放在下一页。

## 第 20 页：前瞻 Rhodopsin gate 验证

这一页与上一页最关键的差别是时间顺序。上一页的规则虽然没有读取 P06241 outcome，但规则本身是在看到 P06241 失败后设计的；这一页则先把同一套准入规则、代码哈希、seed、预算和 fallback 提交到 GitHub，再下载 Rhodopsin 数据。预注册 commit 是 `4130dd4`，记录中明确写着冻结时 raw file 不存在、test outcomes 不可用。因此，这次不是事后把一条规则套到另一个结果上，而是一次 commit-before-download 的外部验证。

数据是官方 FLIP2 Rhodopsin by-wild-type split，共 884 条实测序列、75 个 wild types。官方把 5 个最常见 wild types 放在 train，34 个放在 validation，36 个放在 test，对应 584、116 和 184 条序列。系统只从公开序列计算 20 种氨基酸组成、序列长度以及 hydrophobic、charged、aromatic、polar、gly/pro 等比例；实测 peak absorption wavelength 在候选被选择前始终隐藏。每个 target seed 有 3 个初始观测和 12 轮 reveal，各方法使用相同的起点和预算。

左图的前两组是 gate 唯一允许使用的校准证据。我们做 train 到 validation 和 validation 到 train 两条 source-only 伪目标路线，每条 50 个配对 seeds。准入规则提前锁定为：同一种方法在两条路线上的 paired AUC 95% CI 下界都必须大于 0。固定 skill prior 的路线均值最高，为 +0.788；但它在 train 到 validation 上是 +0.384，区间 [-0.301, +1.069]，跨过 0，所以仍不合格。RGPE 两条路线均略负，ICM-BMA 两条路线显著负，因此三种迁移都被 gate 拒绝，部署策略在看 test 之前已经确定为 target-only GP-UCB。

决定锁定后，我们才在官方 test 上运行 100 个配对 seeds。RGPE 相对 target-only 的 AUC 差为 -3.819，95% CI [-4.498, -3.140]；ICM-BMA 为 -1.539，区间 [-2.066, -1.011]；固定 skill prior 为 -1.548，区间 [-1.996, -1.099]。如果不使用 gate，只按 source-only 平均效果选方法，会部署固定 skill prior，并稳定损失 1.55 AUC。gate 回退到 target-only，把这次损失避免掉了。

这条结果应当准确表述为“在第二个外部蛋白任务族上，预注册的 source-only gate 成功防止了一次负迁移”。它是比上一页更强的 prospective safety evidence，但仍不是 positive efficacy：CARE 没有在 Rhodopsin 上超过 target-only，而是正确选择了不迁移。它也不能证明 gate 在所有领域都会安全，更不是 wet-lab discovery。下一阶段最重要的是在一个完全未参与开发的任务族上得到正向迁移效果，并做至少一次配对的真实实验验证。

## 数据来源

- `experiments/care_replay/configs/online_llm_gate_selection_audit_v1.json`
- `experiments/care_replay/results/2026-08-24-online-llm-route-split-gate-selection-v1/leave_one_route_out_selection.jsonl`
- `experiments/care_replay/results/2026-08-24-online-llm-route-split-gate-selection-v1/selected_policy.json`
- `experiments/care_replay/results/2026-08-24-online-llm-route-split-gate-selection-v1/selected_policy_evaluation_routes.jsonl`
- `experiments/care_replay/results/2026-08-24-online-llm-route-split-gate-selection-v1/controller_selection_stability.png`
- `experiments/care_replay/results/2026-08-22-submission-evidence-audit/route_evidence.csv`
- `experiments/care_replay/results/2026-08-22-submission-evidence-audit/phase_statistics.csv`
- `experiments/care_replay/results/2026-08-16-opus48-bounded-ei-development-v2/molecular_freesolv_to_lipophilicity/llm_trace.jsonl`
- `experiments/care_replay/results/2026-08-16-opus48-bounded-ei-development-v2/molecular_freesolv_to_lipophilicity/summary.json`
- `experiments/care_replay/results/2026-08-23-online-llm-repeated-confirmation-v1/aggregate/repeated_confirmation.json`
- `experiments/care_replay/results/2026-08-23-online-llm-repeated-confirmation-v1/aggregate/repeated_route_effects.svg`
- `experiments/care_replay/results/2026-08-23-online-llm-repeated-confirmation-v2/reizman_cases_123_to_case4/trajectory_2000/summary.json`
- `experiments/care_replay/configs/online_llm_repeated_confirmation_v2.json`
- `experiments/care_replay/results/2026-08-23-llm-trajectory-stability-audit/paired_route_stability.csv`
- `experiments/care_replay/results/2026-08-23-llm-trajectory-stability-audit/paired_route_stability.svg`
- `experiments/care_replay/configs/online_llm_glm53_robustness_v1.json`
- `experiments/care_replay/configs/online_llm_glm53_robustness_v3.json`
- `experiments/care_replay/configs/classical_transfer_benchmark_v1.json`
- `experiments/care_replay/results/2026-08-08-classical-transfer-confirmation/comparison/care_vs_classical.json`
- `experiments/care_replay/results/2026-08-24-online-llm-transport-context-robustness-v3/aggregate/repeated_confirmation.json`
- `experiments/care_replay/results/2026-08-24-online-llm-transport-context-robustness-v3/materials_phonons_to_bulk_modulus/trajectory_5000/summary.json`
- `experiments/care_replay/results/2026-08-24-online-llm-transport-context-robustness-v3/materials_phonons_to_bulk_modulus/trajectory_5000/llm_trace.jsonl`
- `experiments/care_replay/results/2026-08-24-online-llm-transport-context-robustness-v3/materials_phonons_to_bulk_modulus/trajectory_5000/audit/materials_online_trace_audit.json`
- `experiments/care_replay/configs/online_llm_outcome_semantics_robustness_v1.json`
- `experiments/care_replay/results/2026-08-24-online-llm-outcome-semantics-robustness-v1/materials_phonons_to_bulk_modulus/trajectory_6000/summary.json`
- `experiments/care_replay/results/2026-08-24-online-llm-outcome-semantics-robustness-v1/materials_phonons_to_bulk_modulus/trajectory_6000/audit/outcome_semantics_audit.json`
- `experiments/care_replay/configs/online_llm_matched_classical_audit_v1.json`
- `experiments/care_replay/results/2026-08-24-online-llm-matched-classical-audit-v1/aggregate.json`
- `experiments/care_replay/results/2026-08-24-online-llm-matched-classical-audit-v1/comparisons.csv`
- `experiments/care_replay/configs/source_outcome_falsification_v1.json`
- `experiments/care_replay/results/2026-08-24-source-outcome-falsification-v1/falsification_report.json`
- `experiments/care_replay/results/2026-08-24-source-outcome-falsification-v1/trajectory_metrics.csv`
- `experiments/care_replay/scripts/plot_source_outcome_falsification.py`
- `experiments/care_replay/configs/hidden_target_noninterference_v1.json`
- `experiments/care_replay/results/2026-08-24-hidden-target-noninterference-v1/noninterference_report.json`
- `experiments/care_replay/results/2026-08-24-hidden-target-noninterference-v1/decision_states.csv`
- `experiments/care_replay/results/2026-08-24-hidden-target-noninterference-v1/route_summary.csv`
- `experiments/care_replay/scripts/audit_hidden_target_noninterference.py`
- `experiments/care_replay/configs/llm_probability_calibration_audit_v1.json`
- `experiments/care_replay/results/2026-08-24-llm-probability-calibration-audit-v1/probability_calibration_report.json`
- `experiments/care_replay/results/2026-08-24-llm-probability-calibration-audit-v1/decision_calibration.csv`
- `experiments/care_replay/results/2026-08-24-llm-probability-calibration-audit-v1/route_calibration.csv`
- `experiments/care_replay/scripts/audit_llm_probability_calibration.py`
- `experiments/care_replay/results/2026-08-24-online-llm-predeclared-baseline-portfolio-v1/aggregate.json`
- `experiments/care_replay/results/2026-08-24-online-llm-predeclared-baseline-portfolio-v2/aggregate.json`
- `experiments/care_replay/results/2026-08-24-online-llm-calibration-gate-audit-v1/aggregate.json`
- `experiments/care_replay/results/2026-08-24-online-llm-calibration-gate-audit-v1/route_results.csv`
- `experiments/care_replay/configs/online_llm_gated_repeated_confirmation_v1.json`
- `experiments/care_replay/results/2026-08-24-online-llm-fixed-threshold5-gate-replay-v1/aggregate.json`
- `experiments/care_replay/results/2026-08-24-online-llm-fixed-threshold5-gate-replay-v1/route_results.csv`
- `experiments/care_replay/results/2026-08-24-online-llm-route-disjoint-fixed-threshold5-gate-audit-v1/aggregate.json`
- `experiments/care_replay/results/2026-08-24-online-llm-route-disjoint-bounded-authority-v1/aggregate.json`
- `experiments/care_replay/results/2026-08-24-online-llm-route-split-gate-selection-v1/selected_policy.json`
- `experiments/care_replay/results/2026-08-24-online-llm-route-split-gate-selection-v1/selected_policy_evaluation_routes.jsonl`
- `experiments/care_replay/configs/online_llm_bounded_authority_route_disjoint_confirmation_v1.json`
- `experiments/care_replay/results/2026-08-24-online-llm-bounded-authority-route-disjoint-confirmation-v1/pilot_audit/decision_impact_audit.json`
- `experiments/care_replay/results/2026-08-24-online-llm-bounded-authority-route-disjoint-confirmation-v1/pilot_audit/route_decision_metrics.csv`
- `experiments/care_replay/results/2026-07-21-cross-domain-semantic-skills/README.md`
- `docs/GLM53_CROSS_MODEL_PROTOCOL.md`
- `experiments/care_replay/data/raw/flip2_hydro_to_P06241.csv.gz`
- `experiments/care_replay/configs/flip2_hydro_external_family_v1.json`
- `experiments/care_replay/results/2026-08-24-flip2-hydro-external-family-v1/external_family_audit/external_task_family_transfer.png`
- `experiments/care_replay/results/2026-08-24-flip2-hydro-external-family-v1/external_family_audit/external_task_family_transfer.json`
- `experiments/care_replay/results/2026-08-24-opus48-flip2-hydro-external-family-v1/aggregate.json`
- `experiments/care_replay/results/2026-08-24-opus48-flip2-hydro-external-family-v1/protein_flip2_p01053_p0a9x9_to_p06241/llm_trace.jsonl`
- `experiments/care_replay/configs/flip2_hydro_source_only_gate_v1.json`
- `experiments/care_replay/results/2026-08-24-flip2-source-only-family-gate-v1/source_only_gate_summary.json`
- `experiments/care_replay/results/2026-08-24-flip2-source-only-family-gate-v1/route_effects.csv`
- `experiments/care_replay/results/2026-08-24-flip2-source-only-family-gate-v1/source_only_family_gate.png`
- `experiments/care_replay/configs/flip2_rhomax_prospective_gate_v1.json`
- `experiments/care_replay/preregistrations/2026-08-25-flip2-rhomax-prospective-gate-v1/protocol_preregistration.json`
- `experiments/care_replay/results/2026-08-25-flip2-rhomax-prospective-gate-v1/source_only_gate_summary.json`
- `experiments/care_replay/results/2026-08-25-flip2-rhomax-prospective-gate-v1/prospective_rhomax_gate.png`
