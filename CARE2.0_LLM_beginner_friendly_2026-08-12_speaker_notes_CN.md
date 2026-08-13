# CARE 2.0 LLM 参与版：小白友好逐页中文 Speaker Note

对应 PPT：`CARE2.0_LLM_beginner_friendly_2026-08-12.pptx`

本讲稿对应 13 页小白友好汇报版，不要求听众预先了解机器学习、贝叶斯优化或化学实验优化。每页先讲“本页核心”，再按“详细讲解”展开；第一次出现的专业词都在“名词解释”中说明。

## 第 1 页：封面：我们想让旧实验帮助新实验

**本页核心：** CARE 2.0 用真实 LLM 把历史实验经验变成新实验可以执行的建议。

**详细讲解：**

开场可以这样说：我们面对的不是一道已经有答案的题，而是一组昂贵的新实验。每做一次实验都要花时间和成本，所以我们希望过去做过的实验不要白费。CARE 2.0 做的事，就是把旧实验里的经验整理出来，帮助新实验更快找到好条件。

这次介绍的是 LLM 参与版。LLM 不是负责直接猜最终结果，也不是单纯写一段解释。它会阅读已经完成的历史实验，提出一个可以被验证的判断，并推荐新实验最先做哪三个条件。随后系统用统一方法继续优化，并把整个过程保存下来。

**名词解释：** LLM=大语言模型；历史实验=已经做完并知道结果的实验；新实验=还没有揭示结果的目标任务。

**转场：** 下一页进入：先看结论：LLM 已参与，但还需要新任务确认。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-status-and-ppt/CARE2_CURRENT_STATUS_CN.md`

## 第 2 页：先看结论：LLM 已参与，但还需要新任务确认

**本页核心：** 固定方法已有一个严格外部正例；真实 LLM 在五个既有任务上平均提高搜索效率。

**详细讲解：**

这一页只需要让听众记住两件事。第一，在一个事先没有看过结果的新 Suzuki 任务上，固定版 CARE 把全过程搜索效率提高了 8.66。这证明历史实验确实可能帮助新实验更快找到好条件，但这条外部结果本身没有使用 LLM。

第二，我们随后真实调用了 GPT-5.4，让它参与五个任务的开局选择。相对固定版，平均搜索效率提高 2.12，五个任务中有四个没有变差。不过统计区间仍然跨过零，而且这些任务过去被项目使用过。因此最准确的说法是：LLM 已经真实参与并出现正向信号，但还需要在全新任务上确认。

**名词解释：** 搜索效率=好结果出现得越早，效率越高；外部任务=策略确定后才第一次查看结果的任务。

**转场：** 下一页进入：为什么要做知识迁移。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-v2-external-confirmation/README.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/README.md`

## 第 3 页：为什么要做知识迁移

**本页核心：** 实验昂贵，真正的问题是怎样用更少的尝试找到更好的条件。

**详细讲解：**

可以把它想成第一次做一道新菜。我们知道有哪些食材、温度和时间可以选择，但不知道哪一种组合最好。每试一次都要付出成本。历史上做过的类似菜谱可能有帮助，但也可能因为食材不同而误导我们。

CARE 2.0 每一轮只允许选择一个实验，做完以后才看到这个实验的结果，然后再决定下一轮。系统的目标不是把整张答案表背下来，而是在相同实验次数下更早找到高质量条件，同时避免错误迁移浪费预算。

**名词解释：** 候选条件=可以选择的一组实验参数；实验预算=最多允许做多少次新实验。

**转场：** 下一页进入：系统怎么工作。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/CARE2_METHOD.md`

## 第 4 页：系统怎么工作

**本页核心：** 旧实验负责提供开局建议，新实验开始后只根据真实的新结果继续学习。

**详细讲解：**

整条流程可以分成五步。第一，读入已经完成的历史实验。第二，从多个历史任务中找出比较一致的规律。第三，把规律整理成一个可以执行的技能，例如优先尝试哪些区域。第四，选择新任务最先做的三个实验。第五，做完这三个实验以后，历史信息退出，后面只根据新任务已经观察到的结果继续选择。

这样设计的好处是归因清楚。如果 CARE 和基线最后不同，主要差别来自开局三个实验，而不是后面偷偷用了更多数据或不同预算。LLM 参与的是提出科学假设和选择语义锚点，数值优化器负责后续稳定搜索。

**名词解释：** 技能=包含建议、使用范围和失败条件的可执行策略；开局实验=最先执行的少量实验。

**转场：** 下一页进入：Replay Harness 是什么。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`

## 第 5 页：Replay Harness 是什么

**本页核心：** 它用历史真实数据模拟逐轮实验，但算法不能提前看到整张答案表。

**详细讲解：**

Replay Harness 可以理解成一个实验模拟器。完整数据表被系统保存在后台，扮演真实实验环境；算法看不到全部结果。每一轮，算法只能提交一个候选条件，环境返回这个条件的真实测量值，然后算法才能做下一次选择。

这种方式比普通的表格预测更接近真实实验节奏。我们还能保证所有方法使用相同候选集合、相同开局点数和相同后续轮数。每一轮选了什么、为什么选、得到什么结果、当前最好值是多少，都会写入日志。

**名词解释：** Replay=用已有真实实验表重放逐轮决策；日志=每一轮选择和结果的完整记录。

**转场：** 下一页进入：LLM 输出的不只是一段文字。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/CARE2_METHOD.md`

## 第 6 页：LLM 输出的不只是一段文字

**本页核心：** LLM 必须给出明确的三个候选、理由、信心和失败条件。

**详细讲解：**

LLM 能看到的是历史实验结果、新任务有哪些可选参数，以及候选条件的公开信息。它明确看不到新任务的实验结果。模型需要输出一个结构化方案：它认为哪条历史规律可能适用于新任务、建议最先做哪三个实验、每个实验承担什么作用、信心有多高，以及什么现象会说明这个判断不成立。

这一步很重要，因为只有结构化输出才能真正执行和审计。单纯让 LLM 写“这个任务可能相似”没有实验意义。CARE 会把模型的输出冻结成 JSON，保存原始 prompt、原始回答、模型版本和哈希，然后才开始 replay。

**名词解释：** 假设=可以被实验支持或推翻的判断；冻结=生成后不再根据目标结果修改。

**转场：** 下一页进入：怎样证明 CARE 真的有用。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/CARE2_METHOD.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`

## 第 7 页：怎样证明 CARE 真的有用

**本页核心：** 所有方法只有前三个实验不同，后续算法和实验次数完全相同。

**详细讲解：**

我们比较三个主要对照。CARE 根据历史实验选择前三个点；随机对照随机选择前三个点，但之后也运行同一个优化器；空间覆盖对照选择彼此分散的三个点，之后同样运行这个优化器。

所以随机对照不是从头到尾瞎猜，它只是在开局不使用历史知识。后续每一组都采用相同的选点算法、相同的十轮实验和相同的候选空间。这样如果 CARE 更早找到好结果，我们才可以把差异归因于历史知识或 LLM 建议，而不是计算量不公平。

**名词解释：** 基线=用来判断新方法是否真的更好的对照；target-only=只使用新任务已经观察到的数据。

**转场：** 下一页进入：一个严格外部任务上的结果。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/BAUMGARTNER_WARMSTART_V2_PROTOCOL.md`

## 第 8 页：一个严格外部任务上的结果

**本页核心：** 固定 v2 更早找到好条件，并在第五次总观测时找到 100% 产率。

**详细讲解：**

左图比较的是整个搜索过程。绿色的 Frozen v2 得分是 98.06，高于空间覆盖的 89.40 和随机方法的平均 88.84。这个数字越高，代表越早找到高产率条件。右图比较最后找到的最好结果，v2 达到 100% 产率，空间覆盖是 91.82，随机平均是 94.77。

这条证据最严格，因为规则在查看目标任务结果前已经锁定。不过要再次提醒：这个外部结果验证的是历史实验驱动的固定开局，不是 LLM 的外部验证。LLM 的独立结果在后面三页。

**名词解释：** 产率=化学反应得到目标产物的比例；100% 是产率，不是模型准确率。

**转场：** 下一页进入：LLM 真正参与了哪一步。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-10-baumgartner-warmstart-v2-external-confirmation/README.md`

## 第 9 页：LLM 真正参与了哪一步

**本页核心：** LLM 先提出科学方向，再由安全层把建议变成稳妥的三个实验。

**详细讲解：**

左边是 LLM 的输入：历史实验结果、新任务的参数结构、公开候选条件和一个候选短名单，里面没有新任务答案。中间是真实的 GPT-5.4 调用。它输出三个候选、科学理由、信心和失败条件。右边是执行方式。

我们测试两种版本。Raw LLM 直接执行模型选择的三个点；Compiled LLM 保留一个 LLM 认为最有科学意义的点，再加入一个历史数据最支持的点和一个增加覆盖面的点。之后两种方法都进入完全相同的 target-only 优化。所有调用时间、token、原始回答和哈希都已保存，密钥没有保存。

**名词解释：** Raw LLM=直接采用模型建议；Compiled LLM=经过安全层整理后的模型建议。

**转场：** 下一页进入：Suzuki 案例：LLM 的方向对，但直接选点太集中。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/README.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/scripts/run_llm_initial_design_hypothesis.py`

## 第 10 页：Suzuki 案例：LLM 的方向对，但直接选点太集中

**本页核心：** 安全层保留 LLM 的科学判断，同时补足数据覆盖，最终超过固定版。

**详细讲解：**

LLM 判断高温、高催化剂用量区域更有希望，并认为温度是主要因素，信心为 0.73。这个科学方向有道理，但 Raw LLM 选择的三个点都太集中，导致后续模型看不到足够多样的信息，全过程得分只有 91.819，低于固定版的 98.061。

安全层没有丢掉 LLM，而是保留它选出的一个关键点，再加入历史数据最支持的点和一个用于扩大覆盖的点。这个组合的全过程得分达到 100，比固定版高 1.939，最终两者都找到 100% 产率。它说明 LLM 更适合提供科学语义和假设，数值安全层负责把想法变成稳妥实验设计。

**名词解释：** 语义锚点=最能代表 LLM 科学判断的候选；覆盖=让初始实验不要全部挤在同一区域。

**转场：** 下一页进入：五个任务的 LLM 结果。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/suzuki_minlp2/evaluation_compiled/summary.json`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/suzuki_minlp2/llm_hypothesis_record.json`

## 第 11 页：五个任务的 LLM 结果

**本页核心：** 平均搜索效率提高 2.12，四个任务没有下降，但还没有达到统计上的普遍胜出。

**详细讲解：**

五个任务都进行了真实模型调用。与固定 v2 相比，Suzuki 提高 1.939，Morpholine-AlPhos 提高 5.203，Phenethylamine-AlPhos 下降 0.947，Morpholine-tBuBrettPhos 提高 4.422，另一个 preliminary 任务持平。平均提高 2.123，三胜、一平、一负；最终找到的最好结果五个都没有下降。

但红框中的置信区间从 -0.226 到 4.473，仍然包含零。用普通话说，就是平均结果看起来更好，但任务数量还少，不能排除其中一部分来自任务差异或偶然波动。因此我们报告正向信号，也保留失败案例。

**名词解释：** 置信区间跨零=目前数据还不足以断言平均提升一定大于零。

**转场：** 下一页进入：现在可以说什么，不能说什么。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/aggregate/suite_summary.json`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/README.md`

## 第 12 页：现在可以说什么，不能说什么

**本页核心：** LLM 已真实参与且平均为正；还不能说它在所有新任务上都更好。

**详细讲解：**

已经成立的有四点：系统可以逐轮、可审计地重放真实实验；固定版在一个严格外部任务上明显提高搜索效率；LLM 可以生成真正可执行的实验假设；在五个既有任务上，LLM 方案平均提高 2.12。

仍未成立的也要清楚：我们还没有证明任意领域之间都能稳定迁移，也没有证明 LLM 在全新的任务上普遍超过固定规则。项目当前最可靠的定位，是一个可信的知识迁移框架：它会提出建议，也允许拒绝迁移；成功、失败和被拒绝的案例都会进入知识库。

**名词解释：** 可信不等于每次都迁移；能够识别风险并退回基线，也是系统能力。

**转场：** 下一页进入：下一步怎么验证。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-status-and-ppt/CARE2_CURRENT_STATUS_CN.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/README.md`

## 第 13 页：下一步怎么验证

**本页核心：** 先锁定 LLM 方法，再到全新任务上测试，避免看到答案后改规则。

**详细讲解：**

下一步最重要的不是继续在旧任务上调到更高，而是冻结整套确认协议。我们要事先确定使用哪个 prompt、哪个模型版本、Raw 还是 Compiled、使用多少实验轮数、比较什么指标，并保存对应哈希。之后再第一次打开全新目标任务的结果。

对照至少包括固定 v2、LLM hypothesis 方案和格式完全匹配的随机假设。随着独立新任务增加，我们才能计算任务层面的置信区间，判断 LLM 是否具有稳定泛化能力。领域扩展也要循序渐进：先做反应到反应、分子到分子、材料到材料，再讨论更远距离的跨领域迁移。

**名词解释：** 确认协议=在看到新任务结果前锁定的实验规则；泛化=在没有针对性调参的新任务上仍然有效。

**转场：** 汇报结束，进入讨论。

**参考文件：**
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-status-and-ppt/CARE2_CURRENT_STATUS_CN.md`
- `/Users/rentamac/Documents/Codex/2026-05-12/CARE2.0/experiments/care_replay/results/2026-08-12-llm-hypothesis-initial-design/README.md`
