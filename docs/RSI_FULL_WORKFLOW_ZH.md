# CARE 2.0：RSI 全流程图

从数据准备到模型训练、实验选点、反馈修订、配对评价与归档的完整流程。实验节点目前使用离线测量数据重放。图中数字对应本次配置；虚线表示人工发起的后续训练，不是本轮自动回训。

```mermaid
flowchart TB
    subgraph INPUT["① 准备科学问题与证据"]
        I1["历史实验／源任务测量数据"]
        I2["领域知识与科学指令"]
        I3["目标任务、候选集合<br/>允许使用的结构特征"]
        I1 --> P["组织模型输入<br/>科学证据＋任务约束＋输出格式"]
        I2 --> P
        I3 --> P
    end

    subgraph TRAIN["② 可选：离线模型后训练"]
        T0["已有模型请求与规则回答"]
        T0 --> T1["按提示词分组划分训练／验证<br/>检查规则回答能通过编译"]
        T1 --> T2["A800：QLoRA 微调<br/>本次 42 条训练数据，完整训练 3 轮"]
        T2 --> T3["验证 loss 选择 checkpoint<br/>保存 adapter 与训练状态"]
        T3 --> LOCAL["A800 本地生成端<br/>原模型／选定的 LoRA 版本"]
        BASE["不微调，直接加载原模型"] --> LOCAL
    end

    CLOUD["CommonStack 远端模型<br/>用于上下文 RSI"] --> G
    LOCAL --> G
    P --> G

    subgraph GENERATE["③ 生成并检查规则"]
        G["LLM 输出 2 套规则<br/>选择 1 套部署方案＋总结经验"]
        G --> C{"规则是否合法且可执行？"}
        C -->|否| R{"还有重试额度？<br/>本次最多 2 次尝试"}
        R -->|有| ERR["记录错误，补充纠错提示"]
        ERR --> G
        R -->|没有| FAIL["记录失败<br/>中止本条生成链"]
        C -->|是| V["保存当前规则版本与部署选择<br/>初始 v0 → 修订 v1 → 修订 v2"]
        V --> Q{"该分支已完成<br/>2 轮修订？"}
    end

    Q -->|尚未完成| DEV["开发实验<br/>每套规则使用 2 个开发种子"]

    subgraph EXPERIMENT["④ 每条实验轨迹内的选点循环"]
        START["按种子获得 5 个初始测量<br/>对照组使用相同初始候选"]
        START --> GP["GP 根据已有测量<br/>更新预测与不确定性"]
        GP --> RANK["结合规则先验、预测与探索需求<br/>给未测候选排序"]
        RANK --> PICK["选择下一项实验"]
        PICK --> MEASURE["获得该候选的测量结果<br/>当前：从离线数据读取"]
        MEASURE --> OBS["加入新观测<br/>更新已找到的最好结果"]
        OBS --> ROUND{"已完成 10 轮选点？"}
        ROUND -->|否| GP
        ROUND -->|是| PURPOSE{"当前实验用途？"}
    end

    DEV --> START

    subgraph FEEDBACK["⑤ RSI：实验反馈驱动规则修订"]
        PURPOSE -->|开发实验| F["汇总两套规则的实验表现<br/>AUC、最终最好结果、逐种子成绩"]
        F --> MODE["三个分支共享初始 v0，各自修订<br/>真实反馈：正确绑定结果<br/>无反馈：不提供测量反馈<br/>打乱反馈：交换规则与结果的绑定"]
        MODE --> MEMORY["组织各分支的修订输入<br/>上一版规则＋允许提供的历史反馈"]
        MEMORY -->|生成下一版规则| G
    end

    V -->|初始 v0 另存为对照| CONTROL["固定初始规则对照<br/>另设仅 GP-UCB 对照"]

    subgraph EVALUATION["⑥ 冻结后做配对评价"]
        Q -->|已完成| LOCK["三种修订分支全部完成后<br/>冻结 5 组最终执行方案"]
        CONTROL --> LOCK
        LOCK --> TEST["使用 10 个评价种子<br/>相同初始测量、相同实验预算"]
        TEST --> START
        PURPOSE -->|评价实验：禁止回传修订| SCORE["比较原模型与训练后模型<br/>再比较真实反馈、固定初始、无反馈、打乱反馈与 GP"]
        SCORE --> RESULT["分别判断<br/>初始策略提升／反馈增益／递归增益"]
    end

    RESULT --> ARCHIVE["⑦ 全程记录归档<br/>请求、回答、规则版本、实验轨迹<br/>冻结配置、指标、失败记录与模型哈希"]
    FAIL --> ARCHIVE

    ARCHIVE -.-> NEXT["人工发起下一轮训练<br/>重新组织数据并建立新的评价划分"]
    NEXT -.-> T1
```

## 三种更新分别发生在哪里

- **神经网络参数**：在离线 QLoRA 训练阶段更新；在线搜索和规则修订期间保持冻结。
- **GP 预测与不确定性**：每获得一个实验测量结果后更新；仅 GP-UCB 对照不使用 LLM 规则先验。
- **规则与部署选择**：LLM 读取开发反馈后修订，形成 v0 → v1 → v2；无反馈组不接收测量反馈。

评价种子的反馈不进入本轮规则修订或 checkpoint 选择。下一轮训练需要另行整理数据、建立新的训练／评价划分并人工发起。当前训练和评价仍复用任务及有限候选池，流程完成不能单独证明跨任务泛化或递归增益。

## 对应实现与实验记录

- [数据构建](../experiments/care_replay/scripts/export_rsi_full_dataset.py)
- [A800 QLoRA 训练](../experiments/care_replay/scripts/train_rsi_qlora.py)
- [生成、反馈修订、冻结与评价](../experiments/care_replay/scripts/run_rsi_live_feedback.py)
- [实验内 GP 与规则先验选点](../experiments/care_replay/scripts/llm_semantic_skills.py)
- [训练完成报告与权重校验](../experiments/care_replay/results/2026-09-15-rsi-qwen25-32b-full-sft-v4/README.md)
- [训练前后 RSI 配对对照结果](../experiments/care_replay/results/2026-09-15-rsi-posttraining-comparison-v1/README.md)
