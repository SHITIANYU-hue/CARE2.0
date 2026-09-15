# 后训练 RSI：完整运行记录

使用完整三轮 QLoRA 训练后按验证 loss 选出的第 2 轮 adapter。初始生成与真实反馈、无反馈、打乱反馈三组各两轮生成均完成；7 次尝试全部通过规则编译，失败为 0。

保留 50 条配对评价轨迹、开发反馈、原始请求/回答、版本状态和评价前冻结记录。当前评价种子的反馈未进入修订提示；模型权重在整个搜索过程中冻结。本地生成采用生成后规则编译检查。

- [训练前后效果与适用范围](../2026-09-15-rsi-posttraining-comparison-v1/README.md)
- [模型训练与权重校验](../2026-09-15-rsi-qwen25-32b-full-sft-v4/README.md)
- [生成统计及使用的 adapter 哈希](generation_summary.json)
- [最终规则与候选选择变化](execution_audit.json)
- [运行完成状态](run_status.json)
