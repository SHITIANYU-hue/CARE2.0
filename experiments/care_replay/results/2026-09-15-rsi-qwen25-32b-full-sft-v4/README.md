# RSI 完整数据训练：已完成

在 A800 上完成 Qwen2.5-32B-Instruct 的 4-bit QLoRA 监督微调。42 条现有训练样本完整跑 3 个 epoch，共 126 次样本处理、33 次 optimizer step；不是全参数微调，也不是只跑几步的 smoke test。训练与验证均保留完整上下文。

| 项目 | 实测结果 |
|---|---:|
| 训练 / 验证样本 | 42 / 12 |
| 训练 / 验证精确 prompt 重叠 | 0 |
| LoRA 可训练参数 | 134,217,728 |
| 训练前验证 loss | 1.029708 |
| 第 1 / 2 / 3 轮验证 loss | 0.822446 / 0.780409 / 0.783929 |
| 验证选择的模型 | 第 2 轮 |
| CUDA 峰值分配显存 | 50.526 GiB |
| 训练与末尾生成检查耗时 | 1,799.778 秒 |

[训练曲线](training_curve.png)、[训练指标](metrics.json)、[训练审计](training_audit.json)和[完整日志](stdout.log)均保留。每轮 adapter、optimizer、scheduler 和 RNG 状态保存在服务器。

权重检查：896 个 LoRA 张量全部为有限值，448 个 B 矩阵非零。选中 adapter 为 536,991,984 字节，SHA256 为 `30cd27d9ee6250ced68434096a698770ebdd9e2bc420bf5dfd32be285d907ed3`。使用路径：`/work/zeyuwang/care-rsi/repo/experiments/care_replay/results/2026-09-15-rsi-qwen25-32b-full-sft-v4/best_adapter`。

末尾生成检查使用第 3 轮模型：3/3 是 JSON 对象，2/3 通过严格规则编译；FreeSolv 的一个回答用了不支持的字段。这与选中第 2 轮模型的后训练 RSI 测试分开记录。规则编译器负责拒绝非法回答，本地生成没有使用 token 级约束解码。

[模型文件哈希](base_model_manifest.json)、[执行清单](execution_manifest.json)及[服务器代码快照](code_snapshot/README.md)标识实际运行版本。之前的 prompt 重叠、GPU0 OOM 和训练目标内部格式错误均在 attempt_history 中保留；被中止的 v3 模型没有用于部署。

数据涵盖 FreeSolv、Lipophilicity 和实验带隙三个任务；训练和验证仍共享任务及相关历史。这次证明现有数据的完整训练流程已完成，不能据此认定科学实验增益或跨任务泛化。[后训练 RSI 配对对照](../2026-09-15-rsi-posttraining-comparison-v1/README.md)也已完成。
