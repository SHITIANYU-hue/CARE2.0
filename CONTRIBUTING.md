# 开发与实验产物约定

## 代码和输入

- 从整理后的主线创建任务分支；在独立分支提交实验代码和协议。
- 复用 `experiments/care_replay/scripts/` 的候选池、技能、路由和 gate 实现。
- 新实验的可执行配置放 `configs/`，固定输入策略放 `skill_banks/`，预注册文件放 `preregistrations/`；这些 JSON 应继续以文本跟踪。
- 运行参数、随机种子、数据版本、模型信息及证据边界应写进实验协议。修改冻结协议时创建新版本。
- 测试放对应实验的 `tests/`。主线入口是 `python care.py pair|suite`。

## 输出和归档

1. 新运行写入 `experiments/care_replay/outputs/` 或 `results/<run-id>/`；这些目录已被 Git 忽略。
2. 在运行目录保留报告、指标、日志、配置快照、模型调用和负结果。
3. `python care.py pack experiments/care_replay/results/<run-id> --id <run-id>` 创建 tar.gz，并登记原路径和逐文件 SHA-256；不会删除源目录。
4. `python care.py artifacts verify <run-id>` 验证后，提交 `artifacts/` 中的新归档及目录索引。
5. 单个归档源内容超过 90 MiB 时按子实验拆分。模型权重不进入普通 Git。

PPT、图表发布包和报告附件采用同样流程。不要使用 `git add -f` 提交裸运行目录。知识库卡片是检索输入，继续保留为 JSON；SQLite、embedding 缓存和导出文件不提交。

## 提交前检查

```bash
python -m pip install -r requirements-dev.txt
python -m pytest experiments/care_replay/tests/test_transfer_skill.py experiments/care_replay/tests/test_source_outcome_transfer.py experiments/care_replay/tests/test_cross_task_router.py experiments/care_replay/tests/test_hidden_target_noninterference.py tests
python care.py smoke
git add <本次修改的代码、配置、文档、归档>
python care.py check
git diff --cached --check
```

`check` 检查 Git 索引，因此新增/删除文件应先暂存。完整历史证据测试的恢复步骤见 [整理与复现说明](docs/REORGANIZATION.md)。
