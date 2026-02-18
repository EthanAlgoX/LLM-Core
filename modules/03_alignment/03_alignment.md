# Alignment 分类

本目录聚焦后训练对齐方法：

- `sft`
- `dpo`
- `grpo`
- `ppo`
- `policy_gradient`
- `actor_critic`
- `rlhf`

建议学习顺序：
`sft -> dpo -> ppo/grpo -> rlhf`

## 子模块目录说明（统一）
- `code/`：主流程代码（单文件可运行），强调学习路径中的关键训练步骤。
- `data/`：指令数据、偏好数据、奖励相关样本或示例输入。
- `models/`：训练后导出的最终模型文件（用于推理与部署）。
- `checkpoints/`：训练过程中的阶段性快照（含步数/优化器状态等）。
- `output/`：损失曲线、奖励曲线、训练日志、汇总指标（`csv/png/json`）。
