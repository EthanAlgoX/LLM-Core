# Post-Train

该目录用于放置后训练（监督微调与偏好优化）相关模块。

当前模块：
- `sft`
- `dpo`
- `grpo`
- `ppo`
- `policy_gradient`
- `actor_critic`
- `rlhf`
- `mdp`
- `td_learning`
- `gae`
- `advantage`
- `cql`
- `bcq`
- `deepspeed`
- `cuda`

通用目录规范（每个模块内）：
- `code/`: 单文件运行脚本
- `data/`: 数据目录
- `models/`: 最终模型文件
- `checkpoints/`: 训练中间结果
- `output/`: 指标、曲线图、配置快照
