# RL Basics 分类

本目录聚焦强化学习基础概念与算法直觉：

- `mdp`
- `td_learning`
- `gae`
- `advantage`

建议学习顺序：
`mdp -> td_learning -> advantage -> gae`

## 子模块目录说明（统一）
- `code/`：核心算法脚本，便于直接运行观察学习过程。
- `data/`：环境设定、状态转移示例或参数配置。
- `models/`：策略/价值函数等最终结果文件。
- `checkpoints/`：中间训练快照（若该模块启用训练保存）。
- `output/`：可视化图、指标记录、策略与价值表。
