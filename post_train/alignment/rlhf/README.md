# RLHF（Reinforcement Learning from Human Feedback）

## 定位与分类
- 阶段：后训练（完整对齐流程）
- 类型：人类偏好驱动强化学习
- 作用：让模型回答更符合“人类偏好”而不仅是“语料分布”

## 核心原理
1. 先有基础模型（通常经过 SFT）。
2. 用偏好数据训练奖励模型。
3. 用 PPO 等算法优化策略模型，使奖励上升并控制 KL 偏移。

## 与相近方法区别
1. 相比 `SFT`：RLHF 优化的是偏好奖励，不是单一参考答案拟合。
2. 相比 `DPO`：RLHF 常显式包含奖励模型与在线 RL 阶段。
3. 相比 `PPO`：PPO 是算法，RLHF 是完整训练流程。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/alignment/rlhf
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/rlhf.py --reward-model <奖励模型路径或名称>
```

## 输出结果
默认输出到 `output/rlhf_metrics`，包含：
- `training_metrics.csv`
- `training_curves.png`
- `summary.json`
- `log_history.json`


## 目录文件说明（重点）
- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
