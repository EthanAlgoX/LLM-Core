# TD Learning（时序差分学习）

## 定位与分类
- 阶段：RL 基础理论
- 类型：无模型价值学习
- 作用：边交互边更新价值估计，避免完整回合结束后才学习

## 核心原理
1. 使用 bootstrap 目标：`r + γ V(s')` 或 `r + γ max_a Q(s',a)`。
2. 典型算法：TD(0)、Q-learning、SARSA。
3. 在在线学习场景下通常比纯 MC 更高效。

## 与相近方法区别
1. 相比 `MDP` 规划：TD 不依赖显式转移概率模型。
2. 相比 `GAE`：GAE 是优势估计方法，TD 是更基础的价值更新思想。
3. 相比 `Policy Gradient`：TD 主要学习值函数，PG 直接优化策略。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/rl_basics/td_learning
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/td_learning.py
```

## 输出结果
默认输出到 `output/td_learning_metrics`，包含：
- `episode_log.csv`
- `training_curves.png`
- `q_table.json`
- `policy.json`
- `summary.json`


## 目录文件说明（重点）
- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
