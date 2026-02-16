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
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/td_learning
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
