# Actor-Critic

## 定位与分类
- 阶段：后训练（策略优化）
- 类型：策略网络 + 价值网络联合训练
- 作用：在提升奖励的同时降低策略梯度估计方差

## 核心原理
1. Actor 负责输出策略分布。
2. Critic 估计状态价值/优势，指导 Actor 更新。
3. 通过 advantage 减小纯 REINFORCE 的高方差问题。

## 与相近方法区别
1. 相比 `Policy Gradient`：多了 Critic，通常更稳定、更高样本效率。
2. 相比 `PPO`：Actor-Critic 是结构范式，PPO 是具体优化目标/约束策略。
3. 相比 `GAE`：GAE 是优势估计技术，可作为 Actor-Critic 的组成部分。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/alignment/actor_critic
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/actor_critic.py --reward-model <奖励模型路径或名称>
```

## 输出结果
默认输出到 `output/actor_critic_metrics`，包含：
- `training_metrics.csv`
- `training_curves.png`
- `summary.json`
- `log_history.json`
