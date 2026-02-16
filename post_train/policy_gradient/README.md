# Policy Gradient（策略梯度）

## 定位与分类
- 阶段：后训练（策略优化基础）
- 类型：强化学习基础方法
- 作用：直接沿策略梯度方向提升期望奖励

## 核心原理
1. 通过采样轨迹估计 `∇J(θ)`。
2. 依据回报信号增大高回报动作概率。
3. 可配合 baseline 降低梯度方差。

## 与相近方法区别
1. 相比 `Actor-Critic`：Policy Gradient 不显式学习 value critic（或弱依赖）。
2. 相比 `PPO`：Policy Gradient 通常没有 clip 约束，更新稳定性更依赖超参。
3. 相比 `RLHF`：这里只是优化算法视角，不是完整人类反馈流水线。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/policy_gradient
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/policy_gradient.py --reward-model <奖励模型路径或名称>
```

## 输出结果
默认输出到 `output/policy_gradient_metrics`，包含：
- `training_metrics.csv`
- `training_curves.png`
- `summary.json`
- `log_history.json`
