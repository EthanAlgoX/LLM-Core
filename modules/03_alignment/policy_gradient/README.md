# Policy Gradient（策略梯度）

## 定位与分类

- **阶段**：后训练（Post-training）之策略优化基础。
- **类型**：直接策略搜索（Policy-based RL）。
- **作用**：它是强化学习中最直观的一类算法，直接对策略参数进行梯度上升。它是 PPO 和 Actor-Critic 等高级算法的鼻祖。

## 什么是 Policy Gradient？

策略梯度（Policy Gradient）是一类直接对策略进行参数化的强化学习方法。不同于学习价值函数（Q-learning），它直接通过优化神经网络输出的概率分布来最大化期望奖励。其核心哲学是：**“如果一个行为带来了好结果，那就增加它出现的概率；反之，则降低它。”**

## 训练的关键步骤

1. **采样 (Trajectory Generation)**：让模型（Actor）根据当前概率生成一段完整的对话轨迹 $\tau$。
2. **回报计算 (Return Calculation)**：计算该路径上获得的总奖励 $R(\tau)$。
3. **梯度估计 (Gradient Estimation)**：利用对数微分技巧（Log-Derivative Trick）计算梯度的估计值。
4. **策略更新 (Weight Update)**：沿着梯度方向更新模型参数 $\theta$。
5. **迭代 (Iteration)**：采样新数据，不断循环，使模型向高奖励的方向偏移。

## 核心数学公式

### 1. 目标函数 (Objective)

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$

我们的目标是最大化所有可能轨迹的期望奖励。

### 2. 策略梯度基本定理 (Policy Gradient Theorem)

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) G_t \right]$$

- ** $\nabla_\theta \log \pi_\theta(a_t | s_t)$ **：表示如何调整参数才能让某个动作概率变大。
- ** $G_t$ (Return)**：该动作带来的总回报。它是梯度的权重。

### 3. Log-Derivative Trick (对数微分技巧)

这是实现公式转化的关键桥梁：

$$\nabla_\theta \pi_\theta = \pi_\theta \frac{\nabla_\theta \pi_\theta}{\pi_\theta} = \pi_\theta \nabla_\theta \log \pi_\theta$$

这使得我们可以直接通过采样（由于有 $\pi_\theta$ 项）来估计本来看似无法计算的期望梯度。

## 与相近方法区别

1. 相比 `Actor-Critic`：Policy Gradient 不显式学习 value critic（或弱依赖）。
2. 相比 `PPO`：Policy Gradient 通常没有 clip 约束，更新稳定性更依赖超参。
3. 相比 `RLHF`：这里只是优化算法视角，不是完整人类反馈流水线。

## 运行

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/alignment/policy_gradient
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

## 目录文件说明（重点）

- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
