# CQL（Conservative Q-Learning）

## 定位与分类

- **阶段**：离线强化学习（Offline RL）。
- **类型**：保守价值函数学习。
- **作用**：通过在训练中主动“压低”未见动作的 Q 值，确保模型在离线数据上学到的策略不会因为盲目乐观而崩溃。

## 什么是 CQL？

CQL（Conservative Q-Learning）是解决离线强化学习中“过度乐观”问题的核心方案。
在离线训练中，标准 Q-learning 往往会给分布外（OOD）动作打出虚高甚至溢出的分数，导致生成的策略在实际部署时表现极差。CQL 的核心哲学是：**“宁可低估，绝不虚高。对于没见过或不确定的动作，我先假设它很差，直到证据证明它真的好。”**

## 训练的关键步骤

1. **采样 (Sampling)**：从静态离线数据集中读取 $(s, a, r, s')$ 元组。
2. **计算 Bellman 误差 (TD Loss)**：像普通 DQN 一样计算均方误差损失。
3. **计算保守压制项 (Conservative Regularization)**：
   - 计算所有动作 Q 值的 Log-Sum-Exp（代表该状态下动作的平均“潜力”）。
   - 减去数据集里真实发生动作的 Q 值。
4. **加权求和更新**：将 TD Loss 与压制项相加（乘以系数 $\alpha$），进行反向传播。
5. **策略提取**：训练完成后，直接取 $a = \arg\max Q(s, a)$ 作为最优策略。

## 核心数学公式

### 1. 保守损失项 (Conservative Penalty)

$$L_{CQL\_reg} = \mathbb{E}_{s \sim D} \left[ \log \sum_a \exp(Q(s, a)) - \mathbb{E}_{a \sim \pi_b(a|s)} [Q(s, a)] \right]$$

- **第一项 (LogSumExp)**：旨在提升所有动作 Q 值的“地平线”，但主要是为了方便计算梯度的软最大值。
- **第二项 (Mean Q_data)**：由于有负号，它倾向于让数据集中存在的动作 Q 值变大。
- **整体效果**：使得 $Q_{seen} > Q_{unseen}$，且 $Q_{unseen}$ 被大幅压低。

### 2. 总损失函数

$$L_{total} = L_{TD} + \alpha \cdot L_{CQL\_reg}$$

- **$\alpha$ (Alpha)**：保守系数。$\alpha$ 越大，模型越“怂”（保守）；$\alpha$ 越小，模型越接近普通 DQN。

## 与相近方法区别

1. 相比 `BCQ`：CQL 通过值函数约束保守化；BCQ 通过行为策略约束动作空间。
2. 相比在线 `PPO`：CQL 只使用静态数据，不与环境实时交互。
3. 相比 `TD Learning`：CQL 面向离线分布偏移问题，TD 常用于在线学习。

## 运行

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/offline_rl/cql
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/cql.py
```

## 输出结果

默认输出到 `output/cql_metrics`，包含：

- `training_metrics.csv`
- `training_curves.png`
- `policy.json`
- `qmax_by_state.json`
- `summary.json`

## 目录文件说明（重点）

- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
