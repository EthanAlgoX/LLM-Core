# Advantage（优势函数学习与对比）

> [!TIP]
> **一句话通俗理解**：GAE、多步回报 与训练稳定性方差权衡

## 定位与分类

- **阶段**：强化学习基础（RL Basics）。
- **类型**：策略评估与梯度加权技术。
- **作用**：它是 PPO / Actor-Critic 等算法的核心输入。通过计算“超额收益”，它能显著降低策略梯度的方差，让模型更稳定地学习。

## 定义与目标

在强化学习中，优势（Advantage）衡量的是：**“在特定状态下，执行某个动作比平均水平好多少？”**

- 如果 $A > 0$：这个动作表现比平均好，增加其概率。
- 如果 $A < 0$：这个动作表现比平均差，减少其概率。

它就像是一个对比器，消除了状态本身的“基础分”（Value），只关注动作带来的“增量分”。

## 适用场景与边界

- **适用场景**：用于强化学习入门、算法推导和 toy 环境复现。
- **不适用场景**：不适用于直接替代生产级策略系统（需结合业务约束与大规模评测）。
- **使用边界**：结论依赖环境定义、奖励设计与随机种子设置。

## 关键步骤

1. **采样 (Trajectory Generation)**：模型与环境交互，产生带有奖励 $r$ 的序列。
2. **价值估计 (Value Estimation)**：通过 Critic 网络预测每个状态的期望分 $V(s)$。
3. **计算残差 (TD Error)**：计算 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 。
4. **优势合成 (Advantage Combination)**：
   - **MC 方式**：直接用实际总分减去预估分。
   - **TD 方式**：只看一步的即时收益。
   - **GAE 方式**：平滑地加权多步残差（最常用，如 PPO）。
5. **策略更新**：用优势值作为权重去更新策略参数。

## 关键公式

### 1. 基础定义

$$A(s, a) = Q(s, a) - V(s)$$

- $Q(s, a)$：执行动作 $a$ 后的期望收益。
- $V(s)$：状态 $s$ 的平均期望收益（基线）。

### 2. 不同估计方式

- **MC (Monte Carlo)**：
  $$A_t = G_t - V(s_t)$$

- **TD (One-step)**：
  $$A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- **GAE (Generalized Advantage Estimation)**：
  $$\hat{A}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}$$

  - $\lambda$ 是折衷因子，用于平衡方差（Variance）与偏差（Bias）。

## 与相近方法区别

1. 相比 `GAE`：本模块覆盖多种方法并给出横向可视化。
2. 相比 `Policy Gradient`：本模块关注“估计器”，PG关注“更新目标”。
3. 相比 `TD Learning`：TD 更偏价值函数更新，本模块偏策略优化输入信号分析。

## 关键步骤代码（纯文档示例）

```python
# 关键步骤代码（示意）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```

## 输出结果

默认输出到 `output/advantage_metrics`，包含：

- `training_metrics.csv`
- `training_curves.png`
- `advantage_methods.png`
- `advantage_comparison.json`
- `summary.json`

## 目录文件说明（重点）

- 关键步骤代码：见“关键步骤代码（纯文档示例）”章节。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。

## 工程实现要点

- 固定环境版本与随机种子，先确保结果可复现。
- 区分训练集与评估轨迹，避免数据泄漏。
- 同时记录平均回报、方差与收敛速度，不只看单点最优值。

## 常见错误与排查

- **症状**：回报长期不提升。  
  **原因**：学习率过高或探索不足导致策略陷入次优。  
  **解决**：降低学习率并提高探索强度（如 epsilon/entropy）。
- **症状**：不同机器复现结果差异大。  
  **原因**：环境版本、随机种子或预处理流程不一致。  
  **解决**：锁定依赖版本并统一 seed 与评测脚本。

## 参考资料

- [Sutton & Barto《Reinforcement Learning》](http://incompleteideas.net/book/the-book.html)
- [David Silver RL Course](https://www.davidsilver.uk/teaching/)

