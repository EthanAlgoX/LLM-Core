# TD Learning（时序差分学习）

> [!TIP]
> **一句话通俗理解**：Q-Learning (Off-policy) 与 SARSA (On-policy) 差异

## 定位与分类

- **阶段**：强化学习核心基础（RL Core）。
- **类型**：无模型（Model-Free）价值学习。
- **作用**：它是强化学习中最具代表性的思想之一。它结合了蒙特卡洛（采样）和动态规划（迭代）的优点，实现了“边走边学”，无需等待任务结束。

## 定义与目标

TD（Temporal-Difference，时序差分）学习是强化学习的基石。其核心哲学是：**“用昨天的经验预估今天，并用今天的实际结果修正昨天的预估。”**

- 它不需要像蒙特卡洛那样等整场游戏结束才更新。
- 它不需要像动态规划那样预先知道环境的所有概率模型。
- 它通过“时间上的差异”（TD Error）来不断自我进化。

## 适用场景与边界

- **适用场景**：用于强化学习入门、算法推导和 toy 环境复现。
- **不适用场景**：不适用于直接替代生产级策略系统（需结合业务约束与大规模评测）。
- **使用边界**：结论依赖环境定义、奖励设计与随机种子设置。

## 关键步骤

1. **初始化**：创建一个 Q 表（Q-Table），记录每个“状态-动作”对的初始分值。
2. **选择动作**：采用 $\epsilon$ -greedy 策略，平衡探索（尝试新动作）与利用（选择已知最高分动作）。
3. **执行并观测**：执行动作 $a$，观测奖励 $r$ 和新状态 $s'$。
4. **计算目标 (TD Target)**：预估未来的最大收益： $Target = r + \gamma \max_{a'} Q(s', a')$。
5. **更新 Q 值**：根据目标与当前值的差异（TD Error）进行修正。

## 关键公式

### 1. TD 误差 (TD Error)

$$\delta_t = R_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

- $\delta_t$：这就是“时序上的差异”，代表了实际观测比预估好（或坏）了多少。

### 2. Q-Learning 更新公式

$$Q(s, a) \leftarrow Q(s, a) + \alpha \underbrace{[R + \gamma \max_{a'} Q(s', a') - Q(s, a)]}_{\mathrm{TD Error}}$$

- **$\alpha$ (Learning Rate)**：学习率，决定了新知识覆盖旧知识的速度。
- **$\gamma$ (Discount Factor)**：折扣因子，决定了我们多在乎长远利益。

## 与相近方法区别

1. 相比 `MDP` 规划：TD 不依赖显式转移概率模型。
2. 相比 `GAE`：GAE 是优势估计方法，TD 是更基础的价值更新思想。
3. 相比 `Policy Gradient`：TD 主要学习值函数，PG 直接优化策略。

## 关键步骤代码（纯文档示例）

```python
# 关键步骤代码（示意）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```

## 输出结果

默认输出到 `output/td_learning_metrics`，包含：

- `episode_log.csv`
- `training_curves.png`
- `q_table.json`
- `policy.json`
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

