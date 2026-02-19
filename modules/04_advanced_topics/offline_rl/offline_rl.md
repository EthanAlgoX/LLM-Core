# Offline RL 分类

> [!TIP]
> **一句话通俗理解**：不和真实环境交互，只用"历史经验数据"学习，像人类读案例复盘一样

## 定义与目标

- **定义**：Offline RL 是在固定离线数据集上训练策略，不进行在线环境探索。
- **目标**：在减少交互成本与风险的前提下，学到稳健可部署的策略。

## 适用场景与边界

- **适用场景**：用于高交互成本或高风险场景下的策略学习。
- **不适用场景**：不适用于离线数据覆盖度极低且无法补充数据的任务。
- **使用边界**：离线结果高度依赖数据分布与行为策略质量。

## 关键步骤

1. 收集并清洗离线轨迹数据集 \(\mathcal{D}\)。
2. 训练价值函数并抑制分布外动作的过估计。
3. 在行为策略邻域内更新策略，避免“未见动作”导致崩溃。
4. 用离线评估指标（如 FQE）验证策略稳定性。

核心区别：
- `cql` 偏保守 Q 约束。
- `bcq` 偏行为策略动作约束。

## 关键公式

`CQLLoss = E_D[(Q(s,a)-y)^2] + alpha * (logsumexp_a Q(s,a) - E_D[Q(s,a)])`

符号说明：
- `D`：离线数据集。
- `alpha`：保守正则强度。
- 第二项用于压低分布外动作的 Q 值。

## 关键步骤代码（纯文档示例）

```python
for batch in offline_loader:
    q_pred = q_net(batch.s, batch.a)
    y = target_q(batch.r, batch.s_next, batch.done)
    bellman = mse(q_pred, y)
    conservative = logsumexp_q(q_net, batch.s) - q_pred.mean()
    loss = bellman + alpha * conservative
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## 子模块导航

- `cql`: Conservative Q-Learning。
- `bcq`: Batch-Constrained Q-Learning。

## 工程实现要点

- 先评估数据覆盖范围，再选择保守或约束型算法。
- 训练与评估分离，优先使用离线评估方法做迭代。
- 重点监控 OOD 动作占比与价值函数过估计程度。

## 常见错误与排查

- **症状**：离线指标高但部署表现差。  
  **原因**：分布外动作过多导致价值估计失真。  
  **解决**：增加保守正则并约束策略靠近行为分布。
- **症状**：算法对数据集切分极度敏感。  
  **原因**：样本覆盖不足或数据质量波动。  
  **解决**：扩充关键状态样本并统一切分与预处理策略。

## 与相近方法对比

| 方法 | 优点 | 局限 | 适用场景 |
| --- | --- | --- | --- |
| 本文主题方法 | 紧贴本节问题定义 | 依赖数据与实现质量 | 适合结构化评测与迭代优化 |
| 对比方法A | 上手成本更低 | 能力上限可能受限 | 快速原型与基线对照 |
| 对比方法B | 上限潜力更高 | 调参与资源成本更高 | 高要求生产或复杂任务场景 |

## 参考资料

- [Conservative Q-Learning](https://arxiv.org/abs/2006.04779)
- [Batch-Constrained deep Q-learning](https://arxiv.org/abs/1812.02900)
