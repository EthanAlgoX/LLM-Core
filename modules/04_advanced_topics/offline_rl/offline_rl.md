# Offline RL 分类

> [!TIP]
> **一句话通俗理解**：不和真实环境交互，只用"历史经验数据"学习，像人类读案例复盘一样

## 定义与目标

- **定义**：Offline RL 是在固定离线数据集上训练策略，不进行在线环境探索。
- **目标**：在减少交互成本与风险的前提下，学到稳健可部署的策略。

## 关键步骤

1. 收集并清洗离线轨迹数据集 \(\mathcal{D}\)。
2. 训练价值函数并抑制分布外动作的过估计。
3. 在行为策略邻域内更新策略，避免“未见动作”导致崩溃。
4. 用离线评估指标（如 FQE）验证策略稳定性。

核心区别：
- `cql` 偏保守 Q 约束。
- `bcq` 偏行为策略动作约束。

## 关键公式

\[
\mathcal{L}_{\mathrm{CQL}}=\underbrace{\mathbb{E}_{(s,a)\sim \mathcal{D}}\left[(Q_{\theta}(s,a)-y)^2\right]}_{\text{Bellman 误差}}
 + \alpha\left(\log\sum_{a}\exp(Q_{\theta}(s,a))-\mathbb{E}_{a\sim\mathcal{D}}[Q_{\theta}(s,a)]\right)
\]

符号说明：
- \(\mathcal{D}\)：离线数据集。
- \(\alpha\)：保守正则强度。
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
