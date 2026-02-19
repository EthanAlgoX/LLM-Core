# RL Basics 分类

> [!TIP]
> **一句话通俗理解**：强化学习基础概念、算法分类与学习路径指南

## 定义与目标

- **定义**：RL 基础模块讲的是“智能体如何通过与环境交互学习策略”的最小知识闭环。
- **目标**：先建立“状态-动作-回报”的统一心智模型，再理解从价值估计到优势估计的演进。

## 关键步骤

1. 从 `MDP` 建立问题建模框架（状态、动作、转移、奖励、折扣）。
2. 用 `TD Learning` 学习“边采样边更新”的价值估计方式。
3. 用 `Advantage` 理解“相对平均水平有多好”。
4. 用 `GAE` 在偏差与方差之间做工程可用的折中。

建议学习顺序：`mdp -> td_learning -> advantage -> gae`

## 关键公式

\[
V^{\pi}(s)=\mathbb{E}_{a\sim\pi, s'\sim P}[r(s,a)+\gamma V^{\pi}(s')]
\]

符号说明：
- \(V^{\pi}(s)\)：策略 \(\pi\) 在状态 \(s\) 下的价值。
- \(\gamma\)：折扣因子，控制长期回报权重。
- \(P\)：环境转移概率。

## 关键步骤代码（纯文档示例）

```python
V = init_value_table()
for _ in range(num_iters):
    s = env.reset()
    done = False
    while not done:
        a = policy(s, V)
        s_next, r, done = env.step(a)
        V[s] = V[s] + alpha * (r + gamma * V[s_next] - V[s])  # TD 更新
        s = s_next
```

## 子模块导航

- `mdp`: MDP 建模与 Bellman 备份。
- `td_learning`: Q-Learning 与 SARSA。
- `advantage`: 优势函数估计与对比。
- `gae`: 广义优势估计与稳定训练。
