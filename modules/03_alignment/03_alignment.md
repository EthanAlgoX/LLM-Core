# Alignment 分类

> [!TIP]
> **一句话通俗理解**：让模型从“会续写”变成“会按人类偏好回答”的后训练路线图。

## 定义与目标

- **定义**：Alignment 是通过监督数据、偏好数据与强化学习信号约束模型行为的后训练阶段。
- **目标**：提升模型的有用性、可控性与安全性，使其输出更符合用户意图与偏好。

## 关键步骤

1. `SFT`：用高质量指令数据建立基础行为。
2. `DPO/GRPO/PPO`：引入偏好与奖励信号优化回答质量。
3. `RLHF`：整合 SFT、奖励模型与策略优化，形成闭环。
4. 评估与回流：通过离线评估和人工反馈迭代数据与策略。

建议学习顺序：`sft -> dpo -> ppo/grpo -> rlhf`

## 关键公式

\[
\max_{\theta}\ \mathbb{E}_{(x,y)\sim \pi_{\theta}}[r(x,y)]-\beta\,\mathrm{KL}(\pi_{\theta}\|\pi_{\mathrm{ref}})
\]

符号说明：
- \(r(x,y)\)：奖励模型或偏好信号给出的评分。
- \(\pi_{\mathrm{ref}}\)：参考策略（约束模型别偏离太远）。
- \(\beta\)：KL 约束强度。

## 关键步骤代码（纯文档示例）

```python
for batch in train_loader:
    samples = policy.generate(batch["prompt"])
    reward = reward_model.score(batch["prompt"], samples)
    kl = kl_divergence(policy, ref_policy, batch["prompt"], samples)
    loss = -(reward - beta * kl).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## 子模块导航

- `sft`: 监督微调基础。
- `dpo`: 离线偏好优化。
- `grpo`: 组相对优势优化。
- `ppo`: 在线策略优化。
- `policy_gradient`: REINFORCE 基础。
- `actor_critic`: 策略-价值联合训练。
- `rlhf`: 奖励模型与策略优化闭环。
