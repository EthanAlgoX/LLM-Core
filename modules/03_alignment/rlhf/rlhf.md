# RLHF (Reinforcement Learning from Human Feedback) 人类反馈强化学习

> [!TIP]
> **一句话通俗理解**：在沙盒里练习，让 Agent 从实战模拟中学会完成复杂的真实任务

## 定位与分类

- **阶段**：大模型对齐的“终极方案”。
- **类型**：多阶段复合架构（SFT + RM + RL）。
- **作用**：解决 SFT 只能“字面对齐”的问题，通过引入人类偏好，让模型在逻辑、价值观和复杂任务处理上真正具备“灵感的跃迁”。

## 关键步骤

1. **第一阶段：SFT (监督微调)**
   - **目标**：冷启动。
   - **内容**：在高质量的人类对话数据上进行监督学习，让模型学会如何得体地说话。
   - **状态**：这是后续阶段的基石（Actor 的初始状态）。

2. **第二阶段：RM (奖励建模 - Reward Modeling)**
   - **目标**：训练一个“电子裁判”。
   - **内容**：给同一个问题提供两个回答，让人类标出哪个更好（Rank），然后训练奖励模型去拟合这种偏好。
   - **关键公式（Bradley-Terry 模型）**：
     $$P(y_w \succ y_l | x) = \frac{\exp(r_\phi(x, y_w))}{\exp(r_\phi(x, y_w)) + \exp(r_\phi(x, y_l))}$$

     - 通过最小化其**负对数似然（Negative Log-Likelihood）**来学习。

3. **第三阶段：RL (强化学习优化 - PPO/GRPO)**
   - **目标**：终极对齐。
   - **内容**：利用第二阶段训练好的 RM 给 Actor 打分，通过强化学习算法（PPO 或 GRPO）最大化期望奖励。
   - **关键元素**：Reward（得分）、Critic（预估）、KL Penalty（防模型练废的紧箍咒）。

## 关键公式

RLHF 的最终优化目标是两者的平衡：

$$\max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(y|x)} [r_\phi(x, y) - \beta D_{KL}(\pi_\theta || \pi_{ref})]$$

- **第一部分 $r_\phi(x, y)$**：让生成的内容得到奖励模型尽可能高的打分。
- **第二部分 $\beta D_{KL}$**：惩罚偏离参考模型太远的行为，确保语言依然通顺、符合基本分布。

## 进阶对齐：Agentic-RL (智能体强化学习)

传统的 RLHF 关注回答的“即时偏好”，而 **Agentic-RL** 关注长期决策的“任务成功率”。

### 1. 训练范式

- **环境驱动 (Env-driven)**：模型不仅在文本域对齐，还在高真模拟环境（如代码沙箱、浏览器环境）中学习。
- **冷启动与探索**：利用大量高质量的合成轨迹初始化（SFT），再通过强化学习在此基础上进行 **探索-利用权衡 (Exploration-Exploitation Trade-off)**。
- **长程规划 (Long-term Planning)**：优化 Reward 时不仅看当前 Step，更看最终 Task Success。

### 2. 用户模拟器 (User Simulator)

- **Persona-driven**：构建具备特定人设、任务目标和策略多样性的用户模型。
- **交互生成**：利用用户模拟器生成大规模、多轮次的真实交互数据，用于演练 Agent 的冷启动策略。
- **Adversarial User Generation**：合成具备挑战性的边缘案例（Edge Cases）以提升 Agent 的鲁棒性。

### 3. 多智能体强化学习 (MARL)

- **核心算法**：
  - **MAPPO/MADDPG**：在多智能体交互场景下优化集群策略。
  - **共识与协作**：训练 Agent 在分布式环境下通过博弈与通信达成任务共识。

## RLHF vs. DPO vs. Agentic-RL

| 维度 | RLHF | DPO | Agentic-RL |
| :--- | :--- | :--- | :--- |
| **反馈来源** | 静态偏好标签 | 静态偏好对 | 动态环境 Reward / 模拟器反馈 |
| **优化目标** | 符合人言（即时） | 符合人言（即时） | 任务闭环（长程成功率） |
| **工程复杂度** | 高 | 低 | 极高（涉及环境模拟与多智能体博弈） |

## 🛠️ 工程实战：RLHF 三阶段流水线

### LLaMA Factory 一站式 RLHF

```python
# 关键步骤代码（示意）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```

**指标解读**：

- `reward`：应随 Step 稳步上升。
- `kl`：应保持在 1.0~5.0，过高说明模型在胡说八道。
- `loss`：PPO 损失波动较大，重点看 Reward 趋势。

---
## 定义与目标

- **定义**：本节主题用于解释该模块的核心概念与实现思路。
- **目标**：帮助读者快速建立问题抽象、方法路径与工程落地方式。
## 关键步骤代码（纯文档示例）

```python
# 关键流程示意（与具体工程实现解耦）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```
