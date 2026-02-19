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

- **$\nabla_\theta \log \pi_\theta(a_t | s_t)$**：表示如何调整参数才能让某个动作概率变大。
- **$G_t$ (Return)**：该动作带来的总回报。它是梯度的权重。

### 3. Log-Derivative Trick (对数微分技巧)

这是实现公式转化的关键桥梁：

$$\nabla_\theta \pi_\theta = \pi_\theta \frac{\nabla_\theta \pi_\theta}{\pi_\theta} = \pi_\theta \nabla_\theta \log \pi_\theta$$

这使得我们可以直接通过采样（由于有 $\pi_\theta$ 项）来估计本来看似无法计算的期望梯度。

## 与相近方法区别

1. 相比 `Actor-Critic`：Policy Gradient 不显式学习 value critic（或弱依赖）。
2. 相比 `PPO`：Policy Gradient 通常没有 clip 约束，更新稳定性更依赖超参。
3. 相比 `RLHF`：这里只是优化算法视角，不是完整人类反馈流水线。

## 🛠️ 工程实战：REINFORCE 算法实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """简单策略网络：输入状态，输出动作概率"""
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)

    def select_action(self, state):
        probs = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)

# REINFORCE 训练循环
policy = PolicyNetwork(state_dim=4, action_dim=2)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

for episode in range(1000):
    log_probs, rewards = [], []
    state = env.reset()

    # 采样一条完整轨迹
    while not done:
        state_tensor = torch.FloatTensor(state)
        action, log_prob = policy.select_action(state_tensor)
        next_state, reward, done, _ = env.step(action.item())

        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state

    # 计算折扣回报 G_t
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G         # γ = 0.99
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Baseline: 标准化

    # 策略梯度更新
    loss = -sum(lp * Gt for lp, Gt in zip(log_probs, returns))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 在 LLM 中的对应

在 LLM 微调场景中，REINFORCE 的思想体现为：

```python
# 伪代码：LLM 策略梯度
for prompt in prompts:
    response = model.generate(prompt)           # Actor 采样
    reward = reward_model(prompt, response)      # RM 打分

    log_prob = model.log_prob(response | prompt) # 计算对数概率
    loss = -log_prob * reward                    # 策略梯度
    loss.backward()
```

> **注意**：原始 REINFORCE 方差极大，实际 LLM 训练中都使用 PPO/GRPO 等带 Baseline/Clipping 的改进版本。

---

## 原始脚本运行

```bash
cd <YOUR_PROJECT_ROOT>/post_train/alignment/policy_gradient
conda activate finetune
python code/policy_gradient.py
```
