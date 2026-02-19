# Actor-Critic

> [!TIP]
> **一句话通俗理解**：RL 的基础：好结果加分、坏结果扣分，用梯度驱动策略进化

## 定位与分类

- **阶段**：后训练（Post-training）之策略优化基础。
- **类型**：混合架构（Policy-based + Value-based）。
- **作用**：它是 PPO / RLHF 的底层范式。通过“行为者-判官”协作，在提升模型性能的同时，极大降低了学习过程中的不确定性（方差）。

## 定义与目标

Actor-Critic 是一种将“策略梯度”与“价值评估”相结合的经典模型架构：

- **Actor (行为者)**：策略网络。负责根据当前的指令，预测并生成具体的回答（Action）。
- **Critic (判官/记账员)**：价值网络（Value Head）。它不生产内容，而是评估当前状态的“优劣”，并预估未来的总奖励。

在 LLM 训练中，Critic 就像是一个专业的会计，时刻盯着 Actor 的产出，判断其是否超预期地获得了高分。

## 适用场景与边界

- **适用场景**：用于构建指令跟随、偏好对齐与奖励驱动优化流程。
- **不适用场景**：不适用于缺少高质量偏好数据或评测体系的直接落地。
- **使用边界**：对齐收益受数据质量、奖励建模与 KL 约束策略影响明显。

## 关键步骤

1. **采样 (Sampling)**：Actor 接受指令，生成一组对话。
2. **打分 (Reward Calculation)**：模型获得一个奖励分（来自 RM 模型）。
3. **估值 (Value Estimation)**：Critic 对当前的对话状态给出一个“预估分”。
4. **计算优势 (Advantage Computation)**：计算真实得分比 Critic 预估的得分高出多少（ $\mathrm{Reward} - \mathrm{Value}$ ）。
5. **双向更新 (Update)**：
   - **更新 Actor**：如果优势为正，增加该生成行为出现的概率。
   - **更新 Critic**：减小其预估分与真实分数之间的误差，使其预测更准。

## 关键公式

### 1. 优势估计 (Advantage)

$$\hat{A}_t = \mathrm{Reward}_t - V_\phi(s_t)$$

- 如果 $\hat{A}_t > 0$ ，说明 Actor 的表现优于预期，应当获得正反馈。

### 2. Actor 目标 (策略梯度)

$$L_{actor} = - \log \pi_\theta(a|s) \cdot \hat{A}_t$$

- 通过优势函数加权，使高 Advantage 的动作概率变大。

### 3. Critic 目标 (价值均方误差)

$$L_{critic} = \frac{1}{2} (V_\phi(s_t) - G_t)^2$$

- $G_t$ 为真实累计奖励，Critic 通过回归学习减小误差。

## 与相近方法区别

1. 相比 `Policy Gradient`：多了 Critic，通常更稳定、更高样本效率。
2. 相比 `PPO`：Actor-Critic 是结构范式，PPO 是具体优化目标/约束策略。
3. 相比 `GAE`：GAE 是优势估计技术，可作为 Actor-Critic 的组成部分。

## 🛠️ 工程实战：Actor-Critic 实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """Actor-Critic 共享底层特征"""
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
        )
        # Actor Head: 输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(hidden, action_dim),
            nn.Softmax(dim=-1),
        )
        # Critic Head: 输出 V(s) 状态价值
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

# 训练循环
model = ActorCritic(state_dim=4, action_dim=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
gamma = 0.99

for episode in range(1000):
    state = env.reset()
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state)
        probs, value = model(state_tensor)

        # Actor: 采样动作
        dist = Categorical(probs)
        action = dist.sample()

        next_state, reward, done, _ = env.step(action.item())
        _, next_value = model(torch.FloatTensor(next_state))

        # Critic: 计算 TD 目标与 Advantage
        td_target = reward + gamma * next_value * (1 - done)
        advantage = td_target - value              # A(s) = R + γV(s') - V(s)

        # 双向更新
        actor_loss = -dist.log_prob(action) * advantage.detach()  # 策略梯度
        critic_loss = advantage.pow(2)                             # 价值回归

        loss = actor_loss + 0.5 * critic_loss      # 联合损失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
```

### 在 LLM（PPO）中的对应

```python
# PPO 中的 Actor-Critic 架构
from trl import AutoModelForCausalLMWithValueHead

# 自动为 CausalLM 加装 Value Head（Critic）
model = AutoModelForCausalLMWithValueHead.from_pretrained("Qwen/Qwen2.5-7B")

# model.pretrained_model → Actor（生成回复）
# model.v_head           → Critic（预估价值）
# 训练时两者同步更新
```

---

## 关键步骤代码（纯文档示例）

```python
# 关键步骤代码（示意）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```

## 工程实现要点

- 优先保证数据质量与评测一致性，再放大训练规模。
- 在线/离线对齐需分别监控稳定性、奖励漂移与过优化风险。
- 保持参考模型与训练模型版本可追踪，便于回溯问题。

## 常见错误与排查

- **症状**：奖励升高但人工体验下降。  
  **原因**：奖励黑客或偏好模型偏差导致目标错位。  
  **解决**：引入人工抽检与多指标约束，限制单一奖励驱动。
- **症状**：训练不稳定或发散。  
  **原因**：学习率/KL 系数/批量配置不匹配。  
  **解决**：缩小超参搜索范围并分阶段增大训练强度。

## 参考资料

- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

