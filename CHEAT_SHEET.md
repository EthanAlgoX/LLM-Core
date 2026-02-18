# 面试备考速记表 (Interview Cheat Sheet)

## 🧠 显存计算专栏 (Memory Calculation)

### 1. 静态权重 (Weights)

- **显存占用** $\approx Params \times \text{Bytes per Param}$
  - `fp16/bf16`: 2 Bytes
  - `fp8`: 1 Byte
  - `int4`: 0.5 Byte

### 2. 训练状态 (Training States - Adam 优化器)

- **Adam (fp32)**：约为参数量的 **12-16 倍**。
  - 4B 梯度 + 8B 优化器 (Momentum, Variance) + 4B 权重副本。

### 3. 推理 KV Cache (每个 Token)

- **计算公式**：$2 \times \text{layers} \times \text{heads} \times \text{dim} \times \text{precision}$
  - `fp16` 下，7B 模型约 0.5MB/token。

---

## ⚖️ 核心算法对比矩阵 (Comparison Matrix)

| 特性 | SFT | PPO | DPO | GRPO |
| :--- | :--- | :--- | :--- | :--- |
| **基础要求** | 监督数据 (Q/A) | 偏好数据 + 奖励模型 | 偏好对 (C/R) | 规则/奖励函数 |
| **显存压力** | 低 | **极高** (4个模型) | 中 | 中 (省去 Critic) |
| **收敛难度** | 容易 | 难 (RL 抖动) | 较容易 | 较容易 |
| **核心场景** | 习得格式 | 安全边界、复杂对齐 | 离线偏好学习 | **数学推理、CoT** |

---

## ⚡️ VLM 架构演进

| 模型 | 特点 | 融合方式 |
| :--- | :--- | :--- |
| **BLIP-2** | 引入 Q-Former | 瓶颈式抽取 (Fixed number of visual tokens) |
| **LLaVA** | 简单 MLP Projector | 直接线性映射全量视觉特征 |
| **Flamingo** | Perceiver Resampler | 跨注意力层 (Cross-Attention) 注入 |

---

## 🛠️ 分布式训练 (ZeRO)

- **ZeRO-1**：划分优化器状态 (Optimizer States)。
- **ZeRO-2**：划分优化器状态 + 梯度 (Gradients)。
- **ZeRO-3**：划分所有状态（权重 + 梯度 + 优化器状态）。
