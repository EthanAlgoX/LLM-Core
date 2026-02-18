# 面试备考速记表 (Interview Cheat Sheet)

## 🧠 显存与计算 (Memory & Compute)

### 1. 静态权重 (Weights)

- **显存占用** $\approx Params \times \text{Bytes per Param}$
  - `fp16/bf16`: 2 Bytes
  - `fp8`: 1 Byte
  - `int4`: 0.5 Byte

### 2. 注意力机制变体 (Attention Variations)

| 模式 | 全称 | KV 共享状况 | 优势 |
| :--- | :--- | :--- | :--- |
| **MHA** | Multi-Head | 无共享 | 表达能力最强 (最重) |
| **MQA** | Multi-Query | **所有** Query 共享 1 组 K/V | 推理极快，显存极省 |
| **GQA** | Grouped-Query | **分组**共享 (如 8 组) | **Llama 3 主流**，平衡精度与速度 |

### 3. 工程优化：Flash Attention

- **核心逻辑**：IO 感知优化。通过 **Tiling (分块)** 减少显存读写，利用 **Recomputation (重计算)** 在 SRAM 计算 Softmax，速度提升 2-4x。

---

## ⚖️ 核心算法对比矩阵 (Comparison Matrix)

| 特性 | [SFT](./post_train/alignment/sft/README.md) | [PPO](./post_train/alignment/ppo/README.md) | [DPO](./post_train/alignment/dpo/README.md) | [GRPO](./post_train/alignment/grpo/README.md) |
| :--- | :--- | :--- | :--- | :--- |
| **基础要求** | 监督数据 (Q/A) | 偏好数据 + 奖励模型 | 偏好对 (C/R) | 规则/奖励函数 |
| **显存压力** | 低 | **极高** (4个模型) | 中 | 中 (省去 Critic) |
| **收敛难度** | 容易 | 难 (RL 抖动) | 较容易 | 较容易 |
| **核心场景** | 习得格式 | 安全边界、复杂对齐 | 离线偏好学习 | **数学推理、CoT** |

---

## 🚀 高效微调 (PEFT - LoRA)

- **核心公式**：$\Delta W = A \times B$ (秩 $r \ll d, k$)
- **优点**：显存省、不增加推理延迟 (可在推理前合并权重)。
- **QLoRA**：NF4 量化 + 双量化 + 分页优化器，单卡 24G 即可微调 70B 模型。

---

## ⚡️ 推理与系统优化

- **KV Cache**：$2 \times \text{layers} \times \text{heads} \times \text{dim} \times \text{precision}$ (针对每个 Token)。
- **Paged Attention**：解决显存碎片化，提高 Batch Size 极限。
- **ZeRO 优化 (DeepSpeed)**：
  - **ZeRO-1**：划分优化器状态。
  - **ZeRO-2**：划分状态 + 梯度。
  - **ZeRO-3**：划分所有 (权重+梯度+优化器状态)。

---

## 📊 训练指标解读 (RLHF 关键)

- **KL Divergence**：反映新旧策略偏差。如果过高说明奖励攻击 (Reward Hacking) 严重，需要调大 $\beta$。
- **Reward Mean**：奖励均值。应稳步上升，如出现剧烈跳变通常意味着样本质量或学习率过大。
