# DeepSeek-R1 经典解析

> [!TIP]
> **一句话通俗理解**：用纯强化学习让模型自发学会"一步步思考"，无需任何 CoT 标注数据

> **一句话理解**：DeepSeek-R1 通过纯强化学习（无需监督数据）让模型自发涌现出"思维链推理"能力，并用 GRPO 替代 PPO 大幅降低了训练复杂度与成本。

---

## 演进路线

```text
DeepSeek-V2 (MoE 稠密预训练基座)
  ↓  冷启动 SFT（少量 CoT 数据）
DeepSeek-R1-Zero (纯 RL，无监督)
  ↓  多阶段训练 + 数据蒸馏
DeepSeek-R1 (最终版本)
  ↓  知识蒸馏 → 小版本
DeepSeek-R1-Distill-Qwen/LLaMA (1.5B~70B)
```

---

## 核心技术一：GRPO（Group Relative Policy Optimization）

**问题**：PPO 需要训练一个与策略模型同等规模的 Critic（价值函数），显存和计算成本翻倍。

**GRPO 的解法**：

```text
对于同一个问题 q，生成 G 个候选答案 {o₁, o₂, ..., oG}
每个答案的优势 = (r_i - mean(r)) / std(r)  ← 组内相对评估
```

- **无需 Critic 模型**：用同一批输出的平均奖励作为基线（Baseline），代替价值函数。
- **稳定性**：组内归一化使梯度更稳定，不会因单个样本的极端奖励而崩溃。
- **KL 正则化**：仍然保留 KL 惩罚，防止策略偏移过大。

---

## 核心技术二：推理能力的自发涌现

### DeepSeek-R1-Zero（纯 RL）

完全不用任何 CoT 示范数据，仅用**可验证奖励（Verifiable Reward）**：

| 奖励类型 | 说明 |
| --- | --- |
| **Accuracy Reward** | 数学题答案对/错（严格匹配或 MATH 格式验证） |
| **Format Reward** | 输出必须包含 `<think>...</think>` 和 `<answer>...</answer>` 标签 |

实验结果：模型自发涌现出**自我反思（Aha Moment）**——重新检查自己的推理步骤并修正错误，无需任何人工引导。

### DeepSeek-R1（多阶段精炼）

```text
阶段1：冷启动 SFT（少量长 CoT 数据，避免 R1-Zero 的格式混乱问题）
阶段2：面向推理的 RL（GRPO + 可验证奖励）
阶段3：拒绝采样 + SFT（筛选高质量 RL 轨迹，补充通用数据）
阶段4：全场景 RL（同时覆盖推理和对话能力）
```

---

## 核心技术三：知识蒸馏

用 DeepSeek-R1（671B MoE）生成的长思维链数据训练小模型：

- **Distill-Qwen-1.5B/7B/14B/32B**
- **Distill-LLaMA-8B/70B**

**核心发现**：小模型通过蒸馏 R1 的推理轨迹，推理能力远超直接对小模型做 RL 的效果。

---

## MoE 架构基座

DeepSeek-R1 的基座 DeepSeek-V3 使用了 MoE 架构：

- **总参数**：671B，**激活参数**：仅 37B（约 5.5% 激活率）
- **专家路由**：Top-K 稀疏激活，配合负载均衡损失（Auxiliary Loss）
- **多头潜在注意力（MLA）**：压缩 KV Cache，提升推理效率

---

## 关键创新与影响

| 贡献 | 说明 |
| --- | --- |
| **GRPO 简化 RL 训练** | 无需 Critic，大幅降低 o1 级推理模型的训练成本 |
| **推理能力自发涌现** | 证明纯 RL + 可验证奖励可产生复杂推理，无需 CoT 标注 |
| **开源推动生态** | 完整开源模型权重和技术报告，推动全球 R1 变体爆发 |
| **蒸馏范式** | 小模型通过蒸馏大模型推理轨迹，性价比远高于直接训练 |

---

## 📚 核心论文

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2025)](https://arxiv.org/abs/2501.12948)
- [DeepSeek-V3 Technical Report (2024)](https://arxiv.org/abs/2412.19437)

---
## 定义与目标

- **定义**：本节主题用于解释该模块的核心概念与实现思路。
- **目标**：帮助读者快速建立问题抽象、方法路径与工程落地方式。
## 关键步骤

1. 明确输入/输出与任务边界。
2. 按模块主流程执行核心算法或系统步骤。
3. 记录指标并做对比分析，形成可复用结论。
## 关键公式（逻辑表达）

`Result = CoreMethod(Input, Config, Constraints)`

符号说明：
- `Input`：任务输入。
- `Config`：训练或推理配置。
- `Constraints`：方法约束（如资源、稳定性或安全边界）。
## 关键步骤代码（纯文档示例）

```python
# 关键流程示意（与具体工程实现解耦）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```
