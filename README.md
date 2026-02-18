# LLM-Core 核心知识复现与学习

本项目致力于通过“最小闭环”复现，帮助开发者在 14 天内深度掌握 LLM、VLM 与后训练（Alignment）的核心原理与工程实现。

---

## 🛠️ 环境与一键运行

```bash
# 激活环境
conda activate finetune

# 运行模块 (推荐配合 --toy 参数快速体验)
python run.py --module sft --toy
python run.py --module ppo --toy
```

---

## 📅 14 天深度学习路线图 (Roadmap)

### 第一阶段：强化学习基础 (Foundation)

| Day | 重点模块 | 核心原理掌握要求 |
| --- | --- | --- |
| 1 | [项目认知与 MDP](./modules/01_foundation_rl/mdp/README.md) | 理解 MDP 五元组 (S, A, R, P, γ) 的物理建模含义 |
| 2 | [TD Learning](./modules/01_foundation_rl/td_learning/README.md) | 掌握 Q-Learning 与 SARSA 的更新差异 |
| 3 | [优势估计](./modules/01_foundation_rl/gae/README.md) | 深入 GAE (Generalized Advantage Estimation) 的偏差-方差权衡 |

### 第二阶段：大模型微调与对齐 (Alignment)

| Day | 重点模块 | 核心原理掌握要求 |
| --- | --- | --- |
| 4 | [监督微调 (SFT)](./modules/03_alignment/sft/README.md) | 掌握指令遵循数据的构建与 Loss Mask 技巧 |
| 5 | [策略梯度 (PG)](./modules/03_alignment/policy_gradient/README.md) | 理解 REINFORCE 算法及其在高方差下的局限性 |
| 6 | [Actor-Critic](./modules/03_alignment/actor_critic/README.md) | 掌握价值网络 (Critic) 如何辅助策略网络 (Actor) 稳定收敛 |
| 7 | [PPO 深度解析](./modules/03_alignment/ppo/README.md) | 理解重要性采样 (Importance Sampling) 与 Clip 约束 |
| 8 | [DPO 离线对齐](./modules/03_alignment/dpo/README.md) | 掌握对比学习思想在偏好优化中的应用 |
| 9 | [GRPO 推理优化](./modules/03_alignment/grpo/README.md) | 理解组相对策略优化（DeepSeek 系列核心技术） |

### 第三阶段：多模态与进阶领域 (Advanced)

| Day | 重点模块 | 核心原理掌握要求 |
| --- | --- | --- |
| 10 | [离线 RL](./modules/04_advanced_topics/offline_rl/README.md) | 掌握 CQL 如何通过 Conservative 项抑制 OOD 动作 |
| 11 | [多模态 VLM](./modules/02_architecture/vlm/README.md) | 理解 Q-Former 或 MLP Projector 如何实现模态对齐 |
| 12 | [并行策略](./modules/05_engineering/megatron/README.md) | 深度理解 TP/PP/DP 在分布式训练中的通信开销 |
| 13 | [智能体 (Agent)](./agents/README.md) | 掌握 ReAct 循环中 Thought/Action/Observation 的状态流转 |
| 14 | **总结与自测** | [复习自测](./modules/06_quizzes_and_cards/)，原理摘要：`tools/technical_brief.py` |

---

## 🧠 核心技术参考 (Technical Reference)

### 1. 显存计算与容量估算 (Memory & Compute)

- **静态权重**：`fp16` 占 2 Bytes/Param。例如 7B 模型加载需 ~14GB。
- **KV Cache**：$2 \times \text{layers} \times \text{heads} \times \text{dim} \times \text{precision}$。
- **PEFT (LoRA)**：$\Delta W = A \times B$。通过低秩分解显著降低训练时的显存梯度存储需求。

### 2. 核心训练算法对比

| 特性 | SFT | PPO | DPO | GRPO |
| :--- | :--- | :--- | :--- | :--- |
| **显存压力** | 低 | **极高** (涉及4个独立模型) | 中 | 中 (省去 Critic 网络) |
| **收敛特性** | 极稳 | 较敏感 (取决于优势估计精度) | 稳定 | 稳定 (适合数学推理) |
| **优化目标** | 字对字模仿 | 奖励信号最大化 | 偏好映射最大化 | 组内相对反馈优化 |

### 3. Agent 与系统架构

- **核心逻辑**：Agent = LLM + Planning + Memory + Tool Use
- **ReAct 范式**：协同推理（Reason）与行动（Act），使模型具备动态调整计划的能力。
- **Flash Attention**：基于 SRAM 的分块计算，消除显存读写的 IO 瓶颈。

---

## 🎯 深度解析与工程建议 (Core Principles Deep Dive)

- **KL 散度控制**：在对齐训练中，KL 散度过快增长通常预示着模型正在过度拟合奖励函数。
- **分布式瓶颈**：在大规模训练中，PP (Pipeline Parallelism) 虽然节省显存，但会引入 Bubble Time；TP (Tensor Parallelism) 虽效率高但对节点间带宽要求极严。
- **智能体幻觉**：Agent 在复杂任务中易陷入无限循环或调用不存在的工具，建议增加自我反思（Self-Reflection）或强约束 Schema 解析。

---

## 📂 项目结构 (Project Structure)

- `modules/`: 核心学习组件
  - `01_foundation_rl/`: 强化学习基础 (MDP, TD, GAE)
  - `02_architecture/`: 模型架构 (LLM, VLM, Generation)
  - `03_alignment/`: 对齐技术栈 (SFT, DPO, PPO, GRPO)
  - `04_advanced_topics/`: 进阶话题 (Offline RL)
  - `05_engineering/`: 工程与系统 (DeepSpeed, Megatron, CUDA)
  - `06_quizzes_and_cards/`: 原理自测题库与核心知识卡片
- `agents/`: 智能体推理专门模块 (Planning, Tools, Memory)
- `tools/`: 技术摘要生成、自动化回归测试工具
- `data/`: 模拟训练数据
- `output/`: 训练产物、日志与测试报告

---

## 🧪 系统健康度验证

```bash
python tools/smoke_test.py  # 验证全模块运行逻辑，结果输出至 output/smoke_reports/
```
