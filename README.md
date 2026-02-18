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
| 1 | [项目认知](./post_train/rl_basics/mdp/README.md) | 理解 MDP 五元组 (S, A, R, P, γ) 的物理建模含义 |
| 2 | [RL 基础](./post_train/rl_basics/td_learning/README.md) | 掌握 TD 误差如何引导价值函数在动态环境中收敛 |
| 3 | [优势估计](./post_train/rl_basics/advantage/README.md) | 理解 $A(s,a) = Q(s,a) - V(s)$ 对训练方差的缩减作用 |
| 4 | [GAE 算法](./post_train/rl_basics/gae/README.md) | 深度理解 GAE λ 权衡偏差与方差的数学逻辑 |

### 第二阶段：语言模型对齐 (Alignment)

| Day | 重点模块 | 核心原理掌握要求 |
| --- | --- | --- |
| 5 | [SFT 入门](./post_train/alignment/sft/README.md) | 掌握交叉熵损失在文本生成任务中的收敛特性 |
| 6 | [DPO 对齐](./post_train/alignment/dpo/README.md) | 理解如何通过隐式奖励函数避开显式奖励模型的训练 |
| 7 | [PPO 算法](./post_train/alignment/ppo/README.md) | 掌握策略裁剪 (Clipping) 如何保障 RL 对齐的稳定性 |
| 8 | [GRPO 创新](./post_train/alignment/grpo/README.md) | 理解组内相对奖励标准化对推理任务性能的提升 |
| 9 | [RLHF 闭环](./post_train/alignment/rlhf/README.md) | 掌握从 SFT 到 PPO 的全流程数据流与系统架构 |

### 第三阶段：多模态与大规模系统 (Systems)

| Day | 重点模块 | 核心原理掌握要求 |
| --- | --- | --- |
| 10 | [离线 RL](./post_train/offline_rl/README.md) | 掌握 CQL 如何通过 Conservative 项抑制 OOD 动作 |
| 11 | [VLM 入门](./pre_train/vlm/README.md) | 理解 Q-Former 或 MLP Projector 如何实现模态对齐 |
| 12 | [并行策略](./pre_train/llm/megatron/README.md) | 深度理解 TP/PP/DP 在分布式训练中的通信开销 |
| 13 | [复习自测](./learning/quizzes/) | 总结 KV Cache、Flash Attention 等工程优化细节 |
| 14 | **技术总结与导出** | 导出个人技术摘要：`python scripts/technical_brief.py` |

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

### 3. 注意力机制与工程优化

- **架构演进**：MHA $\to$ MQA $\to$ **GQA** (Grouped-Query)。GQA 在 Llama 3 中被广泛采用，实现了精度与推理吞吐量的最佳平衡。
- **Flash Attention**：基于 SRAM 的分块计算与 Tiling 策略，消除了显存读写的 IO 瓶颈。
- **ZeRO 技术**：通过 DeepSpeed 划分模型状态，突破单卡显存对参数规模的限制。

---

## 🎯 深度解析与工程建议 (Core Principles Deep Dive)

- **KL 散度控制**：在对齐训练中，KL 散度过快增长通常预示着模型正在过度拟合奖励函数（Reward Hacking）。建议检查 $\beta$ 系数或样本正则化。
- **分布式瓶颈**：在大规模训练中，PP (Pipeline Parallelism) 虽然节省显存，但会引入 Bubble Time；TP (Tensor Parallelism) 虽效率高但对节点间带宽要求极严。

---

## 📂 项目结构 (Project Structure)

- `pre_train/`：模型架构与预训练（Megatron-LM / VLM / Transformers）
- `post_train/`：对齐技术栈（RL Basics / SFT / DPO / PPO / GRPO）
- `scripts/`：技术摘要生成、自动化回归测试工具。
- `learning/`：原理自测题库、核心知识卡片。

---

## 🧪 系统健康度验证

```bash
python scripts/smoke_test.py  # 验证全模块运行逻辑，结果输出至 output/smoke_reports/
```
