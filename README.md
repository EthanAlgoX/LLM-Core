# LLM-Core: 核心知识审计与复现仓库

本项目是一个系统的 LLM 核心技术栈审计仓库。通过对 LLM、VLM 与后训练（Alignment）关键环节的“最小闭环”复现，记录并巩固大模型底层原理与工程实践方案。

---

## 🛠️ 环境预设与运行

```bash
# 激活工程环境
conda activate finetune

# 运行模块审计 (建议配合 --toy 参数观察闭环逻辑)
python run.py --module sft --toy
python run.py --module ppo --toy
```

---

## 🌐 LLM 核心知识图谱 (Core Knowledge Map)

### 1. 强化学习演进 (RL Foundation)

| 领域 | 核心审计模块 | 原理审计要点 |
| --- | --- | --- |
| 决策建模 | [MDP 模型复现](./modules/01_foundation_rl/mdp/README.md) | MDP 五元组建模与 Bellman 备份 |
| 价值审计 | [TD Learning](./modules/01_foundation_rl/td_learning/README.md) | Q-Learning 与 SARSA 的收敛特性差异 |
| 优势优化 | [GAE 核心实现](./modules/01_foundation_rl/gae/README.md) | 广义优势估计在偏差与方差间的数学权衡 |

### 2. 模型架构与多模态 (Architecture & VLM)

| 领域 | 核心审计模块 | 原理审计要点 |
| --- | --- | --- |
| 基础架构 | [Transformer Core](./modules/02_architecture/llm/README.md) | Attention 计算与架构实现规范 |
| 跨模态 | [多模态 VLM](./modules/02_architecture/vlm/README.md) | 视觉特征空间向语言特征空间的对齐投影 |

### 3. 模型微调与对齐 (SFT & Alignment)

| 领域 | 核心审计模块 | 原理审计要点 |
| --- | --- | --- |
| 指令微调 | [监督微调 (SFT)](./modules/03_alignment/sft/README.md) | 遵循数据的 Loss Mask 策略与预测质量 |
| 强化对齐 | [PPO 深度审计](./modules/03_alignment/ppo/README.md) | 重要性采样约束 (Ratio Clip) 与 KL 惩罚 |
| 离线对齐 | [DPO 算法映射](./modules/03_alignment/dpo/README.md) | 隐式奖励函数在对比学习逻辑下的有效性 |
| 推理对齐 | [GRPO 推理优化](./modules/03_alignment/grpo/README.md) | 组内相对标准化对逻辑链（CoT）生成的提升 |

### 4. 推理与生成优化 (Inference & Generation)

| 领域 | 核心审计模块 | 原理审计要点 |
| --- | --- | --- |
| 优化算子 | [Flash Attention](./modules/02_architecture/generation/README.md) | IO 感知的注意力计算优化原理 |
| 解码搜索 | [Decoding 策略](./modules/02_architecture/generation/README.md) | Greedy/Sampling/Beam Search 的审计与权衡 |
| 缓存机制 | [KV Cache 优化](./modules/02_architecture/generation/README.md) | 推理时显存占用的线性增长控制 |

### 5. 工程与系统性能 (Engineering & Scaling)

| 领域 | 核心审计模块 | 原理审计要点 |
| --- | --- | --- |
| 分布式 | [并行策略 (Megatron)](./modules/05_engineering/megatron/README.md) | TP/PP/DP 模式下的通信开销与算力利用率 |
| 显存管理 | [ZeRO/DeepSpeed](./modules/05_engineering/deepspeed/README.md) | 状态切分与显存冗余消除技术 |
| 算子加速 | [CUDA/Triton 基础](./modules/05_engineering/cuda/README.md) | 高效算子开发的基本原理与内存优化 |

### 6. 智能体深度审计 (Intelligent Agents)

| 领域 | 核心审计模块 | 原理审计要点 |
| --- | --- | --- |
| 自动化决策 | [智能体 (Agent)](./modules/06_agent/README.md) | ReAct 架构中 Thought-Action-Observation 的状态机流转 |
| 架构参考 | [NanoBot 深度分析](./modules/06_agent/README.md) | 动态上下文组装、XML 技能索引与双层记忆持久化 |

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
  - `06_agent/`: 智能体推理专门模块 (Planning, Tools, Memory)
- `tools/`: 技术摘要生成、自动化回归测试工具
- `data/`: 模拟训练数据
- `output/`: 训练产物、日志与测试报告

---

## 🧪 系统健康度验证

```bash
python tools/smoke_test.py  # 验证全模块运行逻辑，结果输出至 output/smoke_reports/
```
