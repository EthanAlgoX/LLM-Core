# LLM-Core: 核心知识审计与复现仓库

本项目是一个系统的 LLM 核心技术栈审计仓库。通过对 LLM、VLM 与后训练（Alignment）关键环节的"最小闭环"复现，记录并巩固大模型底层原理与工程实践方案。

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

### 1. 理论根基：算法与优化 (Theory & Optimization)

| 领域 | 核心内容 | 原理审计要点 |
| --- | --- | --- |
| 决策建模 | [MDP 模型复现](./modules/01_foundation_rl/mdp/mdp.md) | MDP 五元组建模与 Bellman 备份方程 |
| 价值学习 | [TD Learning](./modules/01_foundation_rl/td_learning/td_learning.md) | Q-Learning (Off-policy) 与 SARSA (On-policy) 差异 |
| 偏好对齐 | [DPO / PPO](./modules/03_alignment/ppo/ppo.md) | KL 散度约束、**知识蒸馏**与模型能力迁移控制 |
| 性能调优 | [Performance Tuning](./modules/05_engineering/inference/inference.md) | **CPU/GPU 性能调优**、算子融合与 IO 瓶颈分析 |

### 2. 架构核心：变压器、多模态与 MoE (Architecture & Scaling)

| 领域 | 核心内容 | 原理审计要点 |
| --- | --- | --- |
| 核心架构 | [Transformer Core](./modules/02_architecture/llm/llm.md) | MHA、Pre-LN 稳定性与 **文本/多模态 Embedding** 对齐 |
| 扩展架构 | [MoE (Mixture of Experts)](./modules/02_architecture/llm/llm.md) | **Expert Parallelism**、**Load Balancing** 与稀疏计算优化 |
| 生成推理 | [Decoding Strategy](./modules/02_architecture/generation/generation.md) | Flash Attention、KV Cache 与 **定点量化 (INT8/FP8)** |

### 3. 能力塑形：微调、对齐与仿真 (Post-Training & Alignment)

| 领域 | 核心内容 | 原理审计要点 |
| --- | --- | --- |
| 参数高效微调 | [PEFT 审计](./modules/03_alignment/peft/peft.md) | **LoRA**、**Prefix Tuning** 与模型融合 (**Model Merging**) |
| 偏好对齐 | [RLHF / DPO / PPO](./modules/03_alignment/ppo/ppo.md) | 在线/离线对齐算法深度实践与 **Continual Pre-training** |
| 智能体强化学习 | [Agentic-RL](./modules/03_alignment/rlhf/rlhf.md) | **Agentic-RL** 训练范式、**MARL (MAPPO)** 与 **User Simulator** |
| 数据与评估 | [Data & Evaluation](./modules/03_alignment/data_engineering.md) | **数据处理 (Cleaning)**、**对抗性测试** 与 **LLM-as-a-Judge** |

### 4. 系统性能：并发、并行与 PD 分离 (Engineering & Scaling)

| 领域 | 核心内容 | 原理审计要点 |
| --- | --- | --- |
| 并行策略 | [Distributed Training](./modules/05_engineering/megatron/megatron.md) | TP/PP/EP (专家并行) 通信开销与 **PD 分离架构** |
| 推理框架 | [Inference Frameworks](./modules/05_engineering/inference/inference.md) | **vLLM (PagedAttention)**、**sglang** 与算子融合调优 |

### 5. 应用闭环：自主智能体与多机协作 (Agents & Mesh)

| 领域 | 核心内容 | 原理审计要点 |
| --- | --- | --- |
| 编排范式 | [Agent Orchestration](./modules/06_agent/06_agent.md) | **ReAct**、**Plan-and-Execute** 与 **Self-Ask** 模式 |
| 系统架构 | [Mesh & State Machine](./modules/06_agent/06_agent.md) | **Async Orchestration**、**复杂状态机** 与 **Conditional Routing** |
| 多智能体协作 | [Multi-Agent Systems](./modules/06_agent/06_agent.md) | **Decentralized Orchestration**、通信协议与 **Human-in-the-Loop** |

---

## 🧠 核心技术参考 (Technical Reference)

### 1. 显存计算与容量估算 (Memory & Compute)

- **静态权重**：`fp16` 占 2 Bytes/Param。
- **KV Cache**：显存占用 = `2 × layers × heads × head_dim × precision_bytes`。
- **量化增益**：通过 **定点量化** (INT4/INT8)，显存占用可降低 50%-75%。

### 2. 系统演进：从 Dense 到 MoE

- **MoE 优势**：通过稀疏激活，在不显著增加计算量的前提下极大扩展模型参数量。
- **并行瓶颈**：专家并行 (EP) 会引入额外的 All-to-All 通信开销，需配合负载均衡。

### 3. Agent 架构演进

- **ReAct 范式**：协同推理（Reason）与行动（Act），动态调整计划。
- **Plan and Execute**：先计划再执行，适合长链条任务。
- **Multi-Agent Mesh**：去中心化编排，支持分布式决策与角色分担。

---

## 📂 项目结构 (Project Structure)

- `modules/`: 核心知识组件
  - `01_foundation_rl/`: 理论根基 (MDP, TD, GAE)
  - `02_architecture/`: 架构核心 (LLM, VLM, MoE, Quantization)
  - `03_alignment/`: 对齐技术 (SFT, PEFT, Agentic-RL, Data Engineering)
  - `05_engineering/`: 工程与性能 (DeepSpeed, Megatron, vLLM, sglang, EP)
  - `06_agent/`: 智能体 (RAG, Mesh, Multi-Agent, State Machine)
- `tools/`: 自动化回归测试工具
- `output/`: 训练产物、日志与测试报告

---

## 🧪 系统健康度验证

```bash
python tools/smoke_test.py  # 验证全模块运行逻辑，结果输出至 output/smoke_reports/
```
