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

### 2. 架构核心：变压器与多模态 (Architecture & VLM)

| 领域 | 核心内容 | 原理审计要点 |
| --- | --- | --- |
| 核心架构 | [Transformer Core](./modules/02_architecture/llm/llm.md) | MHA 计算、Pre-LN 稳定性与 **文本/多模态 Embedding** 对齐 |
| 生成推理 | [Decoding Strategy](./modules/02_architecture/generation/generation.md) | Flash Attention、KV Cache 与 **定点量化 (INT8/FP8)** 推理 |
| 模态融合 | [VLM Mapping](./modules/02_architecture/vlm/vlm.md) | 线性投影与交叉注意力层对齐视觉-语言空间 |

### 3. 能力塑形：微调与数据 (Post-Training & Data)

| 领域 | 核心内容 | 原理审计要点 |
| --- | --- | --- |
| 参数高效微调 | [PEFT 审计](./modules/03_alignment/peft/peft.md) | **LoRA**、**Prefix Tuning** 与 AdaLoRA 的低秩分解对比 |
| 数据治理 | [Data Engineering](./modules/03_alignment/data_engineering.md) | **数据处理 (Deduplication/Cleaning)** 与多样性采样策略 |
| 评估体系 | [Model Evaluation](./modules/03_alignment/data_engineering.md) | **模型评估 (Benchmark/Human-eval)** 与对齐稳定性监控 |

### 4. 系统性能：并行与推理框架 (Engineering & Scaling)

| 领域 | 核心内容 | 原理审计要点 |
| --- | --- | --- |
| 推理加速 | [Inference Frameworks](./modules/05_engineering/inference/inference.md) | **vLLM (PagedAttention)**、**sglang (Runtime)** 与 TensorRT |
| 并行策略 | [Distributed Training](./modules/05_engineering/megatron/megatron.md) | TP/PP/DP 通信开销与 ZeRO-3 显存消除 |

### 5. 应用闭环：自主智能体系统 (Intelligent Agents)

| 领域 | 核心内容 | 原理审计要点 |
| --- | --- | --- |
| 信息检索 | [Memory & RAG](./modules/06_agent/06_agent.md) | **RAG**、**Query 理解**、**向量检索** 与 **Rerank 模型** |
| 推理范式 | [Agent Reasoning](./modules/06_agent/06_agent.md) | **ReAct**、**Plan and Execute** 与 Reflection 自反思 |
| 生态集成 | [Frameworks & Tools](./modules/06_agent/06_agent.md) | **Tool-use (Function Calling)**、**LangChain** 与 **LangGraph** |

---

## 🧠 核心技术参考 (Technical Reference)

### 1. 显存计算与容量估算 (Memory & Compute)

- **静态权重**：`fp16` 占 2 Bytes/Param。
- **KV Cache**：显存占用 = `2 × layers × heads × head_dim × precision_bytes`。
- **量化增益**：通过 **定点量化** (INT4/INT8)，显存占用可降低 50%-75%。

### 2. Agent 架构演进

- **ReAct 范式**：协同推理（Reason）与行动（Act），动态调整计划。
- **Plan and Execute**：先生成完整计划再执行，适合复杂逻辑解耦。

---

## 📂 项目结构 (Project Structure)

- `modules/`: 核心知识组件
  - `01_foundation_rl/`: 理论根基 (MDP, TD, GAE)
  - `02_architecture/`: 架构核心 (LLM, VLM, Embedding, Quantization)
  - `03_alignment/`: 对齐技术 (SFT, PEFT/LoRA, Distillation, Data Process)
  - `05_engineering/`: 工程与性能 (DeepSpeed, vLLM, sglang, CPU/GPU Tuning)
  - `06_agent/`: 智能体 (RAG, Rerank, Plan&Execute, Frameworks)
- `tools/`: 自动化回归测试工具
- `output/`: 训练产物、日志与测试报告

---

## 🧪 系统健康度验证

```bash
python tools/smoke_test.py  # 验证全模块运行逻辑，结果输出至 output/smoke_reports/
```
