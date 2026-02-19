# LLM-Core: 核心知识解析与复现仓库

本项目是一个系统的 LLM 核心技术栈解析仓库。通过对 LLM、VLM 与后训练（Alignment）关键环节的"最小闭环"复现，记录并巩固大模型底层原理与工程实践方案。

---

## 🛠️ 环境预设与运行

```bash
# 激活工程环境
conda activate finetune

# 运行模块解析 (建议配合 --toy 参数观察闭环逻辑)
python run.py --module sft --toy
python run.py --module ppo --toy
```

---

## 🌐 LLM 核心知识图谱 (Core Knowledge Map)

### 1. 理论根基：算法与优化 (Theory & Optimization)

| 领域 | 核心内容 | 原理技术要点 |
| --- | --- | --- |
| 理论概览 | [RL foundation](./modules/01_foundation_rl/01_foundation_rl.md) | 强化学习基础概念、算法分类与学习路径指南 |
| 决策建模 | [MDP 模型复现](./modules/01_foundation_rl/mdp/mdp.md) | MDP 五元组建模与 Bellman 备份方程 |
| 价值学习 | [TD Learning](./modules/01_foundation_rl/td_learning/td_learning.md) | Q-Learning (Off-policy) 与 SARSA (On-policy) 差异 |
| 优势估计 | [GAE](./modules/01_foundation_rl/gae/gae.md) & [Advantage](./modules/01_foundation_rl/advantage/advantage.md) | **GAE**、**多步回报** 与训练稳定性方差权衡 |

### 2. 架构核心：变压器、多模态与 MoE (Architecture & Scaling)

| 领域 | 核心内容 | 原理技术要点 | 一句话理解 |
| --- | --- | --- | --- |
| 基础架构 | [Transformer Core](./modules/02_architecture/llm/llm.md) | **MHSA**、**Normalization (RMSNorm)** 与 **RoPE 旋转位置编码** | LLM 的"身体"——每个词怎么被感知并编码成向量 |
| 注意力内核 | [Attention Mechanisms](./modules/02_architecture/llm/attention.md) | **MHA / GQA / MQA** 变体与 **Flash Attention** 工程实现 | 模型"读文章"时同时关注多个角度，Flash Attention 让这更快更省显存 |
| 扩展架构 | [MoE (Mixture of Experts)](./modules/02_architecture/llm/llm.md) | **Expert Parallelism**、**Load Balancing** 与稀疏计算优化 | 超大模型拆成多个"专家"，每次只激活少数几个，省算力 |
| 模态融合 | [VLM Mapping / Hub](./modules/02_architecture/vlm/vlm.md) | **MLP Projector**、**Cross-Attention** 与多模态对齐策略 | 把图像"翻译"成语言模型能理解的格式，让 AI 能看图说话 |
| 多媒体生成 | [Diffusion](./modules/02_architecture/generation/diffusion/diffusion.md) / [DiT](./modules/02_architecture/generation/dit/dit.md) | **Diffusion Transformer (DiT)**、**Stable Diffusion** 与生成控制 | AI 画图原理：从噪声图一步步去噪，变成目标图片 |
| 生成推理 | [Decoding Strategy](./modules/02_architecture/generation/generation.md) | **KV Cache (PagedAttention)**、**解码策略** 与 **投机采样** | 让 AI 吐字又快又省内存——如何高效地一字一字生成文本 |

### 3. 能力塑形：微调、对齐与仿真 (Post-Training & Alignment)

| 领域 | 核心内容 | 原理技术要点 | 一句话理解 |
| --- | --- | --- | --- |
| 对齐总览 | [Alignment Overview](./modules/03_alignment/03_alignment.md) | 后训练对齐全链路解析：从 SFT 到 RLHF 的演进路径 | 从"写作业"到"真正听懂人话"，AI 是怎么一步步被调教的 |
| 监督学习 | [SFT 解析](./modules/03_alignment/sft/sft.md) | **Supervised Fine-Tuning**、数据质量初筛与指令遵循 | 拿人类写的高质量问答对，手把手教模型"怎么说话" |
| 参数高效微调 | [PEFT 解析](./modules/03_alignment/peft/peft.md) | **LoRA**、**Prefix Tuning** 与模型融合 (**Model Merging**) | 只改模型一小部分权重，就能让它学会新技能，省时省显存 |
| 偏好对齐 | [PPO](./modules/03_alignment/ppo/ppo.md) / [DPO](./modules/03_alignment/dpo/dpo.md) | **在线/离线对齐算法**、**奖励模型 (RM)** 与 **隐含偏好优化 (DPO)** | 给 AI 两个答案让它选好的，通过"偏好打分"驯化它说人话 |
| 基础算法 | [PG](./modules/03_alignment/policy_gradient/policy_gradient.md) / [AC](./modules/03_alignment/actor_critic/actor_critic.md) | Policy Gradient (REINFORCE) 与 Actor-Critic 架构基础 | RL 的基础：好结果加分、坏结果扣分，用梯度驱动策略进化 |
| 推理对齐 | [DeepSeek GRPO](./modules/03_alignment/grpo/grpo.md) | **GRPO 对齐范式**、奖励模型建模与复杂逻辑链验证 | DeepSeek 的秘诀：同时生成多个答案，优质的胜出，让模型学会"推理" |
| 智能体强化学习 | [Agentic-RL](./modules/03_alignment/rlhf/rlhf.md) | **Agentic-RL 训练范式**、**基于模拟器的演练** 与 **多智能体博弈 (MARL)** | 在沙盒里练习，让 Agent 从实战模拟中学会完成复杂的真实任务 |
| 数据与评估 | [Data & Evaluation](./modules/03_alignment/data_engineering.md) | **数据处理 (Cleaning)**、**对抗性测试** 与 **LLM-as-a-Judge** | 训练数据的质量决定模型上限，用 LLM 来给 LLM 打分做筛选 |

### 4. 系统性能：并发、并行与 PD 分离 (Engineering & Scaling)

| 领域 | 核心内容 | 原理技术要点 | 一句话理解 |
| --- | --- | --- | --- |
| 工程总览 | [Engineering Overview](./modules/05_engineering/05_engineering.md) | 模型并行、显存管理与大规模训练系统的技术演进入口 | 把一个大到塞不进单张卡的模型，切开分发到成百上千张卡协同训练 |
| 并行策略 | [Distributed Training](./modules/05_engineering/megatron/megatron.md) | **3D Parallelism (TP/PP/DP)**、**专家并行 (EP)** 与通信优化 | 模型切横刀、纵刀、流水线——三把刀解决超大模型的分布式训练难题 |
| 推理框架 | [Inference Frameworks](./modules/05_engineering/inference/inference.md) | **Prefill-Decode 分离**、**PagedAttention** 与算子融合 | 让用户提问和 AI 吐字分开跑，显存更省、吞吐更高 |
| 算子与加速 | [CUDA](./modules/05_engineering/cuda/cuda.md) / [Precision](./modules/05_engineering/mixed_precision/mixed_precision.md) | **CUDA Kernel 优化**、**混合精度 (FP16/BF16)** 与量化加速原理 | 用低精度浮点数代替 FP32，显存省一半，速度翻倍，效果几乎不变 |
| 工程框架 | [DeepSpeed](./modules/05_engineering/deepspeed/deepspeed.md) | **DeepSpeed ZeRO** 系列显存优化与训练流水线 | ZeRO 把优化器状态等分散到每张卡，人人各拿一份不重复，省掉冗余显存 |

### 5. 应用闭环：自主智能体与多机协作 (Agents & Mesh)

| 领域 | 核心内容 | 原理技术要点 | 一句话理解 |
| --- | --- | --- | --- |
| 智能体总览 | [Agentic Overview](./modules/06_agent/06_agent.md) | 核心逻辑：**LLM + Planning + Memory + Toolkit** 深度解析 | 让 AI 从"回答问题"升级为"主动做事"——思考、记忆、用工具、迭代执行 |
| 记忆与检索 | [Memory & RAG](./modules/06_agent/memory_rag/memory_rag.md) | **RAG**、**Query 理解**、**向量检索** 与 **Rerank 模型** | AI 自带可检索"知识库"，让它能回答私有或超出训练范围的问题 |
| 编排范式 | [Agent Orchestration](./modules/06_agent/orchestration/orchestration.md) | **ReAct**、**Plan-and-Execute** 与 **Function Calling** 工具增强 | Agent 的"大脑逻辑"：先思考再行动，遇到新情况随时调整计划 |
| 系统架构 | [Mesh & State Machine](./modules/06_agent/06_agent.md) | **NanoBot 设计模式**、**多层记忆体系** 与 **Conditional Routing** | 用状态机管理 Agent 行为流，保证复杂任务可靠推进不乱套 |
| 多智能体协作 | [Multi-Agent Systems](./modules/06_agent/multi_agent/multi_agent.md) | **Decentralized Orchestration**、通信协议与 **Human-in-the-Loop** | 多个 AI 各自分工合作，像一个智能团队完成单 Agent 搞不定的任务 |
| 本地Agent框架 | [OpenClaw 架构](./modules/06_agent/openclaw/openclaw.md) | **Gateway + Runtime**、**文件记忆系统**、**Heartbeat 事件驱动** 与 **混合检索** | 可本地运行的完整 Agent 框架，用文件系统做记忆，用消息总线解耦通信 |

### 6. 进阶课题：离线强化学习 (Advanced Topics: Offline RL)

| 领域 | 核心内容 | 原理技术要点 | 一句话理解 |
| --- | --- | --- | --- |
| 离线对齐总览 | [Offline RL Overview](./modules/04_advanced_topics/offline_rl/offline_rl.md) | 在无环境交互前提下，利用离线轨迹数据进行策略优化的核心范式 | 不和真实环境交互，只用"历史经验数据"学习，像人类读案例复盘一样 |
| 算法复现 | [Offline RL](./modules/04_advanced_topics/offline_rl/offline_rl.md) | **Offline RL 系统详述** 与数据分布偏移（Distribution Shift）对抗策略 | 离线数据和真实分布有差距，需专门设计算法防止策略学偏 |
| 代表算法 | [BCQ](./modules/04_advanced_topics/offline_rl/bcq/bcq.md) / [CQL](./modules/04_advanced_topics/offline_rl/cql/cql.md) | **外推误差 (Extrapolation Error) 抑制** 与 **下界 Q 学习 (Lower Bound Q-learning)** | BCQ 约束动作别乱飞；CQL 让 Q 值保守低估——两者都防止 AI 对没见过的情况过度自信 |

---

## 🧠 核心技术参考 (Technical Reference)

### 1. 显存计算与容量估算 (Memory & Compute)

- **静态权重**：`fp16` 占 2 Bytes/Param。
- **KV Cache**：显存占用 = `2 × layers × heads × head_dim × precision_bytes`。
- **量化增益**：通过 **定点量化** (INT4/INT8)，显存占用可降低 50%-75%。

### 2. 注意力机制变体

| 机制 | 特点 | 显存优化 |
| --- | --- | --- |
| MHA | Multi-Head Attention | 标准 |
| MQA | Multi-Query Attention | KV 头=1 |
| GQA | Grouped-Query Attention | 折中方案 |

### 3. 对齐算法演进

| 阶段 | 算法 | 核心思想 |
| --- | --- | --- |
| SFT | Supervised Fine-Tuning | 模仿学习 |
| RLHF | PPO + Reward Model | 人类反馈强化学习 |
| DPO | Direct Preference Optimization | 离线对比优化 |
| GRPO | Group Relative Policy Optimization | 组内相对优势 |

### 4. Agent 架构演进

- **ReAct 范式**：协同推理（Reason）与行动（Act），动态调整计划。
- **Plan and Execute**：先计划再执行，适合长链条任务。
- **Multi-Agent Mesh**：去中心化编排，支持分布式决策与角色分担。

### 5. 分布式训练策略

- **数据并行 (DP)**：副本间切分数据，All-Reduce 同步梯度
- **张量并行 (TP)**：切分层内权重，All-Gather 激活
- **流水线并行 (PP)**：按层切分 Stage，Bubble 时间
- **专家并行 (EP)**：MoE 特有，All-to-All 路由专家

---

## 📂 项目结构 (Project Structure)

- `modules/`: 核心知识组件
  - `01_foundation_rl/`: 理论根基 (MDP, TD, GAE)
  - `02_architecture/`: 架构核心 (LLM, VLM, MoE, Quantization, Diffusion, DiT)
  - `03_alignment/`: 对齐技术 (SFT, PEFT, PPO, DPO, GRPO, Agentic-RL)
  - `04_advanced_topics/`: 进阶课题 (Offline RL: BCQ, CQL)
  - `05_engineering/`: 工程与性能 (DeepSpeed, Megatron, vLLM, sglang, CUDA, 混合精度)
  - `06_agent/`: 智能体 (Memory, RAG, Orchestration, Multi-Agent, OpenClaw)
- `tools/`: 自动化回归测试工具
- `output/`: 训练产物、日志与测试报告

---

## 🏗️ 核心模型索引 (Key Model Index)

| 模型分类 | 代表模型 | 核心解析文档 |
| :--- | :--- | :--- |
| **基础语言模型 (LLM)** | LLaMA-3 / Transformer | [Transformer Core](./modules/02_architecture/llm/llm.md) |
| **注意力机制** | Flash Attention | [Attention Mechanisms](./modules/02_architecture/llm/attention.md) |
| **多模态 VLM** | **LLaVA** | [LLaVA 详述](./modules/02_architecture/vlm/llava/llava.md) |
| **多模态 VLM** | **Flamingo** | [Flamingo 详述](./modules/02_architecture/vlm/flamingo/flamingo.md) |
| **多模态 VLM** | **BLIP-2** | [BLIP-2 详述](./modules/02_architecture/vlm/blip2/blip2.md) |
| **生成模型** | Diffusion / DiT | [Diffusion](./modules/02_architecture/generation/diffusion/diffusion.md) / [DiT](./modules/02_architecture/generation/dit/dit.md) |
| **推理增强模型** | DeepSeek (GRPO) | [GRPO 对齐范式](./modules/03_alignment/grpo/grpo.md) |
| **分布式框架** | Megatron-LM | [Megatron 并行策略](./modules/05_engineering/megatron/megatron.md) |
| **推理框架** | vLLM / sglang | [推理框架](./modules/05_engineering/inference/inference.md) |
| **本地Agent** | OpenClaw | [OpenClaw 架构](./modules/06_agent/openclaw/openclaw.md) |

---

## 🧪 系统健康度验证

```bash
python tools/smoke_test.py  # 验证全模块运行逻辑，结果输出至 output/smoke_reports/
```
