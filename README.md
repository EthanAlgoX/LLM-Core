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

### 1. 理论根基：序贯决策与强化学习 (Theory & RL)

| 领域 | 核心审计模块 | 原理审计要点 |
| --- | --- | --- |
| 决策建模 | [MDP 模型复现](./modules/01_foundation_rl/mdp/README.md) | MDP 五元组 (S,A,R,P,γ) 建模与 Bellman 备份方程 |
| 价值学习 | [TD Learning](./modules/01_foundation_rl/td_learning/README.md) | Q-Learning (off-policy) 与 SARSA (on-policy) 的收敛特性差异 |
| 策略梯度 | [Policy Gradient](./modules/03_alignment/policy_gradient/README.md) | REINFORCE 算法及高方差问题的基线 (Baseline) 控制 |
| 价值协同 | [Actor-Critic](./modules/03_alignment/actor_critic/README.md) | Critic 网络对 Actor 策略更新的基准平滑作用 |
| 优势估计 | [GAE 核心实现](./modules/01_foundation_rl/gae/README.md) | 广义优势估计 (λ 调节) 在偏差与方差间的数学权衡 |

### 2. 架构核心：变压器与生成机制 (Architecture & Generation)

| 领域 | 核心审计模块 | 原理审计要点 |
| --- | --- | --- |
| 核心架构 | [Transformer Core](./modules/02_architecture/llm/README.md) | Multi-Head Attention 计算、位置编码与 Pre-LN 稳定性 |
| 生成推理 | [Generation & Decoding](./modules/02_architecture/generation/README.md) | Flash Attention IO 优化、KV Cache 管理与解码策略对比 |

### 3. 架构扩展：多模态对齐与映射 (Multi-modal VLM)

| 领域 | 核心审计模块 | 原理审计要点 |
| --- | --- | --- |
| 视觉编码 | [ViT/CLIP 基础](./modules/02_architecture/vlm/README.md) | 图像分块 (Patchify) 与全局语义特征提取 |
| 模态对齐 | [Q-Former / MLP](./modules/02_architecture/vlm/README.md) | 线性投影与交叉注意力层对齐视觉-语言空间 |
| 深度融合 | [LLaVA / Flamingo](./modules/02_architecture/vlm/README.md) | 特征对齐预训练 (Stage 1) 与视觉指令微调 (Stage 2) |

### 4. 能力塑造：指令遵循与偏好对齐 (Post-Training & Alignment)

| 领域 | 核心审计模块 | 原理审计要点 |
| --- | --- | --- |
| 指令微调 | [监督微调 (SFT)](./modules/03_alignment/sft/README.md) | 指令遵循数据的 Loss Mask 策略与 next-token 预测质量 |
| 在线对齐 | [PPO 深度审计](./modules/03_alignment/ppo/README.md) | 重要性采样约束 (Ratio Clip)、KL 惩罚与 Critic 稳定性 |
| 离线对齐 | [DPO 算法映射](./modules/03_alignment/dpo/README.md) | 隐式奖励函数推导：从 RLHF 到对比学习的等价变换 |
| 推理对齐 | [GRPO 推理优化](./modules/03_alignment/grpo/README.md) | 组内相对标准化 (Group Relative) 对 CoT 逻辑链生成的提升 |
| 保守策略 | [离线 RL (CQL)](./modules/04_advanced_topics/offline_rl/README.md) | Conservative Q-Learning 对 OOD 动作价值的抑制机制 |

### 5. 系统性能：大规模并行与推理加速 (Engineering & Scaling)

| 领域 | 核心审计模块 | 原理审计要点 |
| --- | --- | --- |
| 分布式训练 | [并行策略 (Megatron)](./modules/05_engineering/megatron/README.md) | TP/PP/DP 并行模式下的通信开销与 Bubble Time 分析 |
| 显存优化 | [ZeRO/DeepSpeed](./modules/05_engineering/deepspeed/README.md) | ZeRO-1/2/3 状态切分与显存冗余消除技术 |
| 混合精度 | [Mixed Precision](./modules/05_engineering/mixed_precision/README.md) | FP16/BF16 训练的数值稳定性与 Loss Scaling 策略 |
| 推理加速 | [Inference 优化](./modules/05_engineering/inference/README.md) | 量化 (INT8/INT4)、投机采样与连续批处理 |
| 算子开发 | [CUDA/Triton 基础](./modules/05_engineering/cuda/README.md) | GPU 内存层次、Warp 调度与高效算子编写规范 |

### 6. 应用闭环：自主智能体系统 (Intelligent Agents)

| 领域 | 核心审计模块 | 原理审计要点 |
| --- | --- | --- |
| 推理循环 | [ReAct Agent](./modules/06_agent/README.md) | Thought-Action-Observation 状态机与 Reflection 注入 |
| 记忆系统 | [Memory & Context](./modules/06_agent/README.md) | 双层记忆 (MEMORY.md + HISTORY.md) 与 grep 主动回溯 |
| 工具集成 | [Tool Use & MCP](./modules/06_agent/README.md) | Function Calling 规范、安全沙箱与 MCP 协议集成 |
| 多智能体 | [Subagent 委托](./modules/06_agent/README.md) | 主从 Agent 任务委托、权限约束与总线回传机制 |

---

## 🧠 核心技术参考 (Technical Reference)

### 1. 显存计算与容量估算 (Memory & Compute)

- **静态权重**：`fp16` 占 2 Bytes/Param。例如 7B 模型加载需 ~14GB。
- **KV Cache**：显存占用 = `2 × layers × heads × head_dim × precision_bytes`。
- **PEFT (LoRA)**： $\Delta W = A \times B$ （或 $\Delta W = A \cdot B$ ），通过低秩分解显著降低训练时的显存梯度存储需求。

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

- `modules/`: 核心知识组件
  - `01_foundation_rl/`: 理论根基 (MDP, TD, GAE)
  - `02_architecture/`: 架构核心与扩展 (LLM, VLM, Generation)
  - `03_alignment/`: 能力塑造与对齐技术 (SFT, PPO, DPO, GRPO)
  - `04_advanced_topics/`: 算法扩展 (Offline RL / CQL)
  - `05_engineering/`: 系统性能与工程 (DeepSpeed, Megatron, CUDA, Inference)
  - `06_agent/`: 应用闭环与智能体 (Planning, Tools, Memory)
- `tools/`: 技术摘要生成、自动化回归测试工具
- `data/`: 模拟训练数据
- `output/`: 训练产物、日志与测试报告

---

## 🧪 系统健康度验证

```bash
python tools/smoke_test.py  # 验证全模块运行逻辑，结果输出至 output/smoke_reports/
```
