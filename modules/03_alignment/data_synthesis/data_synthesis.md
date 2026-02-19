# 数据合成 (Data Synthesis)

> **核心问题**：当真实标注数据稀缺、昂贵或存在隐私风险时，如何用程序或 LLM 生成足够多的高质量训练数据？

---

## 为什么需要数据合成？

后训练（Post-Training）阶段对数据质量的要求极高，但真实数据存在以下瓶颈：

- **标注成本**：人工标注的 SFT、RLHF 数据极昂贵，难以大规模获取。
- **覆盖盲区**：长尾推理（数学、代码逻辑）、危险边缘案例（越狱攻击）等场景，真实数据天然稀缺。
- **隐私合规**：医疗、金融等领域的真实数据无法直接用于训练。

数据合成是解决以上问题的核心工程手段。

---

## 核心方法体系

### 1. Self-Instruct / 蒸馏合成（Distillation Synthesis）

- **思路**：用一个强模型（Teacher，如 GPT-4）生成指令-回复对，用于训练弱模型（Student）。
- **代表工作**：Alpaca（LLaMA + GPT-3.5 指令蒸馏），WizardLM（Evol-Instruct 复杂化扩展）。
- **核心挑战**：**知识坍塌（Collapse）**——如果蒸馏数据分布过于单一，学生模型会失去泛化能力。

### 2. 拒绝采样（Rejection Sampling Synthesis）

- **思路**：模型自身生成候选答案（N 条），通过打分函数（奖励模型或代码执行器）筛选出高分样本，再用于 SFT。
- **优点**：闭环自我改进，数据质量有保证（通过了验证器）。
- **代表工作**：DeepSeek-R1（Step-by-step CoT 的拒绝采样）、AlphaCode（代码执行验证）。

### 3. 对抗合成（Adversarial / Red-teaming Synthesis）

- **思路**：专门合成"有意挑战模型"的边缘案例。
  - **Red-teaming LLM**：用合成的有害提示评估模型安全性。
  - **对抗性 SFT**：将被拒绝的错误案例（Negative Samples）引入训练，提高鲁棒性。
- **典型场景**：安全对齐、医疗问诊鲁棒性测试。

### 4. 环境仿真合成（Simulation-based Synthesis）

- **思路**：构建一个可交互的仿真沙盒，让 Agent 反复"演练"，收集轨迹作为训练数据。
- **数据形态**：多步骤推理轨迹（Trajectory），包含 `<Thought, Action, Observation>` 三元组。
- **代表工作**：Agentic-RL（如 WebArena, SWE-bench 上的 Agent 轨迹合成），AgentTuning。

### 5. 长思维链合成（Long CoT Synthesis）

- **思路**：通过程序化方法、数学验证器或模型引导，生成带有完整推理步骤的 Chain-of-Thought 数据。
- **关键技巧**：**Format Reward**（强制输出 `<think>...</think>` 标签），**Process Reward Model (PRM)**（对每一步推理过程打分，而不仅仅是最终答案）。
- **代表工作**：OpenAI o1, DeepSeek-R1, QwQ。

---

## 数据质量的核心指标

| 维度 | 问题 | 解决手段 |
| --- | --- | --- |
| **多样性 (Diversity)** | 数据重复导致过拟合 | K-Means 去重，嵌入距离过滤 |
| **准确性 (Accuracy)** | LLM 生成数据含有幻觉 | 代码执行器/数学验证器二次校验 |
| **复杂度 (Complexity)** | 数据太简单，无法提升能力 | Evol-Instruct（逐步增加难度） |
| **分布对齐 (Alignment)** | 合成分布与真实分布偏差大 | 人工数据做种子，合成数据做放大 |

---

## 与数据工程的区别

- **数据工程 (`data_engineering.md`)**：侧重对**现有数据**的清洗、过滤与评估（如 PPL 过滤、Dedup）。
- **数据合成 (`data_synthesis.md`)**：侧重**从头创造新数据**（无中生有，用于扩充长尾能力）。

两者在实际后训练流水线中是**互补关系**：先用数据工程处理种子数据质量，再用数据合成扩充数量与多样性。

---

## 📂 实战参考

| 工具/框架 | 功能 |
| --- | --- |
| **LLM-Blender** | 多模型回复融合，选出最优答案 |
| **Evol-Instruct** | 指令复杂化扩展（增加约束、反转问题等） |
| **WizardLM** | 基于 Evol-Instruct 的开源合成数据集 |
| **Rejection Sampling** | 用奖励模型/执行器过滤候选样本 |
