# 数据工程与模型评估 (Data & Evaluation)

## 数据处理 (Data Processing)

高质量数据是 LLM 能力的上限。

### 1. 预训练与 SFT 数据清洗

- **去重 (Deduplication)**：使用 MinHash 或 LSH 算法剔除海量网页中的重复内容。
- **语言过滤**：使用特征哈希或 FastText 识别语种。
- **质量评分**：利用启发式规则（如符号密度、困惑度 PPL）或小模型打分剔除垃圾数据。

### 2. 指令遵循数据采样

- **多样性 (Diversity)**：通过 K-Means 聚类确保指令覆盖数学、代码、创意写作等多个维度。
- **复杂度采样**：优先保留逻辑链条 (CoT) 完整的高质量样本。

### 3. 合成数据与仿真 (Synthetic & Simulation)

- **Adversarial User Generation**：合成具备挑战性、甚至是“有毒”的边缘案例（Edge Cases），用于测试 Agent 的安全性与鲁棒性。
- **Multi-turn Interaction Synthesis**：利用 LLM 模拟多轮对话轨迹，解决冷启动时真实交互数据匮乏的问题。
- **隐私保护 (Privacy-preserving)**：在合成数据中自动剔除或替换敏感 PII 信息（个人身份信息），确保训练数据的合规性。

---

## 模型评估 (Model Evaluation)

如何量化“大模型变聪明了”？

### 1. 自动化评测 (Benchmarks)

- **选择题类**：MMLU, C-Eval, GSM8K (数学), HumanEval (代码)。
- **痛点**：Benchmark 污染问题（题目出现在训练集中）。

### 2. 智能体评测 (Agent Evaluation)

- **任务成功率 (Success Rate)**：针对具体指令（如“订一张机票”）的端到端完成情况。
- **LLM-as-a-Judge**：利用强模型 (GPT-4) 作为裁判。引入 **泛化性分析**，确保模型不是在背诵特定的 Tool Calling 序列。
- **对抗性测试 (Adversarial Testing)**：通过模拟器发起非预期指令，评估 Agent 的拒绝服务与防御能力。

### 3. 长效评估工具链

- **Elo Rating**：类似于竞技游戏的排名系统，通过模型双盲对战获取相对胜率。
- **持续评测 (Continuous Eval)**：集成到 CI/CD 流程中，确保每次微调（SFT/DPO）不会导致旧能力的退化（Regression）。
