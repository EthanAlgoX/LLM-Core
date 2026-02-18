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

---

## 模型评估 (Model Evaluation)

如何量化“大模型变聪明了”？

### 1. 自动化评测 (Benchmarks)

- **选择题类**：MMLU, C-Eval, GSM8K (数学), HumanEval (代码)。
- **痛点**：Benchmark 污染问题（题目出现在训练集中）。

### 2. 对齐评测 (Alignment Eval)

- **Elo Rating**：类似于竞技游戏的排名系统，通过模型双盲对战获取相对胜率。
- **LLM-as-a-Judge**：利用强模型 (GPT-4) 作为裁判，对候选模型的生成质量进行多维度打分。

### 3. 在线对齐指标

- **奖励函数稳定性**：监控 PPO 训练中 Reward 和 KL 散度的平衡，防止模型“作弊”骗分。
