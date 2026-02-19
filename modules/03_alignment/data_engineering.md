# 数据工程与模型评估 (Data & Evaluation)

> [!TIP]
> **一句话通俗理解**：训练数据的质量决定模型上限，用 LLM 来给 LLM 打分做筛选

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

---

## 🛠️ 工程实战

### 1. MinHash 文本去重

```python
from datasketch import MinHash, MinHashLSH

def create_minhash(text, num_perm=128):
    """将文本转为 MinHash 指纹"""
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode("utf-8"))
    return m

# 建立 LSH 索引（阈值 0.8 = 80% 相似即认为重复）
lsh = MinHashLSH(threshold=0.8, num_perm=128)

unique_data = []
for i, item in enumerate(dataset):
    mh = create_minhash(item["text"])
    # 查询是否有近似重复
    if not lsh.query(mh):
        lsh.insert(f"doc_{i}", mh)
        unique_data.append(item)

print(f"去重前: {len(dataset)} → 去重后: {len(unique_data)}")
```

### 2. PPL 质量过滤

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

def compute_ppl(text, max_length=512):
    """计算文本困惑度（PPL 越低 = 质量越高）"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

# 过滤高 PPL（低质量）文本
filtered = [item for item in dataset if compute_ppl(item["text"]) < 50.0]
```

### 3. LLM-as-Judge 评测

```python
from openai import OpenAI

client = OpenAI()

def llm_judge(question, answer_a, answer_b):
    """使用 GPT-4 作为裁判，对比两个回答"""
    prompt = f"""请作为公正的裁判，评估以下两个 AI 助手对用户问题的回答质量。

问题：{question}
回答 A：{answer_a}
回答 B：{answer_b}

请从以下维度打分（1-10）：准确性、完整性、清晰度。
输出格式：{{"winner": "A" 或 "B", "reason": "简要原因"}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content

# 批量评测
results = []
for item in eval_dataset:
    judge_result = llm_judge(item["question"], item["model_a"], item["model_b"])
    results.append(judge_result)
```

### 4. 自动化评测 (lm-evaluation-harness)

```bash
# 安装
pip install lm-eval

# 评测 MMLU
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-7B \
    --tasks mmlu \
    --batch_size 8

# 评测 GSM8K（数学推理）
lm_eval --model hf \
    --model_args pretrained=saves/qwen2.5-7b-sft-merged \
    --tasks gsm8k \
    --num_fewshot 5

# 评测 HumanEval（代码生成）
lm_eval --model hf \
    --model_args pretrained=saves/qwen2.5-7b-sft-merged \
    --tasks humaneval \
    --batch_size 1
```

---
## 定义与目标

- **定义**：本节主题用于解释该模块的核心概念与实现思路。
- **目标**：帮助读者快速建立问题抽象、方法路径与工程落地方式。
## 关键步骤

1. 明确输入/输出与任务边界。
2. 按模块主流程执行核心算法或系统步骤。
3. 记录指标并做对比分析，形成可复用结论。
## 关键公式（逻辑表达）

\[
\text{Result} = \text{Core Method}(\text{Input}, \text{Config}, \text{Constraints})
\]

符号说明：
- \(\text{Input}\)：任务输入。
- \(\text{Config}\)：训练或推理配置。
- \(\text{Constraints}\)：方法约束（如资源、稳定性或安全边界）。
## 关键步骤代码（纯文档示例）

```python
# 关键流程示意（与具体工程实现解耦）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```
