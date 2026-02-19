# 数据合成 (Data Synthesis)

> [!TIP]
> **一句话通俗理解**：数据合成用模型自举扩充训练样本，再借助规则或验证器筛选高质量数据以降低标注成本。
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

## 🛠️ 工程实战

### 1. Self-Instruct 蒸馏合成

```python
from openai import OpenAI
import json

client = OpenAI()  # 或使用 vLLM 本地部署的 Teacher 模型

# 种子指令（人工编写 5~10 条高质量样本）
seed_instructions = [
    "请解释什么是梯度消失问题，以及如何解决。",
    "用 Python 实现二叉树的层序遍历。",
]

# 让 Teacher 模型基于种子生成新指令
def generate_instructions(seed, n=50):
    prompt = f"""基于以下示例指令，生成 {n} 条新的、多样化的指令。
要求：覆盖不同难度和领域（数学、代码、常识推理、创意写作）。

示例：
{json.dumps(seed, ensure_ascii=False, indent=2)}

请以 JSON 数组格式输出。"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    return json.loads(response.choices[0].message.content)

# 为每条指令生成回复
def generate_response(instruction):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": instruction}],
        temperature=0.7,
    )
    return response.choices[0].message.content

# 构建 SFT 数据集
new_instructions = generate_instructions(seed_instructions, n=100)
dataset = []
for inst in new_instructions:
    resp = generate_response(inst)
    dataset.append({"instruction": inst, "input": "", "output": resp})

with open("data/synthetic_sft.json", "w") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)
```

### 2. 拒绝采样（Rejection Sampling）

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B")

def rejection_sampling(prompt, answer, n_samples=16, threshold=0.8):
    """生成 N 个候选，用验证器筛选高质量样本"""
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048, n=n_samples)
    outputs = llm.generate([prompt], params)[0]

    accepted = []
    for output in outputs.outputs:
        response = output.text
        # 验证器：检查答案是否正确（数学/代码可用执行器）
        score = verify_answer(response, answer)
        if score >= threshold:
            accepted.append({"instruction": prompt, "output": response, "score": score})

    return accepted

def verify_answer(response, ground_truth):
    """示例：提取数字答案并对比"""
    import re
    match = re.search(r"\\boxed\{(.+?)\}", response)
    if match and match.group(1).strip() == str(ground_truth):
        return 1.0
    return 0.0

# 批量采样
for problem in math_problems:
    samples = rejection_sampling(problem["question"], problem["answer"])
    high_quality_data.extend(samples)
```

### 3. Evol-Instruct 指令进化

```python
EVOL_PROMPT = """请将以下简单指令改写为更复杂、更具挑战性的版本。
你可以通过以下方式增加复杂度：
1. 增加约束条件（如"不使用循环"、"时间复杂度 O(n)"）
2. 增加推理深度（需要多步推理才能解答）
3. 增加领域融合（结合多个知识点）

原始指令：{instruction}

改写后的复杂指令（仅输出改写结果）："""

def evolve_instruction(instruction, rounds=3):
    """多轮进化，逐步增加难度"""
    current = instruction
    for _ in range(rounds):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": EVOL_PROMPT.format(instruction=current)}],
            temperature=0.9,
        )
        current = response.choices[0].message.content
    return current

# 示例
simple = "写一个排序算法"
complex_inst = evolve_instruction(simple)
# 输出: "实现一个混合排序算法，对小于16个元素的子数组使用插入排序，
#        对更大的子数组使用归并排序，要求时间复杂度 O(n log n)，
#        空间复杂度 O(1)，并用 Python 类型注解标注所有参数。"
```

---
## 定义与目标

- **定义**：数据合成 (Data Synthesis) 属于“后训练对齐模块，聚焦 SFT、偏好优化与 RLHF 系列方法。”范畴。
- **目标**：在能力、可控性与安全性之间建立可迭代的对齐训练闭环。
## 适用场景与边界

- **适用场景**：用于构建指令跟随、偏好对齐与奖励驱动优化流程。
- **不适用场景**：不适用于缺少高质量偏好数据或评测体系的直接落地。
- **使用边界**：对齐收益受数据质量、奖励建模与 KL 约束策略影响明显。

## 关键步骤

1. 构建对齐数据与偏好信号（指令数据/偏好对/奖励模型）。
2. 在约束条件下优化策略，使输出更符合人类偏好。
3. 联合有用性、安全性与稳定性指标进行迭代评估。
## 关键公式（逻辑表达）

`J(theta) = E[r(x, y)] - beta * KL(pi_theta || pi_ref)`

符号说明：
- `r(x, y)`：奖励或偏好评分。
- `beta`：约束强度系数。
- `KL`：策略偏移惩罚项。
## 关键步骤代码（纯文档示例）

```python
# 关键流程示意（与具体工程实现解耦）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```

## 工程实现要点

- 优先保证数据质量与评测一致性，再放大训练规模。
- 在线/离线对齐需分别监控稳定性、奖励漂移与过优化风险。
- 保持参考模型与训练模型版本可追踪，便于回溯问题。

## 常见错误与排查

- **症状**：奖励升高但人工体验下降。  
  **原因**：奖励黑客或偏好模型偏差导致目标错位。  
  **解决**：引入人工抽检与多指标约束，限制单一奖励驱动。
- **症状**：训练不稳定或发散。  
  **原因**：学习率/KL 系数/批量配置不匹配。  
  **解决**：缩小超参搜索范围并分阶段增大训练强度。

## 与相近方法对比

| 方法 | 优点 | 局限 | 适用场景 |
| --- | --- | --- | --- |
| 本文主题方法 | 紧贴本节问题定义 | 依赖数据与实现质量 | 适合结构化评测与迭代优化 |
| 对比方法A | 上手成本更低 | 能力上限可能受限 | 快速原型与基线对照 |
| 对比方法B | 上限潜力更高 | 调参与资源成本更高 | 高要求生产或复杂任务场景 |

## 参考资料

- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

