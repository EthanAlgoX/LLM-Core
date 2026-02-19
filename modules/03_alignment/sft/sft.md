# SFT (Supervised Fine-Tuning) 监督微调

> [!TIP]
> **一句话通俗理解**：拿人类写的高质量问答对，手把手教模型"怎么说话"

## 定位与分类

- **阶段**：后训练（Post-training）之对齐起点。
- **类型**：监督学习（Supervised Learning）。
- **作用**：将预训练基座模型（Base Model）转化为能够遵循人类指令（Instruction Following）的对话模型（Chat Model）。它是 RLHF 流程的物理基础。

## 模型训练的关键步骤

SFT 处理流程遵循以下核心步骤：

1. **数据分词 (Tokenization)**：将指令（Instruction）与回答（Output）拼接，并转换为模型可读的 Token IDs。
2. **掩码处理 (Label Masking)**：在计算损失时，通常将指令部分的标签置为 `-100`（忽略），确保模型仅学习如何生成回答，而不去学如何复述指令。
3. **前向传播 (Forward Pass)**：模型根据 Prompt 预测下一个字符（Token）的概率分布。
4. **损失计算 (Loss Calculation)**：使用**交叉熵（Cross-Entropy）**对比预测值与标准答案。
5. **反向传播与优化 (Backprop & Update)**：根据梯度更新模型权重（或 LoRA 权重）。

## 核心原理与损失函数

### 1. 关键公式：交叉熵损失 (Cross-Entropy Loss)

SFT 的本质是**最大似然估计（MLE）**，其核心数学目标是最小化回答序列的负对数似然：

$$L(\theta) = - \sum_{i=1}^{T} \log P_\theta(y_i | x, y_{1}, \dots, y_{i-1})$$

**公式拆解与理解：**

- **$x$ (Input)**：输入的指令内容（Prompt）。
- **$y_i$ (Target)**：标准答案中第 $i$ 个位置的词（Token）。
- **$P_\theta(\dots)$**：模型根据当前参数 $\theta$，在已知指令和前序文字的前提下，预测出正确下一个词的“概率”。
- **$\log$ 与负号**：将概率转化为损失值。概率越大（预测越准）， $\log$ 越接近 0，损失值越小。

### 2. 深度解读：如何直观理解这个过程？

- **逐词对齐 (Token-level Alignment)**：模型在每一个步长上都在尝试预测“下一个词”。它在学习标准答案中词与词之间的统计规律。
- **Teacher Forcing (强制纠偏)**：这是 SFT 的关键特征。在训练前向传播时，无论模型预测出的上一个词是否正确，模型在计算当前词时输入的永远是**真实答案**中的前文。就像老师牵着手写字，错了一笔立即拉回。
- **概率最大化**：公式的终极目的是让模型在看到特定指令时，能够以“最大概率”吐出数据集里的标准字句。

### 3. 与 PPO/GRPO 的本质区别

| 特性 | SFT (监督微调) | RL (PPO/GRPO) |
| :--- | :--- | :--- |
| **学习源** | **静态标签**（Output 字对字模仿）。 | **动态反馈**（Reward 打分驱动）。 |
| **灵活性** | 低。模型被限制在模仿数据集。 | 高。模型可以探索数据集之外更好的解。 |
| **稳定性** | 极高。最简单的梯度下降。 | 低。容易发散，需要复杂的超参控制。 |

## 关键配置解读

| 参数 | 建议值 | 原理解读 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` 或 `5e-5` | 相比 RL，SFT 使用较高的学习率以快速学习任务模式。 |
| `cutoff_len` | `1024` | 决定了模型单次能处理的问题+答案的总长度。 |
| `lora_target` | `all` | 为所有线性层添加低秩适配器，可以在提升效果的同时极大节省显存。 |

## 🛠️ 工程实战：使用 LLaMA Factory 进行 SFT

[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) 是目前最流行的开源微调框架，支持 100+ 模型、LoRA/QLoRA/全量微调、WebUI 可视化训练。

### Step 1: 环境准备

```python
# 关键步骤代码（示意）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
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

`Result = CoreMethod(Input, Config, Constraints)`

符号说明：
- `Input`：任务输入。
- `Config`：训练或推理配置。
- `Constraints`：方法约束（如资源、稳定性或安全边界）。
## 关键步骤代码（纯文档示例）

```python
# 关键流程示意（与具体工程实现解耦）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```
