# SFT (Supervised Fine-Tuning) 监督微调

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

### 1. 损失函数：交叉熵 (Cross-Entropy Loss)

SFT 的核心是 **Next Token Prediction**。

- **逻辑**：给定前 $n$ 个词，预测第 $n+1$ 个词。
- **目标**：最小化预测概率分布与真实离散分布（由 Output 提供）之间的距离。
- **特性**：**Teacher Forcing**。老师（标准答案）牵着模型走，每一步都必须对齐标准。

### 2. 与 PPO/GRPO 的本质区别

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

## 运行与输出

1. **启动**：`python code/sft.py`
2. **可视化**：输出至 `output/sft_metrics`。
   - `loss` 曲线应平滑下降至 1 以下甚至更低。
   - `eval_loss` 用于监控模型是否过拟合于特定的答案。
