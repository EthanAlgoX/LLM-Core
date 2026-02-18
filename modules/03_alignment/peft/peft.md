# 高效微调 (PEFT: Parameter-Efficient Fine-Tuning)

## 定位与分类

- **阶段**：后训练对齐 / 微调优化。
- **类型**：轻量化微调技术。
- **作用**：PEFT 旨在仅训练极少量参数，在大规模预训练模型上实现下游任务适配。LoRA 是面试中几乎必考的高频考点。

## 核心算法：LoRA (Low-Rank Adaptation)

### 核心思想

假设模型权重的更新量 $\Delta W$ 是 **低秩 (Low-Rank)** 的。
我们可以将 $\Delta W$ 分解为两个极小的矩阵相乘：

$$\Delta W = A \times B$$

- 其中 $W \in \mathbb{R}^{d \times k}$，$A \in \mathbb{R}^{d \times r}$，$B \in \mathbb{R}^{r \times k}$，秩 $r \ll d, k$。

### 训练与推理

1. **训练阶段**：冻结原始权重 $W$，仅训练 $A$ 和 $B$。
2. **推理阶段**：将 $A \times B$ 重新合并回 $W$ （即 $W_{new} = W + AB$ ），因此**推理延迟为零**。

### 为什么 LoRA 显存占用低？

因为它不存储庞大的梯度矩阵 $\Delta W$ ，仅存储细小的 $A$ 和 $B$ 。

## 进阶：QLoRA (Quantized LoRA)

### QLoRA 技术亮点

1. **4-bit NormalFloat (NF4)**：专门为正态分布权重设计的量化格式，比 4-bit Float 精度更高。
2. **Double Quantization**：对量化常数本身再进行一次量化，节省额外的几百 MB 显存。
3. **Paged Optimizers**：将优化器状态在显存和内存之间自动切换，防止 OOM。

## 其他轻量化技术

### 1. Prefix Tuning

- **核心逻辑**：在输入 Token 前拼接一组可训练的 **Virtual Tokens (Prefix)**。
- **与 LoRA 区别**：
  - **Prefix Tuning**：改变的是输入 Hidden State，增加了一定的推理计算量。
  - **LoRA**：改变的是权重 $W$，可直接合并，推理零额外开销。

### 2. P-Tuning / Prompt Tuning

- 仅在 Embedding 层增加可训练向量，适用于任务指令极其明确的场景。

## 知识蒸馏 (Knowledge Distillation)

### 蒸馏技术亮点

- **Teacher-Student 架构**：大模型 (Teacher) 引导小模型 (Student) 学习。
- **Logits 蒸馏**：Student 拟合 Teacher 输出的概率分布。
- **能力提取**：常用于将 175B 模型的复杂逻辑蒸馏到 7B 模型中，提升端侧执行速度。

## 面试高频问题

1. LoRA 的 $r$ （秩）选多少合适？
   - 通常 8 或 16 已经足够。过大的 $r$ 会增加显存但并不一定会提升精度。
2. LoRA 与全参微调的收敛速度？
   - LoRA 收敛通常更快，因为它优化的是低秩残差，更容易在局部搜索到最优解。
3. PEFT 在多模态模型中的应用？
   - 常用于固定 ViT 编码器，仅对 Projector 或 LLM 部分进行 LoRA 微调，实现跨模态对齐。
