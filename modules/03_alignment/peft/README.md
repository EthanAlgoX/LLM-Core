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
2. **推理阶段**：将 $A \times B$ 重新合并回 $W$（即 $W_{new} = W + AB$），因此**推理延迟为零**。

### 为什么 LoRA 显存占用低？

因为它不存储庞大的梯度矩阵 $\Delta W$，仅存储细小的 $A$ 和 $B$。

## 进阶：QLoRA (Quantized LoRA)

### 核心亮点

1. **4-bit NormalFloat (NF4)**：专门为正态分布权重设计的量化格式，比 4-bit Float 精度更高。
2. **Double Quantization**：对量化常数本身再进行一次量化，节省额外的几百 MB 显存。
3. **Paged Optimizers**：将优化器状态在显存和内存之间自动切换，防止 OOM。

## 面试高频问题

1. **LoRA 的 $r$（秩）选多少合适？**
   - 通常 8 或 16 已经足够。过大的 $r$ 会增加显存但并不一定会提升精度。
2. **为什么 LoRA 能够合并到模型中？**
   - 矩阵乘法满足分配律：$W_{new}x = (W + AB)x = Wx + ABx$。推理时预计算 $W + AB$ 即可。
3. **LoRA 应该应用在哪些层？**
   - 实验表明，同时在 $Q, K, V, O$ 和 $MLP$ 层应用 LoRA 效果比仅在 $Q, V$ 层好。
