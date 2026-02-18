# Transformer 注意力机制 (Attention Mechanisms)

## 定位与分类

- **阶段**：模型架构设计 / 推理优化。
- **类型**：特征提取内核。
- **作用**：Attention 是 Transformer 的心脏，负责建模序列内部的依赖关系。面试中常考各种变体（MHA/GQA/MQA）以及工程优化（Flash Attention）。

## 核心变体对比

| 模式 | 全称 | 键值对共享 (K/V Sharing) | 优点 | 缺点 |
| :--- | :--- | :--- | :--- | :--- |
| **MHA** | Multi-Head Attention | 每个 Query 都有专属的 K, V | 表达能力最强 | KV Cache 显存占用极大 |
| **MQA** | Multi-Query Attention | 所有 Query 共享一组 K, V | 极大减少显存，推理极快 | 精度下降明显（尤其是长文本） |
| **GQA** | Grouped-Query Attention | Query 分组，每组共享一组 K, V | **折中方案**，目前 LLM 主流（如 Llama 3） | 复杂度介于两者之间 |

### 为什么 GQA 是目前的主流？

GQA 在保持 MHA 精度（多组特征表达）的同时，显著降低了 KV Cache 的显存开销，使得长文本处理和高吞吐并发成为可能。

## 工程优化：Flash Attention

### 核心痛点

传统的 Attention 计算复杂度是 $O(N^2)$，且在显存和 SRAM 之间频繁读写中间矩阵 $S = QK^T$ 和 $P = \mathrm{softmax}(S)$，导致 **IO 受限 (Memory Bound)** 而非计算受限。

### 优化策略

1. **Tiling (分块)**：将 $Q, K, V$ 分块加载到 SRAM 中计算。
2. **Recomputation (重计算)**：反向传播时不存储 $N \times N$ 的 Attention Matrix，而是重新计算，用计算量换显存空间。
3. **IO 感知**：通过减少显存读写次数，实现 $2\times \sim 4\times$ 的端到端加速。

## 面试高频问题

1. **Softmax 为什么需要减去 Max？**
   - 为了数值稳定性，防止指数爆炸溢出。
2. **RoPE (旋转位置编码) 的优势？**
   - 具备外推性（Relative Position），通过复数乘法实现，对长文本友好。
3. **KV Cache 显存如何计算？**
   - $2 \times \mathrm{layers} \times \mathrm{heads} \times \mathrm{dim} \times \mathrm{precision}$ (针对每个 Token)。
