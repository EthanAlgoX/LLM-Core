# Transformer Core (大语言模型架构)

LLM 的核心架构是 Transformer Decoder-Only 结构。理解其每个组件的设计动机是审计所有后续技术的基础。

> **核心公式**：$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

---

## 核心组件审计

### 1. Multi-Head Self-Attention (MHSA)

- **动机**：单头注意力只能捕获一种关联模式，多头并行允许模型同时关注不同子空间的语义关系。
- **计算流程**：
  1. 输入 $X$ 通过三个独立线性变换得到 $Q, K, V$
  2. 每个 Head 独立计算 Scaled Dot-Product Attention
  3. 所有 Head 的输出拼接后经线性投影得到最终输出
- **关键参数**：`num_heads`、`head_dim = d_model / num_heads`

### 2. 位置编码 (Positional Encoding)

| 方案 | 原理 | 代表模型 |
| --- | --- | --- |
| **Sinusoidal (绝对)** | 固定正弦/余弦函数，不可学习 | 原始 Transformer |
| **Learned (可学习)** | 每个位置对应一个可训练向量 | GPT-2, BERT |
| **RoPE (旋转)** | 通过旋转矩阵编码相对位置，外推性强 | LLaMA, Qwen |
| **ALiBi** | 在 Attention 分数上加线性偏置，无需修改嵌入 | MPT, BLOOM |

### 3. 归一化策略 (Normalization)

- **Post-LN**（原始 Transformer）：梯度不稳定，深层网络难以训练。
- **Pre-LN**（现代 LLM 标准）：将 LayerNorm 移至残差连接之前，显著提升训练稳定性。
- **RMSNorm**：去掉均值中心化，只做 RMS 缩放，计算更高效，效果相当。

### 4. FFN 层 (Feed-Forward Network)

- **标准 FFN**：$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$，隐层维度通常为 $4 \times d_{\text{model}}$。
- **SwiGLU**（LLaMA 系列）：$\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xW_2)W_3$，门控机制提升表达能力。

### 5. KV Cache（推理关键）

- **问题**：自回归生成时，每步都需要重新计算所有历史 Token 的 K/V，计算冗余。
- **方案**：将已计算的 K/V 缓存起来，每步只计算新 Token 的 K/V 并追加。
- **代价**：显存占用随序列长度线性增长：$2 \times L \times H \times d \times \text{precision}$。

---

## 工程审计要点

- **参数量估算**：$\approx 12 \times d_{\text{model}}^2 \times n_{\text{layers}}$（忽略 embedding）
- **计算量估算**：$\approx 6 \times N \times T$ FLOPs（N=参数量，T=序列长度）
- **显存分解**：权重 + 梯度 + 优化器状态 + 激活值，训练时约为推理的 12-16 倍

---

## 📂 模块实战

- `code/`：nanoGPT 最小可读实现，包含完整的 Transformer 训练闭环。
