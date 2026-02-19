# Generation & Decoding (推理与生成优化)

> [!TIP]
> **一句话通俗理解**：让 AI 吐字又快又省内存——如何高效地一字一字生成文本

LLM 推理是将训练好的模型转化为实际生产力的关键环节。本模块解析从注意力算子优化到解码策略选择的完整推理链路。

---

## 核心组件解析

### 1. Flash Attention (IO 感知注意力)

**问题**：标准 Attention 需要将 $N \times N$ 的注意力矩阵写入 HBM（显存），IO 成本极高。

**解决方案**：将 Q/K/V 分块加载到 SRAM（片上缓存），在 SRAM 内完成计算，避免大矩阵的 HBM 读写。

| 版本 | 核心改进 |
| --- | --- |
| **Flash Attention v1** | 分块计算 + Online Softmax，消除 $O(N^2)$ HBM 写入 |
| **Flash Attention v2** | 优化并行策略，减少非矩阵乘法运算，提升 GPU 利用率 |
| **Flash Attention v3** | 针对 Hopper 架构 (H100) 的异步流水线优化 |

- **复杂度**：计算 $O(N^2)$ 不变，但 HBM IO 降至 $O(N)$，实际速度提升 2-4x。

### 2. KV Cache 管理

- **基础 KV Cache**：缓存历史 Token 的 K/V，推理时只计算新 Token。
- **PagedAttention (vLLM)**：将 KV Cache 分页管理（类似操作系统虚拟内存），消除显存碎片，支持更大批量。
- **MQA / GQA**：
  - **Multi-Query Attention (MQA)**：所有 Head 共享同一组 K/V，显著减少 KV Cache 体积。
  - **Grouped-Query Attention (GQA)**：折中方案，多个 Head 共享一组 K/V（LLaMA-3 采用）。

### 3. 解码策略 (Decoding Strategies)

| 策略 | 原理 | 适用场景 |
| --- | --- | --- |
| **Greedy Search** | 每步选概率最高的 Token | 确定性任务（代码生成） |
| **Beam Search** | 维护 K 条候选序列，取最优 | 机器翻译、摘要 |
| **Temperature Sampling** | 调整 Softmax 温度控制分布锐度 | 创意写作 |
| **Top-k Sampling** | 只从概率最高的 k 个 Token 中采样 | 通用对话 |
| **Top-p (Nucleus)** | 从累积概率超过 p 的最小集合中采样 | 平衡质量与多样性 |

### 4. 投机采样 (Speculative Decoding)

- **动机**：大模型推理是内存带宽瓶颈，GPU 算力大量闲置。
- **方案**：用小模型（Draft Model）快速生成多个候选 Token，大模型（Target Model）并行验证，接受或拒绝。
- **效果**：在不改变输出分布的前提下，推理速度提升 2-3x。

### 5. 量化推理 (Quantization)

| 方案 | 精度 | 显存节省 | 质量损失 |
| --- | --- | --- | --- |
| **FP16/BF16** | 半精度 | 50% vs FP32 | 几乎无 |
| **INT8 (W8A8)** | 8-bit | ~75% vs FP32 | 极小 |
| **INT4 (GPTQ/AWQ)** | 4-bit | ~87.5% vs FP32 | 可接受 |
| **GGUF (llama.cpp)** | 混合精度 | 灵活 | 取决于量化位数 |

---

## 工程实现要点

- **吞吐量瓶颈**：推理通常是内存带宽瓶颈（Memory-Bound），而非计算瓶颈（Compute-Bound）。
- **批处理策略**：连续批处理（Continuous Batching）允许不同长度的请求动态组批，大幅提升 GPU 利用率。
- **延迟 vs 吞吐**：单请求延迟优化（减少 TTFT）与批量吞吐优化（增大 batch size）存在根本性权衡。

---

## 📂 模块实战

- `code/`：包含 Flash Attention 原理演示与解码策略对比实验。
