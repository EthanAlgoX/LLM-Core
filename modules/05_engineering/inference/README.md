# 推理优化 (Inference Optimization)

## 定位与分类

- **阶段**：推理与部署 (Deployment)。
- **类型**：吞吐与延迟优化。
- **作用**：随着 LLM 参数量和用户并发的增加，如何通过工程手段降低首字延迟 (TTFT) 和提升生成吞吐是面试核心。

## 核心概念：KV Cache

### 痛点

自回归生成中，每个新 Token 都要重新计算之前所有 Token 的表示，导致重复计算。

### 策略

将前向计算中的 Key 和 Value 缓存下来，后续 Token 仅需与缓存进行计算。

- **显存占用公式**：
  $$\mathrm{Mem}_{KV} = 2 \times \mathrm{layers} \times \mathrm{heads} \times \mathrm{hidden\_dim} \times \mathrm{seq\_len} \times \mathrm{precision\_bytes}$$
- **面试点**：Llama 7B (fp16) 处理 1024 长度约占用 0.5GB 显存。

## 量化技术 (Quantization)

| 技术 | 阶段 | 核心逻辑 |
| :--- | :--- | :--- |
| **GPTQ** | 后量化 (PTQ) | 逐层进行二阶导数信息补偿，将权重压到 4-bit。 |
| **AWQ** | 后量化 (PTQ) | 保护 1% 的重要权重（Saliency-based），不依赖反向传播校准。 |
| **GGUF** | 推理格式 | 苹果硅盘等 CPU 推理的主流格式，支持高度定制的量化位宽。 |

## 加速策略：投机采样 (Speculative Decoding)

### 核心思想

用一个极小的“草稿模型 (Draft Model)”预先猜测 N 个 Token，再由大模型 (Target Model) 一次性验证。

- **验证通过**：接受该 Token，相当于一次性生成多个 Token。
- **验证失败**：大模型修正回正确的 Token。
**结果**：在不改变精度的前提下，实现 $2\times \sim 3\times$ 的端到端加速。

## 面试高频问题

1. **如何降低首字延迟 (TTFT)？**
   - 使用 Flash Attention。
   - Prefill 阶段并行化计算。
2. **显存足够时，如何提升吞吐？**
   - 增加 Batch Size。
   - 使用 Continuous Batching (vLLM) 动态调度。
3. **Pined Memory 是什么？**
   - 锁页内存，加速 CPU 到 GPU 的数据搬移速度。
