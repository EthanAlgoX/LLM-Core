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

## 推理框架 (Inference Frameworks)

| 框架 | 核心技术点 | 场景优势 |
| :--- | :--- | :--- |
| **vLLM** | **PagedAttention**：解决 KV Cache 显存碎片化，提升 Batch 吞吐 2-10x。 | 高并发、云端生产环境。 |
| **sglang** | **RadixAttention**：前缀缓存与端到端编译优化。 | 复杂指令流、长对话缓存共享场景。 |
| **TensorRT-LLM** | 深度算子融合 (In-flight Batching) 与硬件极致优化。 | NVIDIA 硬件环境下的极致低延迟。 |

## 性能调优 (Performance Tuning)

### 1. GPU 性能优化

- **算子融合 (Operator Fusion)**：减少 HBM (显存) 与 SRAM (计算核) 间的数据搬移。
- **Flash Attention**：通过分块计算优化 IO 瓶颈。
- **并行策略**：流水线并行 (PP)、张量并行 (TP) 与 数据并行 (DP) 的协同。

### 2. CPU 性能优化

- **模型量化**：使用 **GGUF** 格式进行 4-bit 量化，显著降低显存带宽压力。
- **SIMD 指令集**：利用 AVX-512 等向量化指令加速矩阵乘法。

## 量化深度分析 (Quantization)

- **后量化 (PTQ)**：在训练完成后直接对权重/激活进行量化（如 GPTQ, AWQ）。
- **量化感知训练 (QAT)**：在训练中模拟量化误差，通常精度损失最小。
- **精度对标**：**定点量化 (INT8/INT4)** 与 **浮点量化 (FP8)** 的数值稳定性权衡。

## 面试高频问题

1. **如何降低首字延迟 (TTFT)？**
   - 使用 Flash Attention。
   - Prefill 阶段并行化计算、投机采样验证。
2. **显存足够时，如何提升吞吐？**
   - 使用 Continuous Batching (vLLM) 动态调度。
   - 增加并发请求量，利用 PagedAttention 提升利用率。
3. **Pined Memory 与性能？**
   - 锁页内存，消除 CPU 数据到 GPU 的驱动拷贝损耗，提升搬运速度。
