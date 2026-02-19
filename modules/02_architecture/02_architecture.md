# Architecture 分类

> [!TIP]
> **一句话通俗理解**：架构模块回答三个核心问题: 模型怎么“看懂上下文”、怎么“跨模态理解”、怎么“高效生成”。

## 定义与目标

- **定义**：Architecture 模块聚焦模型结构层，包括 LLM、VLM 与生成解码机制。
- **目标**：建立“结构选择 -> 能力边界 -> 工程代价”的统一分析框架。

## 适用场景与边界

- **适用场景**：架构选型、技术评审、模块化知识学习。
- **不适用场景**：不适用于脱离数据与训练策略直接判断模型最终效果。
- **使用边界**：结论需和数据规模、训练配方、推理系统联合解读。

## 关键步骤

1. 先掌握 Transformer 核心计算链路（Attention、FFN、位置编码）。
2. 再扩展到多模态对齐（视觉特征映射到语言空间）。
3. 最后落到生成系统（解码策略、KV Cache、吞吐优化）。

建议学习顺序：`llm -> attention -> vlm -> generation -> diffusion/dit`

## 关键公式

$$
\text{Model Capability} \approx f(\text{Architecture}, \text{Data}, \text{Training}, \text{Inference})
$$

符号说明：
- `Architecture`：模型结构设计（如 Dense/MoE、LLM/VLM）。
- `Data`：预训练与后训练数据分布和质量。
- `Training`：优化目标、损失函数与训练策略。
- `Inference`：解码与系统优化策略。

## 关键步骤代码（纯文档示例）

```python
# 架构学习最小闭环: 结构理解 -> 代价评估 -> 对比验证
spec = parse_architecture("llm_or_vlm")
cost = estimate_compute_and_memory(spec)
report = compare_with_baselines(spec, cost)
```

## 子模块导航

- LLM 核心
- [Transformer Core](./llm/llm.md)
- [Attention Mechanisms](./llm/attention.md)
- VLM 多模态
- [VLM 总览](./vlm/vlm.md)
- [BLIP-2](./vlm/blip2/blip2.md)
- [LLaVA](./vlm/llava/llava.md)
- [Flamingo](./vlm/flamingo/flamingo.md)
- 生成与扩散
- [Generation & Decoding](./generation/generation.md)
- [Diffusion](./generation/diffusion/diffusion.md)
- [DiT](./generation/dit/dit.md)

## 工程实现要点

- 结构分析必须同时给出精度、吞吐、显存三个维度。
- 模块对比必须固定评测协议，避免“数据和配置不一致”的伪结论。
- 对高层结论保留“适用条件”说明，避免误迁移。

## 常见错误与排查

- **症状**：只看参数量做架构优劣判断。  
  **原因**：忽略路由、并行、序列长度对真实系统开销的影响。  
  **解决**：在结论中同时报告计算量、通信与内存开销。
- **症状**：VLM 与 LLM 结论混用。  
  **原因**：忽略模态对齐模块带来的额外误差与成本。  
  **解决**：分开记录单模态与多模态的评测口径。

## 与相近方法对比

| 方法 | 优点 | 局限 | 适用场景 |
| --- | --- | --- | --- |
| 按结构分层学习（本模块） | 把原理、系统、案例串成统一框架 | 初期学习成本较高 | 系统化掌握 LLM 架构 |
| 只看论文速读 | 上手快 | 易忽略工程落地约束 | 快速了解趋势 |
| 只做代码复现 | 反馈直接 | 缺少抽象总结，迁移性弱 | 工程调优和 PoC |

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LLaVA](https://arxiv.org/abs/2304.08485)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
