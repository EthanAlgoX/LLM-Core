# 多模态 VLM (Vision-Language Models)

> [!TIP]
> **一句话通俗理解**：VLM 的关键是把图像特征“翻译”为语言模型能消费的 token，再让同一套生成机制完成理解和回答。

## 定义与目标

- **定义**：VLM（Vision-Language Model）是将视觉编码器与语言模型通过对齐模块连接的多模态系统。
- **目标**：统一处理图像与文本输入，提升视觉问答、图文理解和多模态推理能力。

## 适用场景与边界

- **适用场景**：图文问答、OCR 增强理解、视觉对话、多模态 Agent。
- **不适用场景**：不适用于仅文本任务中的纯语言效率优化。
- **使用边界**：效果受视觉编码器质量、对齐策略和数据分布影响明显。

## 关键步骤

1. 用视觉编码器提取图像 patch/token 特征。
2. 用对齐模块（Q-Former、MLP Projector、Cross-Attention）映射到语言空间。
3. 将视觉 token 与文本 token 融合后输入 LLM 生成输出。
4. 用多模态指令数据做联合微调，减少幻觉与偏差。

## 关键公式

$$
\text{VLM Output} = \text{LLM}(\text{Text Tokens}, \phi(\text{Image Tokens}))
$$

符号说明：
- `Image Tokens`：视觉编码器输出的图像表示。
- `phi`：跨模态对齐映射函数（如 Q-Former/Projector）。
- `Text Tokens`：文本输入 token 序列。

## 关键步骤代码（纯文档示例）

```python
vision_tokens = vision_encoder(image)
aligned_tokens = projector(vision_tokens)  # phi(image_tokens)
multimodal_tokens = concat(aligned_tokens, prompt_tokens)
answer = llm.generate(multimodal_tokens)
```

## 子模块导航

- [BLIP-2](./blip2/blip2.md)
- [LLaVA](./llava/llava.md)
- [Flamingo](./flamingo/flamingo.md)

## 工程实现要点

- 分辨率提升会线性放大视觉 token 数，需结合压缩策略控制上下文长度。
- 冻结视觉编码器可降低训练成本，但会牺牲领域适配能力。
- 需要专门的多模态评测集与人工抽检，识别视觉幻觉风险。

## 常见错误与排查

- **症状**：模型经常描述不存在的图像内容。  
  **原因**：视觉对齐不足或训练数据偏差导致幻觉。  
  **解决**：增加困难负样本与对比式训练。
- **症状**：高分辨率场景推理延迟显著升高。  
  **原因**：视觉 token 过多导致注意力开销激增。  
  **解决**：降低分辨率或引入 token 压缩方案。

## 与相近方法对比

| 方法 | 优点 | 局限 | 适用场景 |
| --- | --- | --- | --- |
| Q-Former 路线 | Token 数可控、稳定性较好 | 训练流程复杂 | 高质量多模态系统 |
| MLP Projector 路线 | 实现简单、训练成本低 | 高分辨率场景成本高 | 快速原型和中等规模应用 |
| Cross-Attention 路线 | 融合更深、表达能力强 | 参数和算力开销大 | 高性能多模态推理 |

## 参考资料

- [BLIP-2](https://arxiv.org/abs/2301.12597)
- [LLaVA](https://arxiv.org/abs/2304.08485)
- [Flamingo](https://arxiv.org/abs/2204.14198)
