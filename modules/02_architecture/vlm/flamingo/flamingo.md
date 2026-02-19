# Flamingo

## 定位与分类

- **阶段**：大规模多模态预训练（Multimodal Pre-training）。
- **类型**：原生交错式 VLM（Few-shot Learner）。
- **作用**：Flamingo 是 DeepMind 开发的一系列视觉-语言模型，它最大的特色是能够处理**图文交错（Interleaved）**的输入，并且具有极强的**少样本学习（Few-shot）**能力。

## 什么是 Flamingo？

Flamingo 就像是一个能够看懂图画书并理解上下文的“超级读者”。
与 BLIP-2 这种桥接式模型不同，Flamingo 直接在冻结的语言网络内部“动手术”：它在语言模型的神经元之间插入了一些专门的“视觉接收器”（Cross-Attention），让模型在生成每一个词时都能实时参考图像内容。

## 关键架构步骤

1. **Perceiver Resampler (感知重采样器)**：
   - 将视觉编码器输出的任意数量的特征向量，压缩并映射为固定数量的（通常是 64 个）视觉 Token。这使得模型可以处理不同分辨率的图片。
2. **Frozen LLM (冻结语言模型)**：
   - 使用一个强大的、预训练好的语言模型（如 Chinchilla）作为骨干，保持其参数在大规模多模态训练中不动，以保留原生的文本生成能力。
3. **Gated Cross-Attention (门控跨注意力)**：
   - 在 LLM 的层间插入新的注意力层。这些层通过一个**门控参数（tanh gate）**初始化为 0，确保训练初期不扰乱原始语言模型的输出。

## 核心数学公式

### 1. 门控跨注意力机制 (tanh Gating)

$$y = x + \tanh(\alpha) \cdot \mathrm{CrossAttention}(x, z)$$

- $x$：语言模型的隐层状态。
- $z$：来自 Perceiver Resampler 的视觉 Tokens。
- $\alpha$：一个初始为 0 的可学习参数。这保证了“冷启动”时的稳定性。

### 2. 统一建模目标

$$P(y | x) = \prod_{t=1}^L P(y_t | y_{<t}, x_{<t})$$

- 其中 $x_{<t}$ 包含了文本和**交错出现**的图像。Flamingo 通过掩码机制确保文本只关注同一序列中出现在其之前的图像。

## 与相近方法区别

1. 相比 `LLaVA`：Flamingo 的视觉信息注入更深层、更持续。
2. 相比 `BLIP2`：Flamingo 不是单桥接 Q-Former 路径。
3. 相比纯视觉模型：Flamingo 保留强语言生成能力。

## 运行

```bash
cd <YOUR_PROJECT_ROOT>/pre_train/vlm/flamingo

conda activate finetune
python code/flamingo.py --dry-run
```

## 输出结果

默认输出到 `output/flamingo_metrics`，包含：

- `result.json`
- `preview.png`（若安装 matplotlib）

## 目录文件说明（重点）

- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
