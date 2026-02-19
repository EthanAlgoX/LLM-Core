# DiT（Diffusion Transformer）

> [!TIP]
> **一句话通俗理解**：DiT 用 Transformer 替代 U-Net 处理扩散过程中的 token 化图像表示，在规模化训练下提升生成质量。

## 定位与分类

- **阶段**：生成式预训练（架构演进）。
- **类型**：基于 Transformer 的扩散模型（Scalable Diffusion）。
- **作用**：它是 Sora（视频生成）等最新大模型的技术支柱。它证明了扩散模型不需要复杂的 U-Net，只要简单的 Transformer 配合 Patch 机制就能实现更强的生成效果和更好的扩展性（Scaling）。

## 定义与目标

DiT（Diffusion Transformer）是扩散模型的一种新式变体。
传统的扩散模型大多使用卷积神经网络（U-Net）作为骨干。DiT 借鉴了 ViT（Vision Transformer）的思想，将图像看作一串“视觉单词”（Patches），并使用 Transformer 块来处理这些单词，从而预测去噪需要的轨迹。

## 适用场景与边界

- **适用场景**：用于模型结构选型、模块拆解与架构原理学习。
- **不适用场景**：不适用于脱离数据与训练策略单独评估最终能力。
- **使用边界**：实际效果受参数规模、数据分布和推理策略共同影响。

## 关键步骤

1. **Patchify (切片化)**：
   - 将输入的带噪图像（或潜在空间变量 Latent）切分成固定大小的 $p \times p$ 小块，并展平为序列。
2. **Time/Condition Embedding (条件注入)**：
   - 将当前的时间步 $t$ （以及可能的分类标签或文本描述）通过多层感知机转为向量，注入到 Transformer 的每一层中。
3. **Transformer Processing (注意力处理)**：
   - 使用多层 Self-Attention 块处理 Patch 序列，捕捉像素间的全局关联，这在生成大尺寸或复杂结构图像时比卷积更有优势。
4. **Unpatchify (反向还原)**：
   - 将 Transformer 输出的向量序列还原为预测的噪声图。

## 关键公式

### 1. 输入 Token 化

$$z_{tokens} = \mathrm{Patchify}(x_t) + \mathrm{PositionalEmbedding}$$

### 2. 条件注入 (Adaptive Layer Norm)

DiT 常用如下方式将时间信息 $c$ 注入：

$$\mathrm{adaLN}(h, c) = w_c \cdot \mathrm{LayerNorm}(h) + b_c$$

- 其中 $w_c$ 和 $b_c$ 是基于时间步计算出的缩放和平移系数。

### 3. 统一目标函数

与标准 Diffusion 一致：

$$\min_\theta \mathbb{E}_{x_0, \epsilon, t} [ \| \epsilon - \mathrm{DiT}_\theta(x_t, t) \|^2 ]$$

## 与相近方法区别

1. 相比 `Diffusion` 基础实现：DiT 更强调 token 化与全局注意力。
2. 相比 CNN-U-Net：DiT 通常更易扩展到大模型规模。
3. 相比 LLM：DiT 处理图像/latent token，不是自然语言 token。

## 关键步骤代码（纯文档示例）

```python
# 关键步骤代码（示意）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```

## 输出结果

默认输出到 `output/dit_metrics`，包含：

- `training_metrics.csv`
- `training_curves.png`
- `generated_samples.pt`
- `target_samples.pt`
- `summary.json`

## 目录文件说明（重点）

- 关键步骤代码：见“关键步骤代码（纯文档示例）”章节。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。

## 工程实现要点

- 优先明确序列长度、显存预算与吞吐目标，再做结构决策。
- 重点关注 Attention/KV Cache 的内存开销与并行策略匹配。
- 在相同评测集上比较结构变体，避免结论被数据差异干扰。

## 常见错误与排查

- **症状**：长序列下显存快速爆炸。  
  **原因**：KV Cache 与注意力开销评估不足。  
  **解决**：提前做显存预算并限制 max length 或采用更优缓存策略。
- **症状**：结构改动后效果不稳定。  
  **原因**：训练配置与初始化策略未同步调整。  
  **解决**：固定基线配置，逐项 ablation 并记录每次改动。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [FlashAttention](https://arxiv.org/abs/2205.14135)

