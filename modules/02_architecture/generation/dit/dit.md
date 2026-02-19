# DiT（Diffusion Transformer）

> [!TIP]
> **一句话通俗理解**：AI 画图原理：从噪声图一步步去噪，变成目标图片

## 定位与分类

- **阶段**：生成式预训练（架构演进）。
- **类型**：基于 Transformer 的扩散模型（Scalable Diffusion）。
- **作用**：它是 Sora（视频生成）等最新大模型的技术支柱。它证明了扩散模型不需要复杂的 U-Net，只要简单的 Transformer 配合 Patch 机制就能实现更强的生成效果和更好的扩展性（Scaling）。

## 什么是 DiT？

DiT（Diffusion Transformer）是扩散模型的一种新式变体。
传统的扩散模型大多使用卷积神经网络（U-Net）作为骨干。DiT 借鉴了 ViT（Vision Transformer）的思想，将图像看作一串“视觉单词”（Patches），并使用 Transformer 块来处理这些单词，从而预测去噪需要的轨迹。

## 关键结构步骤

1. **Patchify (切片化)**：
   - 将输入的带噪图像（或潜在空间变量 Latent）切分成固定大小的 $p \times p$ 小块，并展平为序列。
2. **Time/Condition Embedding (条件注入)**：
   - 将当前的时间步 $t$ （以及可能的分类标签或文本描述）通过多层感知机转为向量，注入到 Transformer 的每一层中。
3. **Transformer Processing (注意力处理)**：
   - 使用多层 Self-Attention 块处理 Patch 序列，捕捉像素间的全局关联，这在生成大尺寸或复杂结构图像时比卷积更有优势。
4. **Unpatchify (反向还原)**：
   - 将 Transformer 输出的向量序列还原为预测的噪声图。

## 核心数学公式

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

## 运行

```bash
cd <YOUR_PROJECT_ROOT>/pre_train/generation/dit

conda activate finetune
# 纯文档仓库：历史脚本命令已归档
```

## 输出结果

默认输出到 `output/dit_metrics`，包含：

- `training_metrics.csv`
- `training_curves.png`
- `generated_samples.pt`
- `target_samples.pt`
- `summary.json`

## 目录文件说明（重点）

- `历史脚本（归档）`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
