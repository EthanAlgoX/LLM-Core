# Diffusion（扩散模型）

> [!TIP]
> **一句话通俗理解**：AI 画图原理：从噪声图一步步去噪，变成目标图片

## 定位与分类

- **阶段**：生成式预训练（Generative Pre-training）。
- **类型**：基于得分/噪声的生成模型。
- **作用**：它是当前 AI 绘画（Stable Diffusion）、视频生成（Sora）等生成式 AI 的核心技术。它不直接生成数据，而是学会如何将“混乱的噪声”梳理成“有意义的数据”。

## 什么是 Diffusion？

扩散模型（Diffusion Models）是一种通过**迭代去噪**来生成数据的模型。
想象你有一张清晰的照片，你不断往上面洒墨水（加噪），直到它变成一团漆黑。扩散模型学习的就是这个过程的逆过程：**给定一团墨水，预测刚才那次洒墨水的轨迹，从而一点点擦掉墨水，还原出照片**。

## 关键步骤

1. **前向扩散 (Forward Process)**：
   - 这是一个确定过程。给定初始数据 $x_0$，按照预设的调度表（Schedule）不断加入高斯噪声，直到 $x_T$ 变成完全的纯噪声。
2. **训练阶段 (Training)**：
   - 模型（如 U-Net 或 MLP）学习预测在每一个时间步 $t$ 中加入的**噪声 $\epsilon$**。
   - 目标：让模型预测的噪声与实际加入的噪声越接近越好。
3. **反向采样 (Reverse Sampling/Inference)**：
   - 从纯高斯噪声开始，利用训练好的模型，一步步预测并剔除噪声，最终还原出训练集分布中的样本。

## 核心数学公式

### 1. 前向一步加噪 (Forward Sampling)

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

- 只要知道起始点 $x_0$ 和当前时间步 $t$，我们就可以直接计算出任意时刻的带噪样本 $x_t$。

### 2. 训练损失 (Training Loss)

$$L = \mathbb{E}_{t, x_0, \epsilon} [\| \epsilon - \epsilon_\theta(x_t, t) \|^2]$$

- 模型 $\epsilon_\theta$ 的任务是：看着带噪的 $x_t$ 和时间步 $t$，猜出那个噪声 $\epsilon$ 长什么样。

### 3. 反向去噪步 (Reverse Step)

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

## 与相近方法区别

1. 相比 `DiT`：Diffusion 是方法范式，DiT 是 Transformer 化实现。
2. 相比 GAN：扩散训练通常更稳定，但采样步骤更多。
3. 相比自回归：扩散更偏连续去噪链式生成。

## 运行

```bash
cd <YOUR_PROJECT_ROOT>/pre_train/generation/diffusion

conda activate finetune
# 纯文档仓库：历史脚本命令已归档
```

## 输出结果

默认输出到 `output/diffusion_metrics`，包含：

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
