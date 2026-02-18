# LLaVA

## 定位与分类

- **阶段**：多模态指令微调（Multimodal Instruction Tuning）。
- **类型**：端到端多模态大模型（Visual Instruction Follower）。
- **作用**：LLaVA 是多模态领域的里程碑，它首次证明了通过将视觉 Token 直接喂给 LLM 并配合**视觉指令微调（Visual Instruction Tuning）**，可以获得极强的通用视觉理解能力。

## 什么是 LLaVA？

LLaVA (Large Language-and-Vision Assistant) 可以被看作是给 LLM（如 Llama/Vicuna）装上了一双“眼睛”。
它的结构非常简洁：一个图片编码器（通常是 CLIP-ViT）提取特征，一个简单的神经网路层（MLP 投影层）将这些特征翻译成 LLM 能听懂的“语言”，然后直接拼在文本 Token 后面。

## 关键训练步骤

1. **第一阶段：特征对齐预训练 (Feature Alignment)**：
   - 数据：大量的“图-文”对（如 CC3M）。
   - 策略：**冻结**视觉编码器和 LLM，**仅训练**中间的 MLP 投影层。
   - 目的：让模型学会把图片特征映射到 LLM 的向量空间。
2. **第二阶段：端到端指令微调 (Instruction Tuning)**：
   - 数据：复杂的视觉指令数据（由 GPT-4 辅助生成的对话、推理任务）。
   - 策略：**保持**视觉编码器冻结，但**全量微调**投影层和 LLM。
   - 目的：让模型学会按照指令执行多模态任务（如：讲解图中的冷笑话）。

## 核心数学公式

### 1. 输入序列构造

$$X_{input} = [ \mathrm{MLP}(f_{vision}(I)), \mathrm{Embedding}(X_{text}) ]$$

- 将图像 Token 与文本 Token 在向量空间进行物理拼接。

### 2. 生成损失

与 LLM 一致，采用自回归交叉熵损失：

$$L = - \sum_{i=1}^L \log p_\theta(x_i | x_{<i}, I)$$

- 模型在给定图像 $I$ 和上文 $x_{<i}$ 的条件下，预测下一个词 $x_i$。

## 与相近方法区别

1. 相比 `BLIP2`：LLaVA 更强调视觉指令微调数据驱动。
2. 相比 `Flamingo`：LLaVA 常见实现更轻量，Flamingo偏跨注意力深度融合。
3. 相比纯文本 SFT：LLaVA 输入包含图像模态。

## 运行

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/vlm/llava
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/llava.py --dry-run
```

## 输出结果

默认输出到 `output/llava_metrics`，包含：

- `result.json`
- `preview.png`（若安装 matplotlib）

## 目录文件说明（重点）

- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
