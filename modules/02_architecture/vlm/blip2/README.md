# BLIP2

## 定位与分类

- **阶段**：多模态预训练（Multimodal Pre-training）。
- **类型**：多模态大模型（VLM）适配架构。
- **作用**：BLIP-2 通过一个名为 **Q-Former** 的核心组件，实现了在保持图像编码器和语言模型核心参数**冻结（Frozen）**的情况下，高效地将视觉信息注入到文本生成中。

## 什么是 BLIP-2？

BLIP-2 解决了多模态训练中计算成本高昂的问题。
它不再像以前的模型那样去强行端到端训练巨大的视觉和文本模型，而是通过一个“桥梁”（Q-Former）将两者连接起来：图像像是一本书，视觉编码器是扫描仪，Q-Former 则是“速读员”，它提取书中的精华（Query 向量），并讲给语言模型（LLM）听。

## 关键训练步骤

1. **第一阶段：视觉-语言表示学习 (Representation Learning)**：
   - 目标：让 Q-Former 学会如何提取对文本描述最有用的视觉特征。
   - 方法：在 Q-Former 上同时进行对比学习、生成学习和图文匹配学习。
2. **第二阶段：视觉-语言生成学习 (Generative Learning)**：
   - 目标：将 Q-Former 提取的特征通过瓶颈连接到冻结的 LLM。
   - 方法：将 Q-Former 的输出作为 LLM 的 Soft Prompt，让 LLM 根据视觉信息生成文本。

## 核心数学公式

### 1. 图像-文本对比损失 (ITC)

$$L_{itc} = - \sum_i \log \frac{\exp(s(v_i, t_i) / \tau)}{\sum_j \exp(s(v_i, t_j) / \tau)}$$

- 旨在拉近配对图像 $v_i$ 和文本 $t_i$ 的 Embedding 距离。

### 2. 图像-文本匹配 (ITM)

$$L_{itm} = \mathbb{E}[y \log p_{itm}(v, t) + (1-y) \log(1 - p_{itm}(v, t))]$$

- 一个二分类损失，预测当前的图像 $v$ 和文本 $t$ 是否是真的配对对。

### 3. 图文生成损失 (ITG)

通常使用标准的交叉熵 Loss 来训练 Q-Former 根据图像生成相关文本。

## 与相近方法区别

1. 相比 `LLaVA`：BLIP2 强调 Q-Former 桥接；LLaVA 常用 projector + 指令微调。
2. 相比 `Flamingo`：BLIP2 结构更偏“单桥接层”，Flamingo在语言层中插入跨注意力。
3. 相比纯 LLM：BLIP2 可处理图像输入。

## 运行

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/vlm/blip2
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/blip2.py --dry-run
```

## 输出结果

默认输出到 `output/blip2_metrics`，包含：

- `result.json`
- `preview.png`（若安装 matplotlib）

## 目录文件说明（重点）

- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
