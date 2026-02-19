# 多模态 VLM (Vision-Language Models)

> [!TIP]
> **一句话通俗理解**：把图像"翻译"成语言模型能理解的格式，让 AI 能看图说话

VLM 的核心问题是**模态对齐**：如何将视觉编码器的特征空间映射到语言模型的特征空间，使 LLM 能够"理解"图像。

---

## 核心架构解析

### 1. 三大组件

所有主流 VLM 都由以下三部分组成：

```text
[图像] → [视觉编码器] → [模态对齐模块] → [语言模型 (LLM)] → [文本输出]
```

| 组件 | 常用实现 | 作用 |
| --- | --- | --- |
| **视觉编码器** | ViT (CLIP), SigLIP | 将图像切分为 Patch，提取视觉特征 |
| **模态对齐模块** | Q-Former, MLP Projector | 将视觉特征映射到语言空间 |
| **语言模型** | LLaMA, Vicuna, Qwen | 基于融合后的特征生成文本 |

### 2. 模态对齐策略对比

#### Q-Former (BLIP-2)

- **原理**：引入一组可学习的 Query Token，通过交叉注意力从视觉特征中提取语言相关信息。
- **优点**：压缩视觉信息，固定数量的 Query 输出（与图像分辨率无关），LLM 输入长度可控。
- **缺点**：Q-Former 本身需要预训练，流程复杂。

#### MLP Projector (LLaVA)

- **原理**：用一个简单的 2 层 MLP 直接将 ViT 的 Patch 特征线性映射到 LLM 的词嵌入空间。
- **优点**：极简，端到端训练，效果出人意料地好。
- **缺点**：视觉 Token 数量与图像分辨率成正比，高分辨率图像会导致 LLM 输入过长。

#### Cross-Attention (Flamingo)

- **原理**：在 LLM 的每个 Transformer 层中插入交叉注意力层，让语言 Token 直接 attend 到视觉特征。
- **优点**：视觉信息在每一层都能影响语言生成，融合更深。
- **缺点**：需要修改 LLM 架构，参数量增加显著。

### 3. 重要模型库 (Model Gallery)

| 模型 | 核心对齐方式 | 详细文档 | 核心里程碑 |
| --- | --- | --- | --- |
| **BLIP-2** | Q-Former | [BLIP-2 详述](./blip2/blip2.md) | 引入 Q-Former 解决变长视觉 Token 压缩问题 |
| **LLaVA-1.5** | MLP Projector | [LLaVA 详述](./llava/llava.md) | 视觉指令微调 (Visual Instruction Tuning) 开创者 |
| **Flamingo** | Cross-Attention | [Flamingo 详述](./flamingo/flamingo.md) | 原生交错式图文输入与极强少样本学习 |
| **InternVL** | MLP Projector | - | 超大视觉编码器与高性能中文支持 |
| **Qwen-VL** | Cross-Attention | - | 动态高分辨率与多图长序列理解 |

---

## 训练策略解析

### 两阶段训练（LLaVA 范式）

**Stage 1 - 特征对齐预训练**：

- 冻结视觉编码器和 LLM，只训练 MLP Projector。
- 数据：大规模图文对（如 CC3M）。
- 目标：让 Projector 学会将视觉特征映射到语言空间。

**Stage 2 - 指令微调**：

- 解冻 LLM（或使用 LoRA），联合训练 Projector 和 LLM。
- 数据：高质量视觉问答指令数据（如 LLaVA-Instruct）。
- 目标：让模型学会遵循多模态指令。

---

## 工程实现要点

- **分辨率与 Token 数**：高分辨率图像（如 448×448）会产生大量视觉 Token，显著增加 LLM 的计算负担。动态分辨率（如 InternVL 的 Dynamic Resolution）是主流解决方案。
- **视觉编码器冻结**：训练初期冻结视觉编码器可节省显存并防止灾难性遗忘，但会限制模型对特定领域图像的理解能力。
- **幻觉问题**：VLM 容易产生视觉幻觉（描述图中不存在的内容），通常通过负样本对比训练（如 RLHF-V）缓解。

---

## 📂 模块实战

- `code/`：包含 BLIP-2 (Q-Former) 和 LLaVA (MLP Projector) 的对比复现实验。
