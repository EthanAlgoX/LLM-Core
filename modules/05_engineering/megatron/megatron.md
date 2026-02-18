# Megatron 专题

## 定位与分类

- **阶段**：大规模预训练系统（Training Systems）。
- **类型**：大规模分布式并行框架。
- **作用**：Megatron-LM 是 NVIDIA 开发的深度学习训练框架，专门为训练超大规模模型（如 GPT-3, GPT-4）设计。它解决了单张 GPU 显存不足以容纳百亿甚至千亿参数模型的痛点。

## 什么是 Megatron？

Megatron 是分布式并行训练的“集大成者”。
当一个模型太大，一张显卡塞不下时，Megatron 提供了多种切分手段：它可以把一个矩阵运算拆到多张卡上跑（张量并行），也可以把模型的不同层拆到不同卡上（流水线并行）。

## 关键并行步骤

1. **张量并行 (Tensor Parallelism, TP)**：
   - 将单个 Transformer 层内的矩阵乘法进行并行化。例如，将 Attention 的多头或 MLP 的神经元拆分到多张 GPU 上。
2. **流水线并行 (Pipeline Parallelism, PP)**：
   - 将模型的层分为不同的“段”（Stages），每张 GPU 或每组 GPU 负责一段。数据像流水线一样在段间传递。
3. **数据并行 (Data Parallelism, DP)**：
   - 在 TP 和 PP 的基础上，进一步增加总卡数，复制整个模型并行处理不同的数据集。
4. **分布式初始化 (Initialization)**：
   - 配置 `world_size` 以及 TP/PP/DP 的分组，建立进程间的通信（如 NCCL）。

## 核心数学公式

### 1. 通信组大小计算

$$WorldSize = TP_{size} \times PP_{size} \times DP_{size}$$

- 这一公式定义了完成一个完整前向+反向过程所需的总 GPU 数量。

### 2. 梯度累加步数 (Gradient Accumulation)

为了匹配硬件算力与目标 Batch Size：

$$GradAccum = \frac{GlobalBatchSize}{MicroBatchSize \times DP_{size}}$$

- **Micro Batch Size**：单张卡一次读入的数据量。
- **Global Batch Size**：更新一次权重所基于的总数据量。

## 与相近方法区别

1. 相比 `nanoGPT`：Megatron 更偏分布式工程，不是最小教学实现。
2. 相比 `DeepSpeed`：Megatron偏模型并行，DeepSpeed偏 ZeRO 与系统优化。
3. 相比 `mixed_precision`：并行策略解决规模问题，精度策略解决效率问题。

## 运行

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/llm/megatron
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/megatron.py
```

## 输出结果

默认输出到 `output/megatron_metrics`，包含：

- `training_metrics.csv`
- `training_curves.png`
- `summary.json`
- `megatron_config_auto.json`

## 目录文件说明（重点）

- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
