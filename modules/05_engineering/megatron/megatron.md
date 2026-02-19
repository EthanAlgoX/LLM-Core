# Megatron 专题

> [!TIP]
> **一句话通俗理解**：模型切横刀、纵刀、流水线——三把刀解决超大模型的分布式训练难题

## 定位与分类

- **阶段**：大规模预训练系统（Training Systems）。
- **类型**：大规模分布式并行框架。
- **作用**：Megatron-LM 是 NVIDIA 开发的深度学习训练框架，专门为训练超大规模模型（如 GPT-3, GPT-4）设计。它解决了单张 GPU 显存不足以容纳百亿甚至千亿参数模型的痛点。

## 定义与目标

Megatron 是分布式并行训练的“集大成者”。
当一个模型太大，一张显卡塞不下时，Megatron 提供了多种切分手段：它可以把一个矩阵运算拆到多张卡上跑（张量并行），也可以把模型的不同层拆到不同卡上（流水线并行）。

## 关键步骤

1. **张量并行 (Tensor Parallelism, TP)**：
   - 将单个 Transformer 层内的矩阵乘法进行并行化。例如，将 Attention 的多头或 MLP 的神经元拆分到多张 GPU 上。
2. **流水线并行 (Pipeline Parallelism, PP)**：
   - 将模型的层分为不同的“段”（Stages），每张 GPU 或每组 GPU 负责一段。数据像流水线一样在段间传递。
3. **数据并行 (Data Parallelism, DP)**：
   - 在 TP 和 PP 的基础上，复制整个模型并行处理不同的数据集。
4. **专家并行 (Expert Parallelism, EP)**：
   - 专门用于 **MoE 模型**。将不同的专家分布在不同的设备上，配合 **All-to-All** 通信机制，显著降低单卡显存压力。
5. **分布式初始化 (Initialization)**：
   - 配置 `world_size` 以及 TP/PP/DP/EP 的分组，建立进程间的通信。

## 关键公式

### 1. 通信组大小计算

$$WorldSize = TP_{size} \times PP_{size} \times DP_{size} \times EP_{size}$$

- 这一公式定义了完成一个完整前向+反向过程所需的总 GPU 数量。在 MoE 模型中，EP 分组大小通常等于专家数量（或其倍数）。

### 2. 梯度累加步数 (Gradient Accumulation)

为了匹配硬件算力与目标 Batch Size：

$$GradAccum = \frac{GlobalBatchSize}{MicroBatchSize \times DP_{size}}$$

- **Micro Batch Size**：单张卡一次读入的数据量。
- **Global Batch Size**：更新一次权重所基于的总数据量。

## 与相近方法区别

1. 相比 `nanoGPT`：Megatron 更偏分布式工程，不是最小教学实现。
2. 相比 `DeepSpeed`：Megatron偏模型并行，DeepSpeed偏 ZeRO 与系统优化。
3. 相比 `mixed_precision`：并行策略解决规模问题，精度策略解决效率问题。

## 🛠️ 工程实战

### Megatron-LM 预训练启动

```python
# 关键步骤代码（示意）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```

---
## 关键步骤代码（纯文档示例）

```python
# 关键流程示意（与具体工程实现解耦）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```
