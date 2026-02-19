# Megatron 专题

> [!TIP]
> **一句话通俗理解**：模型切横刀、纵刀、流水线——三把刀解决超大模型的分布式训练难题

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
   - 在 TP 和 PP 的基础上，复制整个模型并行处理不同的数据集。
4. **专家并行 (Expert Parallelism, EP)**：
   - 专门用于 **MoE 模型**。将不同的专家分布在不同的设备上，配合 **All-to-All** 通信机制，显著降低单卡显存压力。
5. **分布式初始化 (Initialization)**：
   - 配置 `world_size` 以及 TP/PP/DP/EP 的分组，建立进程间的通信。

## 核心数学公式

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

```bash
# 关键参数：3D 并行配置
TENSOR_PARALLEL=4          # TP: 同节点内 NVLink 互联的卡数
PIPELINE_PARALLEL=2        # PP: 跨节点的流水线段数
DATA_PARALLEL=2            # DP: 自动计算 = WORLD_SIZE / (TP × PP)
WORLD_SIZE=16              # 总 GPU 数 = 4 × 2 × 2
TRAIN_ENTRY="<megatron_training_entry>"  # 训练入口占位符（由你自己的外部工程提供）

# 启动 Megatron-LM GPT 预训练（纯文档示例：展示关键参数组合）
torchrun --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=6000 \
    "$TRAIN_ENTRY" \
    --tensor-model-parallel-size $TENSOR_PARALLEL \
    --pipeline-model-parallel-size $PIPELINE_PARALLEL \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --lr 1.5e-4 \
    --min-lr 1.5e-5 \
    --lr-decay-style cosine \
    --train-iters 100000 \
    --bf16 \
    --data-path my_dataset_text_document \
    --tokenizer-type HFTokenizer \
    --tokenizer-model Qwen/Qwen2.5-7B \
    --save checkpoints/megatron_gpt \
    --save-interval 1000 \
    --log-interval 10
```

### Megatron + DeepSpeed 联合训练

```bash
# 结合 Megatron 的模型并行 + DeepSpeed 的 ZeRO 优化
TRAIN_ENTRY="<megatron_training_entry>"  # 训练入口占位符（由你自己的外部工程提供）
deepspeed --num_gpus=8 "$TRAIN_ENTRY" \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --deepspeed \
    --deepspeed_config ds_config.json \
    --zero-stage 1 \
    --bf16
```

### PyTorch 层面理解 TP 切分

```python
import torch
import torch.distributed as dist

# 张量并行核心：列切分 Linear
class ColumnParallelLinear(torch.nn.Module):
    """将 Linear 的输出维度按 TP 分到不同 GPU"""
    def __init__(self, in_features, out_features, tp_size):
        super().__init__()
        self.tp_size = tp_size
        self.out_per_partition = out_features // tp_size
        self.weight = torch.nn.Parameter(
            torch.randn(self.out_per_partition, in_features)
        )

    def forward(self, x):
        # 每张卡只计算 out_features / tp_size 列
        output = torch.nn.functional.linear(x, self.weight)
        return output  # 后续通过 AllReduce 汇总

# 示例：4096 → 16384 的 MLP，4 卡 TP
# 每卡只存 4096 → 4096 的权重（1/4）
mlp = ColumnParallelLinear(4096, 16384, tp_size=4)
```

---

## 原始脚本运行

```bash
cd <YOUR_PROJECT_ROOT>/pre_train/llm/megatron
conda activate finetune
# 纯文档仓库：历史脚本命令已归档
```
