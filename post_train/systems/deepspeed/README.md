# DeepSpeed 专题

## 定位与分类

- **阶段**：训练工程优化（Training Optimization）。
- **类型**：大规模深度学习系统框架。
- **作用**：DeepSpeed 是微软开发的高性能训练库。它通过 **ZeRO (Zero Redundancy Optimizer)** 等突破性技术，极大地降低了训练超大模型所需的显存，使我们能够在有限的硬件资源上训练出更大的模型。

## 什么是 DeepSpeed？

DeepSpeed 是大模型训练的“超级内存管理器”。
在普通的分布式训练中，每个 GPU 都会完整地保存一份优化器状态、梯度和参数。对于千亿级模型，这会瞬间撑爆显存。DeepSpeed 的核心思想是：**“既然是分布式，为什么不把这些数据也分布开来存呢？”**

## ZeRO 优化阶段 (Stages)

1. **ZeRO-1 (Optimizer State Partitioning)**：
   - 将优化器状态（如 Momentum, Variance）切分并分布到不同 GPU 上。
2. **ZeRO-2 (Gradient Partitioning)**：
   - 在 ZeRO-1 的基础上，进一步将梯度分布存储。
3. **ZeRO-3 (Parameter Partitioning)**：
   - 在 ZeRO-2 的基础上，将模型参数本身也切分分布。这意味着每张卡只存模型的一部分，需要时再临时拉取。

## 关键集成步骤

1. **配置 JSON 编写**：
   - 定义 `zero_optimization` 级别、混合精度 (fp16/bf16)、梯度累加步数等。
2. **Engine 初始化**：
   - 调用 `deepspeed.initialize`，将普通的 PyTorch Model 和 Optimizer 包装成一个 `DeepSpeedEngine`。
3. **训练逻辑重构**：
   - 使用 `engine.backward(loss)` 代替 `loss.backward()`。
   - 使用 `engine.step()` 自动处理参数更新、梯度清零和梯度累加。

## 核心数学收益

### 显存压缩比

$$Memory_{ZeRO3} \approx \frac{Memory_{Baseline}}{N}$$

- 其中 $N$ 为并行的 GPU 数量。理论上，ZeRO-3 可以将显存占用降低至原先的 $1/N$。

## 与相近方法区别

1. 相比 `Megatron`：DeepSpeed 侧重系统优化与 ZeRO；Megatron强调模型并行切分。
2. 相比 `CUDA`：CUDA 是底层硬件与算子；DeepSpeed 是训练系统层。
3. 相比 `mixed_precision`：混合精度是技术点，DeepSpeed 是整体训练框架。

## 运行

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/systems/deepspeed
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/deepspeed.py
```

## 输出结果

默认输出到 `output/deepspeed_metrics`，包含：

- `training_metrics.csv`
- `training_curves.png`
- `summary.json`
- `deepspeed_config_auto.json`

## 目录文件说明（重点）

- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
