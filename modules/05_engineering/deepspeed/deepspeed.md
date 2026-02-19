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

## 🛠️ 工程实战

### Step 1: ZeRO 配置文件

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "bf16": {
    "enabled": true
  },
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

**ZeRO Stage 选择指南**：

| Stage | 切分内容 | 单卡可训规模 (8×A100-80G) | 通信开销 |
| --- | --- | --- | --- |
| ZeRO-1 | 优化器状态 | ~30B | 低 |
| ZeRO-2 | + 梯度 | ~60B | 中 |
| ZeRO-3 | + 参数 | ~100B+ | 高 |

### Step 2: PyTorch 集成代码

```python
import deepspeed
import torch
from transformers import AutoModelForCausalLM

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# 2. DeepSpeed 引擎初始化
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config="ds_config.json",             # 指向上面的 JSON
)

# 3. 训练循环（替换原生 PyTorch）
for batch in dataloader:
    inputs = batch["input_ids"].to(model_engine.device)
    labels = batch["labels"].to(model_engine.device)

    outputs = model_engine(input_ids=inputs, labels=labels)
    loss = outputs.loss

    model_engine.backward(loss)          # 替代 loss.backward()
    model_engine.step()                  # 替代 optimizer.step() + zero_grad()
```

### Step 3: 启动命令

```bash
# 单机多卡
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json

# 多机多卡（hostfile 方式）
deepspeed --hostfile=hostfile.txt --num_gpus=8 train.py --deepspeed ds_config.json
```

`hostfile.txt` 格式：

```text
node1 slots=8
node2 slots=8
```

### 与 LLaMA Factory / HuggingFace Trainer 集成

```yaml
# 在 LLaMA Factory YAML 中启用 DeepSpeed
deepspeed: ds_config.json               # 自动取代默认分布式后端
```

```python
# 在 HuggingFace TrainingArguments 中启用
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="saves/model",
    deepspeed="ds_config.json",          # 一行即可
    bf16=True,
    per_device_train_batch_size=2,
)
```

---

## 原始脚本运行

```bash
cd <YOUR_PROJECT_ROOT>/post_train/systems/deepspeed
conda activate finetune
python code/deepspeed.py
```

## 输出结果

默认输出到 `output/deepspeed_metrics`，包含：

- `training_metrics.csv`
- `training_curves.png`
- `summary.json`
- `deepspeed_config_auto.json`
