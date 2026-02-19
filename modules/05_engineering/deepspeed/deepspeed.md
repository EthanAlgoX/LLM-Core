# DeepSpeed 专题

> [!TIP]
> **一句话通俗理解**：ZeRO 把优化器状态等分散到每张卡，人人各拿一份不重复，省掉冗余显存

## 定位与分类

- **阶段**：训练工程优化（Training Optimization）。
- **类型**：大规模深度学习系统框架。
- **作用**：DeepSpeed 是微软开发的高性能训练库。它通过 **ZeRO (Zero Redundancy Optimizer)** 等突破性技术，极大地降低了训练超大模型所需的显存，使我们能够在有限的硬件资源上训练出更大的模型。

## 定义与目标

DeepSpeed 是大模型训练的“超级内存管理器”。
在普通的分布式训练中，每个 GPU 都会完整地保存一份优化器状态、梯度和参数。对于千亿级模型，这会瞬间撑爆显存。DeepSpeed 的核心思想是：**“既然是分布式，为什么不把这些数据也分布开来存呢？”**

## ZeRO 优化阶段 (Stages)

1. **ZeRO-1 (Optimizer State Partitioning)**：
   - 将优化器状态（如 Momentum, Variance）切分并分布到不同 GPU 上。
2. **ZeRO-2 (Gradient Partitioning)**：
   - 在 ZeRO-1 的基础上，进一步将梯度分布存储。
3. **ZeRO-3 (Parameter Partitioning)**：
   - 在 ZeRO-2 的基础上，将模型参数本身也切分分布。这意味着每张卡只存模型的一部分，需要时再临时拉取。

## 适用场景与边界

- **适用场景**：用于分布式训练、推理加速与系统瓶颈定位。
- **不适用场景**：不适用于缺少性能观测指标的“盲调”优化。
- **使用边界**：优化结论受硬件拓扑、并行策略与请求分布影响。

## 关键步骤

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

```python
# 关键步骤代码（示意）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```

## 输出结果

默认输出到 `output/deepspeed_metrics`，包含：

- `training_metrics.csv`
- `training_curves.png`
- `summary.json`
- `deepspeed_config_auto.json`

---
## 关键公式（逻辑表达）

`GlobalBatch = micro_batch * grad_accum * data_parallel`

符号说明：
- `micro_batch`：单卡每步样本数。
- `grad_accum`：梯度累积步数。
- `data_parallel`：数据并行副本数。
## 关键步骤代码（纯文档示例）

```python
# 关键流程示意（与具体工程实现解耦）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```

## 工程实现要点

- 先建立基准（TTFT/吞吐/显存），再做分项优化。
- 并行策略、精度策略与算子优化要协同评估。
- 保留压测脚本与配置快照，确保优化可复验。

## 常见错误与排查

- **症状**：吞吐提升但延迟恶化。  
  **原因**：批处理策略偏向吞吐，牺牲了单请求时延。  
  **解决**：按业务目标拆分延迟/吞吐档位并分别调参。
- **症状**：多机训练效率低。  
  **原因**：通信开销或并行划分与硬件拓扑不匹配。  
  **解决**：重排并行维度并用 profiler 定位通信热点。

## 参考资料

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed](https://www.deepspeed.ai/)

