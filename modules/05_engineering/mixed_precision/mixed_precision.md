# 混合精度训练（Mixed Precision）

> [!TIP]
> **一句话通俗理解**：用低精度浮点数代替 FP32，显存省一半，速度翻倍，效果几乎不变

## 定位与分类

- **阶段**：训练工程优化（Training Optimization）。
- **类型**：混合精度（Mixed Precision）数值计算。
- **作用**：混合精度训练旨在通过结合 FP16（半精度）和 FP32（单精度）来显著提高模型训练速度并降低显存占用，同时保持与全精度训练相当的收敛精度。

## 什么是混合精度？

混合精度是深度学习训练的“平衡术”。

- **FP32 (Single Precision)**：精度高，范围广，但占用内存多且计算慢。
- **FP16/BF16 (Half Precision)**：精度较低，范围窄，但计算快且省内存。
**策略**：在计算密集型且对精度不敏感的操作（如矩阵乘法）中使用 FP16/BF16，而在数值范围敏感的累计操作（如权重更新）中保留 FP32。

## 适用场景与边界

- **适用场景**：用于分布式训练、推理加速与系统瓶颈定位。
- **不适用场景**：不适用于缺少性能观测指标的“盲调”优化。
- **使用边界**：优化结论受硬件拓扑、并行策略与请求分布影响。

## 关键步骤

1. **维护 FP32 权重副本 (Master Weights)**：
   - 在内存/显存中保留一份 FP32 的权重副本，用于在更新时保持精度。
2. **前向与反向计算 (FP16/BF16)**：
   - 将权重转换为 FP16，执行前向传播和反向传播计算梯度。
3. **损失缩放 (Loss Scaling - 针对 FP16)**：
   - 为了防止 FP16 因表示范围过窄导致梯度下溢（变为 0），在计算 Loss 后先乘一个很大的缩放因子 $S$。
4. **梯度更新 (FP32)**：
   - 得到 FP16 梯度后，将其还原（Unscale）并转换回 FP32，然后作用于 Master Weights。

## 关键公式

### 1. 损失缩放公式

$$\mathrm{Scaled\_Loss} = \mathrm{Loss} \times S$$

$$\mathrm{Update\_Gradient} = \frac{\nabla_{\theta_{FP16}} (\mathrm{Scaled\_Loss})}{S}$$

- 通过 $S$ 将微小的梯度“顶”回 FP16 的表示区间内。

### 2. BF16 vs FP16

- **FP16**：5 位指数，10 位尾数。范围窄，必须配合 **Loss Scaling**。
- **BF16**：8 位指数，7 位尾数。范围与 FP32 一致，精度略低。由于其范围优势，通常**不需要** Loss Scaling，是大模型训练的首选。

## 与相近方法区别

1. 相比 `CUDA`：混合精度是数值策略，不是硬件 API 本身。
2. 相比 `DeepSpeed`：混合精度是局部技术点，可被 DeepSpeed 集成。
3. 相比算法模块：不改变目标函数，仅改变计算方式。

## 🛠️ 工程实战

### PyTorch AMP（自动混合精度）

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()  # FP16 专用的 Loss Scaler

for batch in dataloader:
    inputs, labels = batch["input_ids"].cuda(), batch["labels"].cuda()

    # 自动将部分计算转为 FP16
    with autocast(dtype=torch.float16):
        outputs = model(inputs, labels=labels)
        loss = outputs.loss

    # Loss Scaling + 反向传播
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### BF16 训练（A100/H100 推荐）

```python
# BF16 不需要 GradScaler（范围与 FP32 一致）
with autocast(dtype=torch.bfloat16):
    outputs = model(inputs, labels=labels)
    loss = outputs.loss

loss.backward()
optimizer.step()
optimizer.zero_grad()
```

### HuggingFace Trainer 中启用

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="saves/model",
    bf16=True,                  # 启用 BF16（A100+）
    # fp16=True,                # 或启用 FP16（V100/T4）
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
)
```

### 精度对比基准测试

```python
import torch
import time

def benchmark_precision(dtype, n=1000):
    """对比不同精度的矩阵乘法性能"""
    a = torch.randn(4096, 4096, device="cuda", dtype=dtype)
    b = torch.randn(4096, 4096, device="cuda", dtype=dtype)

    # 预热
    for _ in range(10):
        _ = torch.mm(a, b)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(n):
        _ = torch.mm(a, b)
    torch.cuda.synchronize()

    elapsed = time.time() - start
    print(f"{dtype}: {elapsed:.3f}s ({n/elapsed:.0f} ops/s)")

benchmark_precision(torch.float32)   # FP32 基准
benchmark_precision(torch.float16)   # FP16（V100 约 2x 提速）
benchmark_precision(torch.bfloat16)  # BF16（A100 约 2x 提速）
```

---

## 关键步骤代码（纯文档示例）

```python
# 关键步骤代码（示意）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```

---
## 定义与目标

- **定义**：混合精度训练（Mixed Precision） 属于“工程优化模块，关注训练与推理系统的吞吐、显存和稳定性。”范畴。
- **目标**：在既定硬件预算下实现更高性能与更稳定的端到端流程。

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

