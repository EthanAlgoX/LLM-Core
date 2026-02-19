# Systems 分类

> [!TIP]
> **一句话通俗理解**：把一个大到塞不进单张卡的模型，切开分发到成百上千张卡协同训练

## 定义与目标

- **定义**：Engineering 模块关注大模型训练与推理的系统级优化，包括算子、并行和显存管理。
- **目标**：在同等硬件下提升吞吐、降低显存、缩短训练时长并保持收敛稳定。

## 关键步骤

1. 先在单机侧做算子与内核优化（`cuda`）。
2. 再用混合精度减少计算与显存开销（`mixed_precision`）。
3. 最后引入系统级并行与 ZeRO 优化（`deepspeed`）。

建议学习顺序：`cuda -> mixed_precision -> deepspeed`

## 关键公式

\[
\text{Global Batch} = \text{micro\_batch} \times \text{grad\_accum} \times \text{data\_parallel\_size}
\]

符号说明：
- `micro_batch`：单卡每次前向样本数。
- `grad_accum`：梯度累积步数。
- `data_parallel_size`：数据并行副本数。

## 关键步骤代码（纯文档示例）

```python
for step, batch in enumerate(loader):
    with autocast(enabled=use_mixed_precision):
        loss = model(batch).loss / grad_accum
    scaler.scale(loss).backward()
    if (step + 1) % grad_accum == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

## 子模块导航

- `cuda`: 内核与吞吐分析。
- `mixed_precision`: FP16/BF16 训练实践。
- `deepspeed`: ZeRO 与分布式优化。
