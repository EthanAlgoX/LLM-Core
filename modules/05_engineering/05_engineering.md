# Systems 分类

> [!TIP]
> **一句话通俗理解**：把一个大到塞不进单张卡的模型，切开分发到成百上千张卡协同训练

## 定义与目标

- **定义**：Engineering 模块关注大模型训练与推理的系统级优化，包括算子、并行和显存管理。
- **目标**：在同等硬件下提升吞吐、降低显存、缩短训练时长并保持收敛稳定。

## 适用场景与边界

- **适用场景**：用于分布式训练、推理加速与系统瓶颈定位。
- **不适用场景**：不适用于缺少性能观测指标的“盲调”优化。
- **使用边界**：优化结论受硬件拓扑、并行策略与请求分布影响。

## 关键步骤

1. 先在单机侧做算子与内核优化（`cuda`）。
2. 再用混合精度减少计算与显存开销（`mixed_precision`）。
3. 最后引入系统级并行与 ZeRO 优化（`deepspeed`）。

建议学习顺序：`cuda -> mixed_precision -> deepspeed`

## 关键公式

`GlobalBatch = micro_batch * grad_accum * data_parallel_size`

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

## 与相近方法对比

| 方法 | 优点 | 局限 | 适用场景 |
| --- | --- | --- | --- |
| 本文主题方法 | 紧贴本节问题定义 | 依赖数据与实现质量 | 适合结构化评测与迭代优化 |
| 对比方法A | 上手成本更低 | 能力上限可能受限 | 快速原型与基线对照 |
| 对比方法B | 上限潜力更高 | 调参与资源成本更高 | 高要求生产或复杂任务场景 |

## 参考资料

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed](https://www.deepspeed.ai/)
