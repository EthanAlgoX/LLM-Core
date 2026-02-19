# Generation & Decoding (推理与生成优化)

> [!TIP]
> **一句话通俗理解**：生成优化是在质量不明显下降的前提下，把“首字延迟、吞吐、显存”三件事同时做平衡。

## 定义与目标

- **定义**：Generation & Decoding 关注大模型推理阶段的解码策略、缓存机制与系统级优化。
- **目标**：在高并发场景下稳定提升响应速度和资源利用率，并保持可接受的输出质量。

## 适用场景与边界

- **适用场景**：在线对话、批量生成、低延迟服务和高吞吐推理集群。
- **不适用场景**：不适用于离线训练阶段的收敛问题分析。
- **使用边界**：优化收益受硬件带宽、请求分布和模型结构共同制约。

## 关键步骤

1. 选择解码策略（Greedy、Top-k、Top-p、Beam）并明确质量目标。
2. 引入 KV Cache，避免历史 token 重复计算。
3. 使用 Flash Attention、Continuous Batching 等机制提升吞吐。
4. 按目标场景调节延迟与吞吐（TTFT vs TPS）权衡。

## 关键公式

$$
\text{Global Batch} = \text{micro\_batch} \times \text{grad\_accum} \times \text{data\_parallel\_size}
$$

符号说明：
- `micro_batch`：单卡单步处理样本数。
- `grad_accum`：梯度累积步数。
- `data_parallel_size`：数据并行副本数。

## 关键步骤代码（纯文档示例）

```python
tokens = prompt_tokens
kv_cache = None

for _ in range(max_new_tokens):
    logits, kv_cache = model.step(
        tokens[:, -1:],
        kv_cache=kv_cache,
        use_flash_attention=True,
    )
    next_token = sample(logits[:, -1, :], top_p=0.9, temperature=0.8)
    tokens = append(tokens, next_token)

decoded_text = tokenizer.decode(tokens[0])
```

## 子模块导航

- [Diffusion](./diffusion/diffusion.md)
- [DiT](./dit/dit.md)

## 工程实现要点

- 推理常见瓶颈是内存带宽而不是纯算力，需优先做 IO 优化。
- 连续批处理可显著提高 GPU 利用率，但要防止长尾请求拖慢整体时延。
- 量化（INT8/INT4）和缓存策略需要联合评估，避免出现“省显存但质量掉点过大”。

## 常见错误与排查

- **症状**：吞吐上去了，但首字延迟（TTFT）变差。  
  **原因**：批处理策略过度偏向吞吐，排队等待增加。  
  **解决**：拆分低延迟与高吞吐服务档位，分别调参。
- **症状**：长上下文下显存爆炸。  
  **原因**：KV Cache 预算不足或分页策略缺失。  
  **解决**：增加 KV 分页和长度上限控制。

## 与相近方法对比

| 方法 | 优点 | 局限 | 适用场景 |
| --- | --- | --- | --- |
| Flash + KV + 动态批处理 | 吞吐高、资源利用率好 | 系统实现复杂 | 线上高并发推理 |
| 仅静态批处理 | 实现简单 | 对变长请求不友好 | 负载稳定的离线任务 |
| 仅改解码策略 | 接入快 | 系统瓶颈改善有限 | 快速实验与参数调优 |

## 参考资料

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [vLLM](https://arxiv.org/abs/2309.06180)
- [Speculative Decoding](https://arxiv.org/abs/2302.01318)
