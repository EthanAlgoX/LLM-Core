# Transformer 注意力机制 (Attention Mechanisms)

> [!TIP]
> **一句话通俗理解**：注意力机制通过 Q-K 匹配动态分配信息流，GQA/MQA 与 FlashAttention 在精度与成本间给出不同工程折中。

## 定位与分类

- **阶段**：模型架构设计 / 推理优化。
- **类型**：特征提取内核。
- **作用**：Attention 是 Transformer 的心脏，负责建模序列内部的依赖关系。技术评估中常考各种变体（MHA/GQA/MQA）以及工程优化（Flash Attention）。

## 核心变体对比

| 模式 | 全称 | 键值对共享 (K/V Sharing) | 优点 | 缺点 |
| :--- | :--- | :--- | :--- | :--- |
| **MHA** | Multi-Head Attention | 每个 Query 都有专属的 K, V | 表达能力最强 | KV Cache 显存占用极大 |
| **MQA** | Multi-Query Attention | 所有 Query 共享一组 K, V | 极大减少显存，推理极快 | 精度下降明显（尤其是长文本） |
| **GQA** | Grouped-Query Attention | Query 分组，每组共享一组 K, V | **折中方案**，目前 LLM 主流（如 Llama 3） | 复杂度介于两者之间 |

### 为什么 GQA 是目前的主流？

GQA 在保持 MHA 精度（多组特征表达）的同时，显著降低了 KV Cache 的显存开销，使得长文本处理和高吞吐并发成为可能。

## 工程优化：Flash Attention

### 核心痛点

传统的 Attention 计算复杂度是 $O(N^2)$，且在显存和 SRAM 之间频繁读写中间矩阵 $S = QK^T$ 和 $P = \mathrm{softmax}(S)$，导致 **IO 受限 (Memory Bound)** 而非计算受限。

### 优化策略

1. **Tiling (分块)**：将 $Q, K, V$ 分块加载到 SRAM 中计算。
2. **Recomputation (重计算)**：反向传播时不存储 $N \times N$ 的 Attention Matrix，而是重新计算，用计算量换显存空间。
3. **IO 感知**：通过减少显存读写次数，实现 $2\times \sim 4\times$ 的端到端加速。

## 技术核心解析

1. **Softmax 为什么需要减去 Max？**
   - 为了数值稳定性，防止指数爆炸溢出。
2. **RoPE (旋转位置编码) 的优势？**
   - 具备外推性（Relative Position），通过复数乘法实现，对长文本友好。
3. **KV Cache 显存如何计算？**
   - $2 \times \mathrm{layers} \times \mathrm{heads} \times \mathrm{dim} \times \mathrm{precision}$ (针对每个 Token)。

---

## 🛠️ 工程实战

### Flash Attention 使用

```python
# 方式一：PyTorch 原生（2.0+）
import torch
import torch.nn.functional as F

q = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.bfloat16)  # [B, H, N, D]
k = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.bfloat16)

# 自动启用 Flash Attention（SDPA）
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# 方式二：flash-attn 库
from flash_attn import flash_attn_func

# [B, N, H, D] 格式
q = q.transpose(1, 2)  # → [1, 4096, 32, 128]
k = k.transpose(1, 2)
v = v.transpose(1, 2)
output = flash_attn_func(q, k, v, causal=True)
```

### GQA (Grouped-Query Attention) 实现

```python
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    """GQA: Query 分组共享 KV，Llama 3 / Qwen2.5 标配"""
    def __init__(self, d_model=4096, n_heads=32, n_kv_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads   # 每组 KV 被多少个 Q 共享

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, N, _ = x.shape
        q = self.wq(x).view(B, N, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, N, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, N, self.n_kv_heads, self.head_dim)

        # 扩展 KV 头以匹配 Q 头数量
        k = k.repeat_interleave(self.n_rep, dim=2)  # [B, N, 8, D] → [B, N, 32, D]
        v = v.repeat_interleave(self.n_rep, dim=2)

        # 转为 [B, H, N, D] 用于 SDPA
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(B, N, -1)
        return self.wo(output)

# 对比显存：MHA 32 KV heads vs GQA 8 KV heads → KV Cache 省 75%
```

### RoPE（旋转位置编码）

```python
import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算 RoPE 频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 复数形式
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """将 RoPE 应用到 Q 和 K"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# 用法：freqs_cis = precompute_freqs_cis(128, 8192)  # dim=128, max_len=8192
```

---
## 定义与目标

- **定义**：Transformer 注意力机制 (Attention Mechanisms) 属于“模型架构模块，关注 Transformer、多模态与生成机制的核心设计。”范畴。
- **目标**：理解结构选择如何影响模型表达能力、训练稳定性与推理效率。
## 适用场景与边界

- **适用场景**：用于模型结构选型、模块拆解与架构原理学习。
- **不适用场景**：不适用于脱离数据与训练策略单独评估最终能力。
- **使用边界**：实际效果受参数规模、数据分布和推理策略共同影响。

## 关键步骤

1. 确定核心结构（注意力、位置编码、归一化与前馈层）。
2. 结合序列长度与显存预算设计训练/推理路径。
3. 通过统一评测集比较结构变体的效果与效率差异。
## 关键公式（逻辑表达）

`Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V`

符号说明：
- `Q, K, V`：查询、键、值向量。
- `d_k`：键向量维度，用于缩放。
- `softmax`：将注意力分数归一化。
## 关键步骤代码（纯文档示例）

```python
# 关键流程示意（与具体工程实现解耦）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```

## 工程实现要点

- 优先明确序列长度、显存预算与吞吐目标，再做结构决策。
- 重点关注 Attention/KV Cache 的内存开销与并行策略匹配。
- 在相同评测集上比较结构变体，避免结论被数据差异干扰。

## 常见错误与排查

- **症状**：长序列下显存快速爆炸。  
  **原因**：KV Cache 与注意力开销评估不足。  
  **解决**：提前做显存预算并限制 max length 或采用更优缓存策略。
- **症状**：结构改动后效果不稳定。  
  **原因**：训练配置与初始化策略未同步调整。  
  **解决**：固定基线配置，逐项 ablation 并记录每次改动。

## 与相近方法对比

| 方法 | 优点 | 局限 | 适用场景 |
| --- | --- | --- | --- |
| 本文主题方法 | 紧贴本节问题定义 | 依赖数据与实现质量 | 适合结构化评测与迭代优化 |
| 对比方法A | 上手成本更低 | 能力上限可能受限 | 快速原型与基线对照 |
| 对比方法B | 上限潜力更高 | 调参与资源成本更高 | 高要求生产或复杂任务场景 |

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [FlashAttention](https://arxiv.org/abs/2205.14135)

