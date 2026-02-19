# CUDA 专题

> [!TIP]
> **一句话通俗理解**：用低精度浮点数代替 FP32，显存省一半，速度翻倍，效果几乎不变

## 定位与分类

- **阶段**：底层的计算加速（Hardware Acceleration）。
- **类型**：并行计算架构与编程模型。
- **作用**：CUDA (Compute Unified Device Architecture) 是由 NVIDIA 推出的运算平台。它让开发者能够利用 GPU 的成千上万个核心来加速原本在 CPU 上运行缓慢的数值密集型任务（如矩阵乘法、卷积运算）。

## 什么是 CUDA？

简单来说，CUDA 是深度学习的“引擎”。
如果说 CPU 是一位精通各种杂活的“全能主管”，那么 GPU 就是由几千名只擅长算数的“计算员”组成的阵列。CUDA 则是这支阵列的**调度手册**，它负责向这些计算员分派任务，并收集他们的计算结果。

## 关键编程步骤

1. **Host-to-Device 数据拷贝**：
   - 将内存（CPU）中的数据通过 PCIe 总线拷贝到显存（GPU）。
2. **核函数 (Kernel) 启动**：
   - 定义计算逻辑并指定**执行配置 (Execution Configuration)**，即启动多少个线程块 (Blocks) 和每个块多少个线程 (Threads)。
3. **并行计算执行**：
   - GPU 上的大量核心同时执行相同的指令，但处理不同的数据分片 (SIMT)。
4. **Device-to-Host 数据回传**：
   - 将 GPU 计算好的结果拷贝回 CPU 内存供后续处理。

## 核心内存架构

### 1. 线程分层模型

- **Grid (网格)**：一次核函数启动的总规模。
- **Block (线程块)**：一组可以共享内存并进行同步的线程。
- **Thread (线程)**：执行计算的最小单元。

### 2. 存储分层 (Memory Hierarchy)

- **Global Memory (全局显存)**：容量大但延迟高，所有线程可见。
- **Shared Memory (共享内存)**：容量极小但速度极快，同一个 Block 内部的线程可见，常用于优化带宽。
- **Registers (寄存器)**：最快的存储，每个线程独占。

## 与相近方法区别

1. 相比 `mixed_precision`：CUDA 关注设备与算子，混合精度关注数值格式。
2. 相比 `DeepSpeed`：CUDA 是底层执行层，DeepSpeed 是上层系统优化。
3. 相比算法模块：CUDA 不改变学习目标，仅影响训练效率。

## 🛠️ 工程实战

### Triton Kernel 编写（推荐入门方式）

Triton 是 OpenAI 开源的 GPU 编程语言，比原生 CUDA C 更易上手：

```python
import triton
import triton.language as tl
import torch

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """GPU 并行向量加法 Kernel"""
    pid = tl.program_id(0)                          # 当前线程块 ID
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 每个线程负责的元素索引
    mask = offsets < n_elements                       # 边界检查

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

# 调用
def vector_add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output

x = torch.randn(10000, device="cuda")
y = torch.randn(10000, device="cuda")
result = vector_add(x, y)
```

### PyTorch 自定义 CUDA Extension

```python
# 使用 torch.utils.cpp_extension 编译自定义算子
from torch.utils.cpp_extension import load_inline

cuda_source = """
__global__ void relu_kernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = fmaxf(x[idx], 0.0f);
    }
}

torch::Tensor custom_relu(torch::Tensor x) {
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), n);
    return x;
}
"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources="torch::Tensor custom_relu(torch::Tensor x);",
    cuda_sources=cuda_source,
    functions=["custom_relu"],
)

x = torch.randn(1000, device="cuda")
result = custom_ops.custom_relu(x)  # 自定义 CUDA ReLU
```

### GPU 性能分析

```python
# PyTorch Profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
) as prof:
    for step, batch in enumerate(dataloader):
        output = model(batch)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        prof.step()
```

```bash
# NVIDIA Nsight Systems 命令行性能分析
TRAIN_ENTRY="<training_entry>"  # 训练入口占位符（由你自己的外部工程提供）
nsys profile -o report python "$TRAIN_ENTRY"

# 查看 GPU 利用率
nvidia-smi dmon -s u -d 1
```

---

## 原始脚本运行

```bash
cd <YOUR_PROJECT_ROOT>/post_train/systems/cuda
conda activate finetune
# 纯文档仓库：历史脚本命令已归档
```
