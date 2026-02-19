# CUDA 专题

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

## 运行

```bash
cd <YOUR_PROJECT_ROOT>/post_train/systems/cuda

conda activate finetune
python code/cuda.py
```

## 输出结果

默认输出到 `output/cuda_metrics`，包含：

- `benchmark.csv`
- `training_metrics.csv`
- `training_curves.png`
- `summary.json`

## 目录文件说明（重点）

- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
