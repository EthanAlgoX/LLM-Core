# CUDA 专题

## 定位与分类
- 阶段：训练工程基础
- 类型：硬件加速与算子性能
- 作用：理解 GPU/CUDA 对训练速度与吞吐的影响

## 核心原理
1. CUDA 通过并行线程执行张量计算。
2. 训练性能受算子效率、带宽、同步策略影响。
3. 通过 benchmark + toy 训练观察瓶颈。

## 与相近方法区别
1. 相比 `mixed_precision`：CUDA 关注设备与算子，混合精度关注数值格式。
2. 相比 `DeepSpeed`：CUDA 是底层执行层，DeepSpeed 是上层系统优化。
3. 相比算法模块：CUDA 不改变学习目标，仅影响训练效率。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/systems/cuda
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/cuda.py
```

## 输出结果
默认输出到 `output/cuda_metrics`，包含：
- `benchmark.csv`
- `training_metrics.csv`
- `training_curves.png`
- `summary.json`
