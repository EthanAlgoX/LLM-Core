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


## 目录文件说明（重点）
- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
