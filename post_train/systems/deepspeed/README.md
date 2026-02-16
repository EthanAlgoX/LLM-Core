# DeepSpeed 专题

## 定位与分类
- 阶段：训练工程优化
- 类型：大模型训练系统
- 作用：提升吞吐、降低显存占用、支持更大规模训练

## 核心原理
1. ZeRO 对优化器状态/梯度/参数分片。
2. 多种并行与通信优化提升训练效率。
3. 与混合精度协同降低成本。

## 与相近方法区别
1. 相比 `Megatron`：DeepSpeed 侧重系统优化与 ZeRO；Megatron强调模型并行切分。
2. 相比 `CUDA`：CUDA 是底层硬件与算子；DeepSpeed 是训练系统层。
3. 相比 `mixed_precision`：混合精度是技术点，DeepSpeed 是整体训练框架。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/systems/deepspeed
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/deepspeed.py
```

## 输出结果
默认输出到 `output/deepspeed_metrics`，包含：
- `training_metrics.csv`
- `training_curves.png`
- `summary.json`
- `deepspeed_config_auto.json`


## 目录文件说明（重点）
- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
