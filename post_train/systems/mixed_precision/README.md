# 混合精度训练（Mixed Precision）

## 定位与分类
- 阶段：训练工程优化
- 类型：数值精度与吞吐优化
- 作用：在精度可接受前提下降低显存占用、提升训练速度

## 核心原理
1. 前向与部分反向使用 `fp16/bf16`。
2. 关键参数与累计值可保持更高精度避免不稳定。
3. 动态损失缩放（fp16 场景）减少下溢风险。

## 与相近方法区别
1. 相比 `CUDA`：混合精度是数值策略，不是硬件 API 本身。
2. 相比 `DeepSpeed`：混合精度是局部技术点，可被 DeepSpeed 集成。
3. 相比算法模块：不改变目标函数，仅改变计算方式。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/systems/mixed_precision
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/mixed_precision.py
```

## 输出结果
默认输出到 `output/mixed_precision_metrics`，包含：
- `training_metrics.csv`
- `training_curves.png`
- `summary.json`
- `mixed_precision_resolved_amp.json`


## 目录文件说明（重点）
- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
