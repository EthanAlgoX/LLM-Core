# 混合精度训练（Mixed Precision）

## 定位与分类

- **阶段**：训练工程优化（Training Optimization）。
- **类型**：混合精度（Mixed Precision）数值计算。
- **作用**：混合精度训练旨在通过结合 FP16（半精度）和 FP32（单精度）来显著提高模型训练速度并降低显存占用，同时保持与全精度训练相当的收敛精度。

## 什么是混合精度？

混合精度是深度学习训练的“平衡术”。

- **FP32 (Single Precision)**：精度高，范围广，但占用内存多且计算慢。
- **FP16/BF16 (Half Precision)**：精度较低，范围窄，但计算快且省内存。
**策略**：在计算密集型且对精度不敏感的操作（如矩阵乘法）中使用 FP16/BF16，而在数值范围敏感的累计操作（如权重更新）中保留 FP32。

## 关键训练步骤

1. **维护 FP32 权重副本 (Master Weights)**：
   - 在内存/显存中保留一份 FP32 的权重副本，用于在更新时保持精度。
2. **前向与反向计算 (FP16/BF16)**：
   - 将权重转换为 FP16，执行前向传播和反向传播计算梯度。
3. **损失缩放 (Loss Scaling - 针对 FP16)**：
   - 为了防止 FP16 因表示范围过窄导致梯度下溢（变为 0），在计算 Loss 后先乘一个很大的缩放因子 $S$。
4. **梯度更新 (FP32)**：
   - 得到 FP16 梯度后，将其还原（Unscale）并转换回 FP32，然后作用于 Master Weights。

## 核心数学逻辑

### 1. 损失缩放公式

$$\mathrm{Scaled\_Loss} = \mathrm{Loss} \times S$$
$$\mathrm{Update\_Gradient} = \frac{\nabla_{\theta_{FP16}} (\mathrm{Scaled\_Loss})}{S}$$

- 通过 $S$ 将微小的梯度“顶”回 FP16 的表示区间内。

### 2. BF16 vs FP16

- **FP16**：5 位指数，10 位尾数。范围窄，必须配合 **Loss Scaling**。
- **BF16**：8 位指数，7 位尾数。范围与 FP32 一致，精度略低。由于其范围优势，通常**不需要** Loss Scaling，是大模型训练的首选。

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
