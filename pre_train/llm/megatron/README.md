# Megatron 专题

## 定位与分类
- 阶段：LLM 预训练工程
- 类型：大模型并行训练（张量并行/流水并行/数据并行）
- 作用：理解如何在多设备上高效训练更大模型

## 核心原理
1. Tensor Parallel：按矩阵维度切分参数与计算。
2. Pipeline Parallel：按层切分网络并流水执行。
3. Data Parallel：多副本并行处理不同批次。
4. 多并行策略可组合实现规模扩展。

## 与相近方法区别
1. 相比 `nanoGPT`：Megatron 更偏分布式工程，不是最小教学实现。
2. 相比 `DeepSpeed`：Megatron偏模型并行，DeepSpeed偏 ZeRO 与系统优化。
3. 相比 `mixed_precision`：并行策略解决规模问题，精度策略解决效率问题。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/llm/megatron
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/megatron.py
```

## 输出结果
默认输出到 `output/megatron_metrics`，包含：
- `training_metrics.csv`
- `training_curves.png`
- `summary.json`
- `megatron_config_auto.json`
