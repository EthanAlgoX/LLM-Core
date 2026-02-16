# Megatron

目录结构：
- `code/`: 代码（`code/megatron.py`）
- `data/`: toy 语料与数据统计
- `models/`: 最终模型参数
- `checkpoints/`: 训练中间 checkpoint
- `output/`: 指标、曲线图、配置快照

运行：
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/megatron
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune

# 默认可运行（Torch 训练路径）
python code/megatron.py

# 尝试检测 Megatron 依赖并导出并行配置
python code/megatron.py --use-megatron --tensor-model-parallel-size 2 --pipeline-model-parallel-size 2
```

说明：
- 该脚本用于学习 Megatron 并行配置思路（TP/PP/DP），并提供可直接运行的 toy Causal LM 训练。
- 输出目录 `output/megatron_metrics` 包含：
  - `training_metrics.csv`
  - `training_log.json`
  - `training_curves.png`
  - `summary.json`
- 并行配置快照在 `output/megatron_config_auto.json`。
