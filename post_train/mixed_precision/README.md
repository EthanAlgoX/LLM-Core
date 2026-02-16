# Mixed Precision

目录结构：
- `code/`: 代码（`code/mixed_precision.py`）
- `data/`: 数据目录（预留）
- `models/`: 最终模型参数
- `checkpoints/`: 训练中间 checkpoint
- `output/`: 指标、曲线图、配置快照

运行：
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/mixed_precision
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune

# 自动模式（按设备选择 fp16/bf16/no_amp）
python code/mixed_precision.py

# 手动模式
python code/mixed_precision.py --amp-mode fp16
python code/mixed_precision.py --amp-mode bf16
python code/mixed_precision.py --amp-mode no_amp
```

说明：
- 脚本会自动记录最终生效的混合精度配置（`output/mixed_precision_resolved_amp.json`）。
- 输出目录 `output/mixed_precision_metrics` 包含：
  - `training_metrics.csv`
  - `training_log.json`
  - `training_curves.png`
  - `summary.json`
