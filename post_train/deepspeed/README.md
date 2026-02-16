# DeepSpeed

目录结构：
- `code/`: 代码（`code/deepspeed.py`）
- `data/`: 数据目录（预留）
- `models/`: 最终模型参数
- `checkpoints/`: 训练中间 checkpoint
- `output/`: 指标、曲线图、配置快照

运行：
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/deepspeed
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune

# 默认可运行（若无 deepspeed 自动回退 Torch）
python code/deepspeed.py

# 尝试 DeepSpeed 引擎
python code/deepspeed.py --use-deepspeed --zero-stage 2
```

说明：
- 输出目录 `output/deepspeed_metrics` 包含：
  - `training_metrics.csv`
  - `training_log.json`
  - `training_curves.png`
  - `summary.json`
- 同时会生成 `output/deepspeed_config_auto.json` 作为 DeepSpeed 配置参考。
