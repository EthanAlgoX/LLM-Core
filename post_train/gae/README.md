# GAE

目录结构：
- `code/`: 代码（`code/gae.py`）
- `data/`: 数据目录（预留）
- `models/`: 最终模型参数
- `checkpoints/`: 训练中间 checkpoint
- `output/`: 指标、曲线图、配置快照

运行：
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/gae
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/gae.py
```

说明：
- 这是 Actor-Critic + GAE 的轻量教学示例（LineWorld）。
- 输出目录 `output/gae_metrics` 包含：
  - `training_metrics.csv`
  - `training_log.json`
  - `training_curves.png`
  - `summary.json`
