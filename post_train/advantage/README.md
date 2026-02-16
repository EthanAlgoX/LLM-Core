# Advantage

目录结构：
- `code/`: 代码（`code/advantage.py`）
- `data/`: 数据目录（预留）
- `models/`: 最终模型参数
- `checkpoints/`: 训练中间 checkpoint
- `output/`: 指标、曲线图、配置快照

运行：
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/advantage
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/advantage.py
```

可切换优势估计方式：
```bash
python code/advantage.py --advantage-method mc
python code/advantage.py --advantage-method td
python code/advantage.py --advantage-method gae
```

说明：
- 该脚本用于学习优势函数估计，不同方法可直接对比。
- 输出目录 `output/advantage_metrics` 包含：
  - `training_metrics.csv`
  - `training_log.json`
  - `training_curves.png`
  - `advantage_methods.png`（MC/TD/GAE 对比图）
  - `summary.json`
