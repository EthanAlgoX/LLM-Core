# TD Learning

目录结构：
- `code/`: 代码（`code/td_learning.py`）
- `data/`: 数据目录（预留）
- `models/`: 最终 Q 表与策略文件
- `checkpoints/`: 训练过程 checkpoint
- `output/`: 指标、曲线图、配置快照

运行：
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/td_learning
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/td_learning.py
```

说明：
- 这是 GridWorld 上的表格型 Q-learning（TD Learning）示例。
- 输出目录 `output/td_learning_metrics` 包含：
  - `episode_log.csv/json`
  - `q_table.json`
  - `policy.json`
  - `training_curves.png`（奖励/TD误差/epsilon/策略热力图）
  - `summary.json`
