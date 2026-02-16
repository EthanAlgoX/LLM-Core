# CQL

目录结构：
- `code/`: 代码（`code/cql.py`）
- `data/`: 离线数据集与统计信息
- `models/`: 最终 Q 网络参数
- `checkpoints/`: 训练中间 checkpoint
- `output/`: 指标、曲线图、配置快照

运行：
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/cql
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/cql.py
```

说明：
- 这是 CQL 的教学示例：先构建离线数据，再离线训练 Q 网络。
- 输出目录 `output/cql_metrics` 包含：
  - `training_metrics.csv`
  - `training_log.json`
  - `training_curves.png`
  - `policy.json`
  - `summary.json`
