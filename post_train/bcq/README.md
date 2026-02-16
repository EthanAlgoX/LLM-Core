# BCQ

目录结构：
- `code/`: 代码（`code/bcq.py`）
- `data/`: 离线数据集与统计信息
- `models/`: 最终 Q 网络与行为模型参数
- `checkpoints/`: 训练中间 checkpoint
- `output/`: 指标、曲线图、配置快照

运行：
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/bcq
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/bcq.py
```

说明：
- 这是离散版 BCQ 的教学示例：行为模型约束动作集合，Q 网络在约束集合内选动作。
- 输出目录 `output/bcq_metrics` 包含：
  - `training_metrics.csv`
  - `training_log.json`
  - `training_curves.png`
  - `policy.json`
  - `summary.json`
