# MDP

目录结构：
- `code/`: 代码（`code/mdp.py`）
- `data/`: 数据目录（预留）
- `models/`: 最终策略与价值函数
- `checkpoints/`: 迭代过程 checkpoint
- `output/`: 指标、曲线图、配置快照

运行：
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/mdp
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/mdp.py
```

说明：
- 这是 GridWorld MDP + Value Iteration 的可学习示例。
- 输出目录 `output/mdp_metrics` 包含：
  - `iteration_log.csv/json`
  - `value_function.json`
  - `policy.json`
  - `training_curves.png`（价值热力图 + 收敛曲线）
  - `summary.json`
