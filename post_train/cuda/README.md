# CUDA

目录结构：
- `code/`: 代码（`code/cuda.py`）
- `data/`: 数据目录（预留）
- `models/`: toy 模型参数
- `checkpoints/`: 中间结果目录（预留）
- `output/`: 指标、曲线图、配置快照

运行：
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/cuda
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/cuda.py
```

说明：
- 脚本会自动检测 CUDA 环境；即使无 CUDA，也会在可用设备上运行演示。
- 输出目录 `output/cuda_metrics` 包含：
  - `cuda_info.json`
  - `benchmark.csv/json`
  - `training_metrics.csv`
  - `training_curves.png`
  - `summary.json`
