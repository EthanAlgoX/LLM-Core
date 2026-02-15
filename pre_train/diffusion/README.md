# Diffusion

目录结构：
- `code/`: 训练代码（`code/diffusion.py`）
- `data/`: 数据目录（预留）
- `models/`: 最终导出的模型文件
- `checkpoints/`: 训练过程 checkpoint
- `output/`: 指标、曲线图、日志与采样结果

运行：
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/diffusion
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/diffusion.py
```

说明：
- 该脚本是一个可快速跑通的 2D toy diffusion 示例。
- 会在 `output/diffusion_metrics` 下生成：
  - `training_metrics.csv`
  - `training_curves.png`
  - `summary.json`
  - `generated_samples.pt` / `target_samples.pt`
