# DPO

目录结构：
- `code/`: 训练代码（`code/dpo.py`）
- `data/`: 数据目录（自定义偏好数据可放这里）
- `models/`: 最终导出的模型文件
- `checkpoints/`: 训练过程 checkpoint
- `output/`: 指标、曲线图、日志与配置快照

运行：
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/dpo
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/dpo.py
```
