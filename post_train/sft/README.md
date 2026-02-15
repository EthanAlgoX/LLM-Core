# SFT

目录结构：
- `code/`: 训练代码与依赖（`code/sft.py`）
- `LLaMA-Factory/`: 训练框架源码（第三方目录）
- `data/`: 数据目录（自定义数据可放这里）
- `models/`: 最终导出的模型文件
- `checkpoints/`: 训练过程 checkpoint
- `output/`: 指标、曲线图、日志与配置快照

运行：
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/sft
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/sft.py
```
