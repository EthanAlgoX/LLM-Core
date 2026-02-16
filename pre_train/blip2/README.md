# BLIP2

目录结构：
- `code/`: 推理代码（`code/blip2.py`）
- `data/`: 数据目录（预留）
- `models/`: 模型缓存目录（可选）
- `checkpoints/`: 训练/微调中间结果目录（预留）
- `output/`: 结果输出目录（JSON 与可视化）

运行：
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/blip2
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune

# 干跑测试（不加载模型）
python code/blip2.py --dry-run

# 实际推理
python code/blip2.py --image /absolute/path/to/image.jpg --task caption
python code/blip2.py --image /absolute/path/to/image.jpg --task vqa --question "图里有什么？"
```

说明：
- 推理结果默认写入 `output/blip2_metrics/result.json`。
- 若安装了 `matplotlib`，会额外输出 `output/blip2_metrics/preview.png`。
