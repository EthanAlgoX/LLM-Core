# LLaVA

目录结构：
- `code/`: 推理代码（`code/llava.py`）
- `data/`: 数据目录（预留）
- `models/`: 模型缓存目录（可选）
- `checkpoints/`: 训练/微调中间结果目录（预留）
- `output/`: 结果输出目录（JSON 与可视化）

运行：
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/llava
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune

# 干跑测试（不加载模型）
python code/llava.py --dry-run

# 实际推理
python code/llava.py --image /absolute/path/to/image.jpg --task vqa --question "图里有什么？"
python code/llava.py --image /absolute/path/to/image.jpg --task caption
```

说明：
- 推理结果默认写入 `output/llava_metrics/result.json`。
- 若安装了 `matplotlib`，会额外输出 `output/llava_metrics/preview.png`。
