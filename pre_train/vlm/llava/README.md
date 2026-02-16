# LLaVA

## 定位与分类
- 阶段：多模态预训练/推理
- 类型：视觉指令微调 VLM
- 作用：让语言模型在图像条件下执行问答与描述任务

## 核心原理
1. 图像编码后通过投影层接入 LLM token 空间。
2. 使用视觉指令数据进行对话式训练。
3. 支持 VQA、图像描述等任务。

## 与相近方法区别
1. 相比 `BLIP2`：LLaVA 更强调视觉指令微调数据驱动。
2. 相比 `Flamingo`：LLaVA 常见实现更轻量，Flamingo偏跨注意力深度融合。
3. 相比纯文本 SFT：LLaVA 输入包含图像模态。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/vlm/llava
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/llava.py --dry-run
```

## 输出结果
默认输出到 `output/llava_metrics`，包含：
- `result.json`
- `preview.png`（若安装 matplotlib）


## 目录文件说明（重点）
- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
