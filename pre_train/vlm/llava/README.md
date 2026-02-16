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
