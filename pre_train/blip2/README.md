# BLIP2

## 定位与分类
- 阶段：多模态预训练/推理
- 类型：VLM（视觉编码器 + Q-Former + LLM）
- 作用：学习轻量桥接模块将视觉特征对齐到语言模型

## 核心原理
1. 视觉编码器提取图像特征。
2. Q-Former 压缩视觉信息为可被 LLM 使用的查询表示。
3. 在冻结大部分组件情况下实现图文生成。

## 与相近方法区别
1. 相比 `LLaVA`：BLIP2 强调 Q-Former 桥接；LLaVA 常用 projector + 指令微调。
2. 相比 `Flamingo`：BLIP2 结构更偏“单桥接层”，Flamingo在语言层中插入跨注意力。
3. 相比纯 LLM：BLIP2 可处理图像输入。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/blip2
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/blip2.py --dry-run
```

## 输出结果
默认输出到 `output/blip2_metrics`，包含：
- `result.json`
- `preview.png`（若安装 matplotlib）
