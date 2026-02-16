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
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/vlm/blip2
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/blip2.py --dry-run
```

## 输出结果
默认输出到 `output/blip2_metrics`，包含：
- `result.json`
- `preview.png`（若安装 matplotlib）


## 目录文件说明（重点）
- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
