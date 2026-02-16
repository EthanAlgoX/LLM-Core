# Flamingo

## 定位与分类
- 阶段：多模态预训练/推理
- 类型：跨注意力注入式 VLM
- 作用：在语言模型层间周期性注入视觉条件实现图文联合生成

## 核心原理
1. 图像由视觉编码器编码。
2. 在语言模型中插入跨注意力层读取视觉信息。
3. 通过 interleaved multimodal 序列完成生成。

## 与相近方法区别
1. 相比 `LLaVA`：Flamingo 的视觉信息注入更深层、更持续。
2. 相比 `BLIP2`：Flamingo 不是单桥接 Q-Former 路径。
3. 相比纯视觉模型：Flamingo 保留强语言生成能力。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/vlm/flamingo
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/flamingo.py --dry-run
```

## 输出结果
默认输出到 `output/flamingo_metrics`，包含：
- `result.json`
- `preview.png`（若安装 matplotlib）


## 目录文件说明（重点）
- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
