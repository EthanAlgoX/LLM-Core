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
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/flamingo
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/flamingo.py --dry-run
```

## 输出结果
默认输出到 `output/flamingo_metrics`，包含：
- `result.json`
- `preview.png`（若安装 matplotlib）
