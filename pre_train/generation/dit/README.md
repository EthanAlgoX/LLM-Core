# DiT（Diffusion Transformer）

## 定位与分类
- 阶段：预训练/生成建模
- 类型：Transformer 架构扩散模型
- 作用：学习如何用 Transformer 替代 U-Net 做扩散去噪

## 核心原理
1. 将图像切为 patch token。
2. 注入时间步嵌入并做 Transformer 编码。
3. 预测噪声并按扩散反演过程采样。

## 与相近方法区别
1. 相比 `Diffusion` 基础实现：DiT 更强调 token 化与全局注意力。
2. 相比 CNN-U-Net：DiT 通常更易扩展到大模型规模。
3. 相比 LLM：DiT 处理图像/latent token，不是自然语言 token。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/generation/dit
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/dit.py
```

## 输出结果
默认输出到 `output/dit_metrics`，包含：
- `training_metrics.csv`
- `training_curves.png`
- `generated_samples.pt`
- `target_samples.pt`
- `summary.json`


## 目录文件说明（重点）
- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
