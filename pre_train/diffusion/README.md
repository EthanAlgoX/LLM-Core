# Diffusion（扩散模型）

## 定位与分类
- 阶段：预训练/生成建模
- 类型：噪声到数据的迭代生成
- 作用：理解“前向加噪 + 反向去噪”范式

## 核心原理
1. 前向过程逐步加噪把数据变成近似高斯噪声。
2. 训练网络预测噪声或重建信号。
3. 推理时从噪声出发逐步反演生成样本。

## 与相近方法区别
1. 相比 `DiT`：Diffusion 是方法范式，DiT 是 Transformer 化实现。
2. 相比 GAN：扩散训练通常更稳定，但采样步骤更多。
3. 相比自回归：扩散更偏连续去噪链式生成。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/diffusion
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/diffusion.py
```

## 输出结果
默认输出到 `output/diffusion_metrics`，包含：
- `training_metrics.csv`
- `training_curves.png`
- `generated_samples.pt`
- `target_samples.pt`
- `summary.json`
