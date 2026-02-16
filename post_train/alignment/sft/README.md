# SFT（监督微调）

## 定位与分类
- 阶段：后训练（对齐起点）
- 类型：监督学习（有标准答案）
- 作用：让模型先学会“按指令说话”的基本行为

## 核心原理
1. 使用 `prompt-response` 样本做 token-level 交叉熵训练。
2. 目标是最大化参考答案的条件概率。
3. 常用 LoRA 降低显存与训练成本。
4. 通过 `loss / eval_loss / lr / grad_norm` 观察收敛与稳定性。

## 与相近方法区别
1. 相比 `DPO`：SFT 不利用偏好对（chosen/rejected），只学习“标准答案”。
2. 相比 `PPO/GRPO`：SFT 不需要奖励模型或在线采样，训练更稳定、成本更低。
3. 相比 `RLHF`：SFT 是 RLHF 流程中的第一步，不是完整闭环。

## 目录结构
- `code/sft.py`: 一键训练 + 可视化导出
- `LLaMA-Factory/`: 训练框架源码
- `data/`, `models/`, `checkpoints/`, `output/`: 统一目录

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/alignment/sft
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/sft.py
```

## 输出结果
默认输出到 `output/sft_metrics`，包含：
- `training_metrics.csv`
- `training_curves.png`
- `summary.json`
- `log_history.json`


## 目录文件说明（重点）
- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
