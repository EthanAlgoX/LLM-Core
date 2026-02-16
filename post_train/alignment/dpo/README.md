# DPO（直接偏好优化）

## 定位与分类
- 阶段：后训练（偏好对齐）
- 类型：偏好学习（无需显式奖励模型）
- 作用：让模型偏向 `chosen` 回答，抑制 `rejected` 回答

## 核心原理
1. 输入样本为 `prompt + chosen + rejected`。
2. 直接优化偏好目标，避免“奖励模型 + PPO”两阶段复杂度。
3. `pref_beta` 控制偏好强度，越大更新通常越激进。

## 与相近方法区别
1. 相比 `SFT`：DPO 学“相对偏好”，而不是“绝对标准答案”。
2. 相比 `PPO/RLHF`：DPO 不需要在线 rollouts，工程更简洁。
3. 相比 `GRPO`：DPO 常基于成对偏好数据，GRPO 常基于组内多采样奖励比较。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/alignment/dpo
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/dpo.py
```

## 输出结果
默认输出到 `output/dpo_metrics`，包含：
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
