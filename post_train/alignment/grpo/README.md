# GRPO（Group Relative Policy Optimization）

## 定位与分类
- 阶段：后训练（策略优化）
- 类型：强化学习风格偏好优化
- 作用：在同一 prompt 的多候选回答中进行相对比较，提升高奖励行为概率

## 核心原理
1. 每个 prompt 采样多条回答（`num_generations`）。
2. 多奖励函数打分（正确性、格式、简洁性等）。
3. 组内标准化奖励（`scale_rewards=group`）降低方差。
4. 用相对优势更新策略，而非仅依赖单条样本分数。

## 与相近方法区别
1. 相比 `PPO`：GRPO 更强调组内相对比较，弱化绝对值尺度问题。
2. 相比 `DPO`：GRPO 不局限于成对偏好，可直接利用多候选采样。
3. 相比 `SFT`：GRPO 使用奖励驱动优化，不是纯监督拟合。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/alignment/grpo
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/grpo_demo.py
```

## 输出结果
默认输出到 `output/grpo_metrics`，包含：
- `training_metrics.csv`
- `training_curves.png`
- `summary.json`
- `log_history.json`
- `train_summary.json`


## 目录文件说明（重点）
- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
