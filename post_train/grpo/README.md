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
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/grpo
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/grpo_demo.py
```

## 输出结果
默认输出到 `output/metrics`，包含：
- `training_metrics.csv`
- `training_curves.png`
- `summary.json`
- `log_history.json`
- `train_summary.json`
