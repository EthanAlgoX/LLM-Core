# PPO（Proximal Policy Optimization）

## 定位与分类
- 阶段：后训练（在线策略优化）
- 类型：强化学习（策略梯度 + KL 约束）
- 作用：在奖励模型指导下提升回答质量，并控制策略漂移

## 核心原理
1. 策略模型生成回答，奖励模型打分。
2. 使用 PPO clip 目标限制单次更新幅度。
3. 通过参考模型 KL 项抑制分布偏移。
4. 迭代更新使高奖励输出更高概率出现。

## 与相近方法区别
1. 相比 `DPO`：PPO 需要在线采样与奖励评估，成本更高但灵活性更强。
2. 相比 `SFT`：PPO 优化目标是奖励，不是 token-level teacher forcing。
3. 相比 `GRPO`：PPO 是经典剪切策略梯度框架，GRPO更偏组内相对奖励。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/alignment/ppo
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/ppo.py --reward-model <奖励模型路径或名称>
```

## 输出结果
默认输出到 `output/ppo_metrics`，包含：
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
