# DPO（直接偏好优化）

## 定位与分类

- 阶段：后训练（偏好对齐）
- 类型：偏好学习（无需显式奖励模型）
- 作用：让模型偏向 `chosen` 回答，抑制 `rejected` 回答

## 核心原理与关键公式

### 1. 关键公式：DPO 损失函数

DPO 的伟大之处在于它证明了可以直接利用偏好数据优化策略，而不需要训练显式的奖励模型。其目标函数为：

$$L_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{ref}(y_l | x)} \right) \right]$$

**公式拆解与理解：**

- **$\pi_\theta$ 与 $\pi_{ref}$**：当前优化的模型与冻结的参考模型（通常是 SFT 后的模型）。
- **$y_w$ (Chosen) 与 $y_l$ (Rejected)**：偏好对中的“好答案”与“坏答案”。
- **$\log \frac{\pi_\theta}{\pi_{ref}}$ (Log-Ratio)**：衡量当前模型相对于参考模型，对某个回答概率的“提升程度”。
- **偏好边际 (Preference Margin)**：括号内的两项相减，代表了模型对“好答案”的提升程度是否远大于对“坏答案”的提升程度。
- **$\beta$ (Beta 系数)**：调节因子。控制对偏好的敏感度，同时也起到了类似 PPO 中 KL 散度的约束作用，防止模型跑得太偏。

### 2. 深度解读：为什么它能取代奖励模型？

- **隐含奖励 (Implicit Reward)**：DPO 发现，一个最优策略 $\pi$ 与奖励函数之间存在一一映射关系。
- **直接对比**：DPO 不去问“这个回答得多少分”，而是问“当前的这个模型，是不是比原始模型更喜欢好回答、更讨厌坏回答”。
- **工程简化**：省去了训练奖励模型、在线采样、以及复杂的 Critic 调优过程。

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
