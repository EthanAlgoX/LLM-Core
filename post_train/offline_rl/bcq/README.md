# BCQ（Batch-Constrained Q-learning）

## 定位与分类
- 阶段：离线强化学习
- 类型：行为约束策略学习
- 作用：限制策略偏离离线数据分布，降低离线外推误差

## 核心原理
1. 学习行为近似模型，近似“数据中可能动作”。
2. 只在候选动作集合中选择高 Q 动作。
3. 减少对数据外动作的过度利用。

## 与相近方法区别
1. 相比 `CQL`：BCQ 主打动作约束；CQL 主打 Q 函数保守惩罚。
2. 相比在线算法：BCQ 不需要在线采样，适合仅有历史数据的场景。
3. 相比 `DQN/Q-learning`：BCQ 专门处理离线分布偏移问题。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/offline_rl/bcq
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/bcq.py
```

## 输出结果
默认输出到 `output/bcq_metrics`，包含：
- `training_metrics.csv`
- `training_curves.png`
- `policy.json`
- `qmax_by_state.json`
- `summary.json`
