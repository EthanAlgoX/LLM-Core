# CQL（Conservative Q-Learning）

## 定位与分类
- 阶段：离线强化学习
- 类型：保守值函数学习
- 作用：在离线数据上抑制 OOD 动作的过高 Q 值估计

## 核心原理
1. 在 Bellman 误差之外加入 conservative regularization。
2. 降低策略对数据分布外动作的乐观估计。
3. 提升离线部署时的稳健性。

## 与相近方法区别
1. 相比 `BCQ`：CQL 通过值函数约束保守化；BCQ 通过行为策略约束动作空间。
2. 相比在线 `PPO`：CQL 只使用静态数据，不与环境实时交互。
3. 相比 `TD Learning`：CQL 面向离线分布偏移问题，TD 常用于在线学习。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/cql
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/cql.py
```

## 输出结果
默认输出到 `output/cql_metrics`，包含：
- `training_metrics.csv`
- `training_curves.png`
- `policy.json`
- `qmax_by_state.json`
- `summary.json`
