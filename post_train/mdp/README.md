# MDP（马尔可夫决策过程）

## 定位与分类
- 阶段：RL 基础理论
- 类型：模型已知的规划问题
- 作用：建立状态、动作、转移、奖励的统一数学框架

## 核心原理
1. 定义五元组 `(S, A, P, R, γ)`。
2. 通过 Bellman 最优方程进行值迭代/策略迭代。
3. 在已知环境模型下可直接规划最优策略。

## 与相近方法区别
1. 相比 `TD Learning`：MDP 示例偏“有模型规划”，TD 偏“无模型学习”。
2. 相比 `Policy Gradient`：MDP 是问题定义，不限定具体求解算法。
3. 相比 `CQL/BCQ`：后者是离线 RL 具体算法，不是基础框架定义。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/mdp
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/mdp.py
```

## 输出结果
默认输出到 `output/mdp_metrics`，包含：
- `iteration_log.csv`
- `training_curves.png`
- `value_function.json`
- `policy.json`
- `summary.json`
