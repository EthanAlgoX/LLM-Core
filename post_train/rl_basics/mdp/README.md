# MDP（马尔可夫决策过程）

## 定位与分类

- **阶段**：强化学习理论基石（RL Foundation）。
- **类型**：序列决策数学模型。
- **作用**：它是所有强化学习问题的共同底座。无论是在线 PPO 还是离线 BCQ，其底层逻辑都可以抽象为一个 MDP 过程。

## 什么是 MDP？

MDP（Markov Decision Process，马尔可夫决策过程）是一个描述**智能体（Agent）**与**环境（Environment）**交互的数学模型。
其核心特性是“马尔可夫性”：**未来仅取决于现在，而与过去无关**。即只要知道了当前状态，我们就拥有了决定下一步动作所需的全部信息。

## 核心组成部分（五元组）

1. **状态空间 $S$ (States)**：智能体可能处在的所有情况（如网格坐标）。
2. **动作空间 $A$ (Actions)**：智能体在每个状态下可以采取的行为（如上下左右）。
3. **转移概率 $P(s'|s, a)$ (Transitions)**：在状态 $s$ 执行动作 $a$ 后，到达 $s'$ 的概率。
4. **奖励函数 $R(s, a, s')$ (Rewards)**：环境给出的反馈得分。
5. **折扣因子 $\gamma$ (Discount)**：对未来奖励的重视程度（0到1之间）。

## 核心数学公式

### 1. 贝尔曼方程 (Bellman Equation)

这是 MDP 的灵魂，它定义了当前价值与未来价值的关系：

$$V^\pi(s) = \sum_{a \in A} \pi(a|s) \sum_{s' \in S} P(s'|s, a) [R(s, a, s') + \gamma V^\pi(s')]$$

### 2. 最优值函数 (Optimal Value Function)

通过值迭代（Value Iteration）寻找最优分数的迭代式：

$$V^*(s) = \max_{a \in A} \sum_{s' \in S} P(s'|s, a) [R(s, a, s') + \gamma V^*(s')]$$

## 与相近方法区别

1. 相比 `TD Learning`：MDP 示例偏“有模型规划”，TD 偏“无模型学习”。
2. 相比 `Policy Gradient`：MDP 是问题定义，不限定具体求解算法。
3. 相比 `CQL/BCQ`：后者是离线 RL 具体算法，不是基础框架定义。

## 运行

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/rl_basics/mdp
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

## 目录文件说明（重点）

- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
