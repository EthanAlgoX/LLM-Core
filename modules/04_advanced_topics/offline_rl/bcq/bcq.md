# BCQ（Batch-Constrained Q-learning）

## 定位与分类

- **阶段**：离线强化学习（Offline RL）。
- **类型**：动作空间约束策略。
- **作用**：通过限制模型只在“见过”的动作分布中挑选高分动作，解决离线学习中因分布偏移（Distribution Shift）导致的 Q 值虚高问题。

## 什么是 BCQ？

BCQ（Batch-Constrained deep Q-learning）是解决离线强化学习中“外推误差（Extrapolation Error）”的奠基之作。
在离线场景下，如果模型尝试执行一个数据集中从未见过的动作，Q 网络往往会给出一个不准确的高分。BCQ 的核心哲学是：**“如果我不知道这个动作好不好（因为没见过），那我就坚决不去试。我只在确信的范围内挑选最好的。”**

## 训练的关键步骤

1. **训练行为模型 (Imitation Model)**：使用监督学习学习数据集中历史动作的分布 $P(a|s)$。在连续版本中通常使用 VAE，离散版本则直接线性拟合。
2. **Q 网络更新 (Constrained Q-Update)**：
   - 采样 $(s, a, r, s')$。
   - 在计算下一时刻 $s'$ 的目标 Q 值时，仅考虑那些在行为模型中出现概率大于阈值的动作。
3. **阈值过滤 (Action Masking)**：设置阈值 $\tau$，过滤掉低概率动作，确保策略更新受限于历史“批次（Batch）”。
4. **策略推理**：最优动作 $a = \arg\max_{a: P(a|s) > \tau} Q(s, a)$。

## 核心数学公式

### 1. 约束优化目标

BCQ 的核心在于如何计算下一步的标靶值（Target）：

$$Q(s, a) \leftarrow r + \gamma \max_{a' \mathrm{ s.t. } \frac{\pi_b(a'|s')}{\max_{\hat{a}} \pi_b(\hat{a}|s')} > \tau} Q_{target}(s', a')$$

- ** $\pi_b(a'|s')$ **：行为模型预测的动作概率。
- ** $\tau$ (Threshold)**：约束强度。 $\tau=0$ 退化为普通 Q-learning， $\tau=1$ 则完全变成行为克隆（BC）。
- **逻辑**：只有当一个动作相对于该状态下“最可能动作”的比例超过 $\tau$ 时，才会被纳入 Q 值的最大化搜索范围。

### 2. 离线误差抑制

通过限制 $\max$ 操作的搜索空间，BCQ 强制让模型在数据集支持度（Support）高的范围内进行改进，从根本上抑制了外推带来的不稳定性。

## 与相近方法区别

1. 相比 `CQL`：BCQ 主打动作约束；CQL 主打 Q 函数保守惩罚。
2. 相比在线算法：BCQ 不需要在线采样，适合仅有历史数据的场景。
3. 相比 `DQN/Q-learning`：BCQ 专门处理离线分布偏移问题。

## 运行

```bash
cd <YOUR_PROJECT_ROOT>/post_train/offline_rl/bcq

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

## 目录文件说明（重点）

- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
