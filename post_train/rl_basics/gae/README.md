# GAE（Generalized Advantage Estimation）

## 定位与分类

- **阶段**：策略优化的核心组件。
- **类型**：高级优势估计技术。
- **作用**：它是 PPO 等现代强化学习算法的标配。通过指数加权平均多步残差，它在保证方向正确（低偏差）的同时，大幅降低了训练的波动性（低方差）。

## 什么是 GAE？

GAE（Generalized Advantage Estimation）是一种通过“折衷”来提升训练效率的方法。
如果只看一步奖励（TD），虽然稳定但可能“短视”；如果看完整条序列（MC），虽然长远但波动异常剧烈。GAE 引入了一个因子 $\lambda$，通过对所有可能的步长进行加权平滑，找到了一个完美的平衡点。

## 计算的关键步骤

1. **TD 残差计算 (Compute Temporal Difference)**：
   - 首先计算每一步的即时差异：$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。
2. **指数加权累计 (Exponential Weighting)**：
   - 使用 $\lambda$ 对未来的残差进行加权求和。
3. **优势赋值**：
   - 对于每个时间步 $t$，根据未来所有步的残差合成出一个优势值 $A_t$。
4. **损失计算与反向传播**：
   - 将估算出的 $A_t$ 代入策略梯度公式进行更新。

## 核心数学公式

### 1. TD 残差 (TD Error)

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

### 2. GAE 估计量

$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}$$

展开形式：
$$\hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \dots$$

- **$\gamma$ (Gamma)**：折扣因子，决定了对未来奖励的重视程度。
- **$\lambda$ (Lambda)**：GAE 特有因子。$\lambda=0$ 时退化为 1-step TD；$\lambda=1$ 时退化为 Monte Carlo。

## 与相近方法区别

1. 相比 `TD(0)`：GAE 融合多步信息，估计更平滑。
2. 相比 `Advantage` 模块：`advantage` 是多方法对比，`gae` 专注 GAE 训练过程。
3. 相比 `Actor-Critic`：GAE 常作为 Actor-Critic 的优势估计组件。

## 运行

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/rl_basics/gae
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/gae.py
```

## 输出结果

默认输出到 `output/gae_metrics`，包含：

- `training_metrics.csv`
- `training_curves.png`
- `summary.json`
- `training_log.json`

## 目录文件说明（重点）

- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
