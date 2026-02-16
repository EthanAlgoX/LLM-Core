# GAE（Generalized Advantage Estimation）

## 定位与分类
- 阶段：RL 基础到策略优化的桥梁
- 类型：优势函数估计方法
- 作用：在偏差与方差之间做可控折中，稳定策略更新

## 核心原理
1. 使用 `lambda-return` 融合多步 TD 误差。
2. `λ` 越大越接近 Monte Carlo（低偏差高方差）。
3. `λ` 越小越接近一步 TD（高偏差低方差）。

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
