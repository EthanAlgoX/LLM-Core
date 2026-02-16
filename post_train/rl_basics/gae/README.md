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


## 目录文件说明（重点）
- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
