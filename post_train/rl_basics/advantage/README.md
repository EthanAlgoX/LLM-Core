# Advantage（优势函数学习与对比）

## 定位与分类
- 阶段：RL 基础专题
- 类型：优势估计方法对比实验
- 作用：帮助理解 MC、TD、GAE 三类优势估计的差异

## 核心原理
1. 优势定义：`A(s,a) = Q(s,a) - V(s)`。
2. 优势用于降低策略梯度方差并保留相对好坏信号。
3. 不同估计方式会影响训练稳定性和样本效率。

## 与相近方法区别
1. 相比 `GAE`：本模块覆盖多种方法并给出横向可视化。
2. 相比 `Policy Gradient`：本模块关注“估计器”，PG关注“更新目标”。
3. 相比 `TD Learning`：TD 更偏价值函数更新，本模块偏策略优化输入信号分析。

## 运行
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/rl_basics/advantage
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/advantage.py
```

## 输出结果
默认输出到 `output/advantage_metrics`，包含：
- `training_metrics.csv`
- `training_curves.png`
- `advantage_methods.png`
- `advantage_comparison.json`
- `summary.json`


## 目录文件说明（重点）
- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
