# GRPO Demo

该目录提供一个最小可运行的 GRPO 样例，基于 `trl` 的 `GRPOTrainer`。
目录已按统一分类整理为：`code / data / models / output / checkpoints`。

## 文件说明

- `code/grpo_demo.py`: GRPO 训练样例（合成数学数据 + 多 reward 函数 + 指标导出）
- `code/requirements.txt`: 运行依赖

## 使用方式

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/grpo
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python -m pip install -r code/requirements.txt
python code/grpo_demo.py
```

传参示例（更大数据集）：

```bash
python code/grpo_demo.py --train-size 32 --num-train-epochs 1
```

传参示例（快速 smoke）：

```bash
python code/grpo_demo.py --train-size 8 --num-train-epochs 0.3
```

## 说明

- 默认模型：`Qwen/Qwen3-0.6B`
- 训练中间文件写入：`post_train/grpo/checkpoints`
- 最终模型写入：`post_train/grpo/models`
- 指标与曲线写入：`post_train/grpo/output`
- 默认训练数据：24 条合成算术题（可通过 `--train-size` 调整）
- 奖励函数：`correctness_reward` + `distance_reward` + `format_reward` + `compact_output_reward`
- 脚本对 `trl` 参数做了兼容处理，可适配不同版本的 `GRPOTrainer` 常见签名差异

## 训练结果数据

训练完成后，会在以下目录生成模型与指标文件：

- `/Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/grpo/checkpoints`
- `/Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/grpo/models`
- `/Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/grpo/output/metrics`

`metrics` 目录中包含：

- `log_history.json`: TRL 原始日志（完整键值）
- `train_summary.json`: 训练汇总指标（如 `train_loss`、`train_runtime`）
- `summary.json`: 关键学习摘要（如 `final_loss`、`final_reward`、`best_reward`）
- `training_metrics.csv`: 可用于表格/二次分析的结构化指标
- `training_curves.png`: 训练曲线图（loss、reward、reward components、learning rate）

## 历史目录说明

历史实验目录已统一归档到：

- `/Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/grpo/output/history_runs`

如 `qwen3_grpo_out`、`qwen3_grpo_out_opt_v2`、`qwen3_grpo_out_test_loss` 等，
属于不同阶段的实验产物，作为历史对照是合理的；但对学习和日常迭代不够简洁。
建议后续统一使用当前分类目录结构。
