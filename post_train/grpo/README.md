# GRPO Demo

该目录提供一个最小可运行的 GRPO 样例，基于 `trl` 的 `GRPOTrainer`。

## 文件说明

- `grpo_demo.py`: GRPO 训练样例（合成数学数据 + 多 reward 函数 + 指标导出）
- `requirements.txt`: 运行依赖
- `run_demo.sh`: 一键安装依赖并启动样例（支持透传参数）

## 使用方式

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/grpo
bash run_demo.sh
```

或手动执行：

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/grpo
python -m pip install -r requirements.txt
python grpo_demo.py
```

传参示例（更大数据集）：

```bash
bash run_demo.sh --train-size 32 --num-train-epochs 1
```

传参示例（快速 smoke）：

```bash
bash run_demo.sh --train-size 8 --num-train-epochs 0.3 --output-dir qwen3_grpo_out_smoke
```

## 说明

- 默认模型：`Qwen/Qwen3-0.6B`
- 默认输出目录：`qwen3_grpo_out`
- 默认训练数据：24 条合成算术题（可通过 `--train-size` 调整）
- 奖励函数：`correctness_reward` + `format_reward` + `compact_output_reward`
- 脚本对 `trl` 参数做了兼容处理，可适配不同版本的 `GRPOTrainer` 常见签名差异

## 训练结果数据

训练完成后，会在以下目录生成模型与指标文件：

- `/Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/grpo/qwen3_grpo_out`
- `/Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/grpo/qwen3_grpo_out/metrics`

`metrics` 目录中包含：

- `log_history.json`: TRL 原始日志（完整键值）
- `train_summary.json`: 训练汇总指标（如 `train_loss`、`train_runtime`）
- `training_metrics.csv`: 可用于表格/二次分析的结构化指标
- `training_curves.png`: 训练曲线图（loss、reward、reward components、learning rate）
