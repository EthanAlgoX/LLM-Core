# GRPO Demo

该目录提供一个最小可运行的 GRPO 样例，基于 `trl` 的 `GRPOTrainer`。

## 文件说明

- `grpo_demo.py`: GRPO 训练样例（玩具数学数据 + 两个 reward 函数）
- `requirements.txt`: 运行依赖
- `run_demo.sh`: 一键安装依赖并启动样例

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

## 说明

- 默认模型：`Qwen/Qwen3-0.6B`
- 默认输出目录：`qwen3_grpo_out`
- 样例数据是脚本内置的 4 条算术题，用于演示 GRPO 流程而非追求效果
- 脚本对 `trl` 的参数做了兼容处理，可适配不同版本的 `GRPOTrainer` 常见签名差异
