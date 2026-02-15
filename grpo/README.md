# GRPO (Group Relative Policy Optimization) 示例

本项目提供了一个基于 `trl` 库的 GRPO 微调示例。GRPO 是目前最前沿的强化学习对齐算法之一，由 DeepSeek 团队在其 V3 和 R1 模型中首次应用。

## 什么是 GRPO？

与传统的 PPO 算法相比，GRPO **去掉了 Critic 网络**（价值判别网络），通过在同一 Prompt 下生成一组样本（Group），并利用这组样本的平均奖励值作为基准来计算相对优势（Advantage）。

**优势：**

- **节省显存**：少了 Critic 网络，显存占用降低约 50%。
- **由于简单而高效**：通过组内比对，能更有效地从逻辑奖励（如代码运行成功、答案正确）中学习。

## 文件说明

- `grpo_example.py`: 核心训练脚本，集成了奖励函数定义、组生成配置以及 Mac MPS 优化参数。
- `dummy_data.jsonl`: 用于演示的极简数据文件。

## 如何运行

1. 确保已安装编译好的 `trl` 库（建议版本 >= 0.14.0）。
2. 在 `grpo` 目录下运行：

   ```bash
   python grpo_example.py
   ```

## 注意事项

- 本示例默认使用 `Qwen/Qwen3-0.6B` 模型，非常适合在 Mac 本地进行算法流程调试。
- 奖励函数（Reward Function）是 GRPO 的核心，您可以根据需要自定义逻辑，例如检查正则表达式、计算 BLEU 得分或代码解析情况。
