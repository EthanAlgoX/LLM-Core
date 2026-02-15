# Policy Gradient

目录结构：
- `code/`: 训练代码（`code/policy_gradient.py`）
- `data/`: 数据目录（自定义 prompt 数据可放这里）
- `models/`: 最终导出的模型文件
- `checkpoints/`: 训练过程 checkpoint
- `output/`: 指标、曲线图、日志与配置快照

运行：
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/policy_gradient
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/policy_gradient.py --reward-model <奖励模型路径或名称>
```

说明：
- 本实现使用 LLaMA-Factory 的 `stage=ppo` 作为 Policy Gradient 的稳定近似实现。
- `--reward-model` 为必填参数。
- 默认会复用 `post_train/sft/LLaMA-Factory` 作为训练框架源码。
