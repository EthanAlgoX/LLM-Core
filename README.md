# Finetune 学习项目（LLM / VLM / RLHF）

这个项目用于快速学习并复现 LLM、VLM 与后训练核心流程，面向面试准备场景。

## 一键入口（面试模式）
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune

# 查看所有模块
python run.py --list

# 运行某个模块（默认参数）
python run.py --module mdp

# 运行某个模块（toy 参数，快速出结果）
python run.py --module sft --toy
python run.py --module grpo --toy
```

## 自动回归（避免改坏）
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune

# 全量 smoke（包含 help + toy）
python scripts/smoke_test.py

# 只测指定模块
python scripts/smoke_test.py --modules sft,grpo,mdp
```

测试报告输出到：
- `output/smoke_reports/*.json`

## 目录说明
- `pre_train/`: 预训练与多模态建模（nanoGPT、diffusion、dit、blip2、llava、flamingo、megatron）
- `post_train/`: 后训练与强化学习（sft、dpo、grpo、ppo、rlhf、mdp、td_learning、gae、advantage、cql、bcq 等）

详细原理、区别、运行与产物说明见各子目录 README。
