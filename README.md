# Finetune 学习项目（LLM / VLM / RLHF）

这个项目用于快速学习并复现 LLM、VLM 与后训练核心流程，面向面试准备场景。

## 新手学习入口（推荐）
从这里开始：
- `/Users/yunxuanhan/Documents/workspace/ai/Finetune/learning/README.md`
- `/Users/yunxuanhan/Documents/workspace/ai/Finetune/learning/ROADMAP_14_DAYS.md`
- `/Users/yunxuanhan/Documents/workspace/ai/Finetune/learning/LEVEL_CHECKPOINTS.md`

每次跑完模块后生成口述稿：
```bash
python scripts/interview_brief.py --module sft
python scripts/interview_brief.py --module mdp
```

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
- `pre_train/llm/`: 语言模型预训练（`nanoGPT`、`megatron`）
- `pre_train/generation/`: 生成模型（`diffusion`、`dit`）
- `pre_train/vlm/`: 多模态模型（`blip2`、`llava`、`flamingo`）
- `post_train/alignment/`: 对齐训练（`sft`、`dpo`、`grpo`、`ppo`、`policy_gradient`、`actor_critic`、`rlhf`）
- `post_train/rl_basics/`: RL 基础（`mdp`、`td_learning`、`gae`、`advantage`）
- `post_train/offline_rl/`: 离线 RL（`cql`、`bcq`）
- `post_train/systems/`: 工程优化（`deepspeed`、`cuda`、`mixed_precision`）
- `assets/`: 示例数据与历史实验产物归档

详细原理、区别、运行与产物说明见各子目录 README。
