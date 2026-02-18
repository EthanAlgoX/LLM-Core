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

# 批量导出所有模块口述稿
python scripts/export_interview_briefs.py
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

## 📚 核心文档索引 (Documentation Hub)

| 类别 (Category) | 模块 (Module) | 核心原理说明 (Quick Link) |
| :--- | :--- | :--- |
| **对齐训练 (Alignment)** | PPO | [PPO README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/alignment/ppo/README.md) |
| | GRPO | [GRPO README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/alignment/grpo/README.md) |
| | SFT | [SFT README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/alignment/sft/README.md) |
| | DPO | [DPO README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/alignment/dpo/README.md) |
| | RLHF | [RLHF README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/alignment/rlhf/README.md) |
| | Actor-Critic | [Actor-Critic README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/alignment/actor_critic/README.md) |
| | Policy Gradient | [Policy Gradient README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/alignment/policy_gradient/README.md) |
| **强化学习基础 (RL Basics)** | MDP | [MDP README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/rl_basics/mdp/README.md) |
| | TD Learning | [TD Learning README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/rl_basics/td_learning/README.md) |
| | GAE | [GAE README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/rl_basics/gae/README.md) |
| | Advantage | [Advantage README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/rl_basics/advantage/README.md) |
| **离线强化学习 (Offline RL)** | BCQ | [BCQ README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/offline_rl/bcq/README.md) |
| | CQL | [CQL README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/offline_rl/cql/README.md) |
| **多模态模型 (Multimodal)** | LLaVA | [LLaVA README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/vlm/llava/README.md) |
| | BLIP-2 | [BLIP-2 README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/vlm/blip2/README.md) |
| | Flamingo | [Flamingo README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/vlm/flamingo/README.md) |
| **生成模型 (Generative)** | Diffusion | [Diffusion README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/generation/diffusion/README.md) |
| | DiT | [DiT README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/generation/dit/README.md) |
| **系统与工程 (Systems)** | Megatron-LM | [Megatron-LM README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/llm/megatron/README.md) |
| | DeepSpeed | [DeepSpeed README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/systems/deepspeed/README.md) |
| | CUDA | [CUDA README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/systems/cuda/README.md) |
| | Mixed Precision | [Mixed Precision README](file:///Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/systems/mixed_precision/README.md) |

- `assets/`: 示例数据与历史实验产物归档

## 模块内标准目录含义

- `code/`: 主流程代码，直接运行即可看到训练/推理过程。
- `data/`: 样本数据、数据索引与配置。
- `models/`: 训练完成后的最终模型文件（用于推理和部署）。
- `checkpoints/`: 训练过程中的中间状态（用于断点续训和回溯）。
- `output/`: 可视化图、指标表、日志与总结（常见为 `csv/png/json`）。

详细原理、区别、运行与产物说明见各子目录 README。
