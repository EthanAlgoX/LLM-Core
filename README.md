# Finetune 学习项目（LLM / VLM / RLHF）

这个项目用于快速学习并复现 LLM、VLM 与后训练核心流程，面向面试准备场景。

## 新手学习入口（推荐）

从这里开始：

- [learning/README.md](./learning/README.md)
- [learning/ROADMAP_14_DAYS.md](./learning/ROADMAP_14_DAYS.md)
- [learning/LEVEL_CHECKPOINTS.md](./learning/LEVEL_CHECKPOINTS.md)

每次跑完模块后生成口述稿：

```bash
python scripts/interview_brief.py --module sft
python scripts/interview_brief.py --module mdp

# 批量导出所有模块口述稿
python scripts/export_interview_briefs.py

# 模拟面试环节 (针对选定模块提问)
python scripts/interview_qa.py --module ppo
```

## 🧠 面试备考速记表 (Interview Cheat Sheet)

### 1. 显存计算公式 (Memory Calculation)

- **模型权重**：$Params \times Bytes$ (fp16 为 2B 每一参数)。
- **KV Cache** (针对每个 Token)：$2 \times \text{layers} \times \text{heads} \times \text{dim} \times \text{precision}$。
- **训练梯度与优化器**：
  - **Adam (fp32)**：模型权重的 ~12~16 倍 (4B 梯度 + 8B 优化器状态 + 4B 权重副本)。
  - **LoRA**：仅占模型权重的 ~1~5%。

### 2. 核心算法对比矩阵

| 特性 | SFT | PPO | DPO | GRPO |
| :--- | :--- | :--- | :--- | :--- |
| **基础要求** | 监督数据 (Q/A) | 偏好数据 + 奖励模型 | 偏好对 (Chosen/Rejected) | 偏好数据 + 分数奖励 |
| **显存压力** | 低 | **极高** (4个模型同时在显存) | 中 | 中 (省去 Critic) |
| **收敛难度** | 容易 (梯度下降) | 难 (强化学习抖动) | 较容易 | 较容易 |
| **核心场景** | 初始化、习得格式 | 逻辑推理、安全边界 | 离线偏好学习 | **大规模在线强化学习** |

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
| **对齐训练 (Alignment)** | PPO | [PPO README](./post_train/alignment/ppo/README.md) |
| | GRPO | [GRPO README](./post_train/alignment/grpo/README.md) |
| | SFT | [SFT README](./post_train/alignment/sft/README.md) |
| | DPO | [DPO README](./post_train/alignment/dpo/README.md) |
| | RLHF | [RLHF README](./post_train/alignment/rlhf/README.md) |
| | PEFT | [PEFT README](./post_train/alignment/peft/README.md) |
| | Actor-Critic | [Actor-Critic README](./post_train/alignment/actor_critic/README.md) |
| | Policy Gradient | [Policy Gradient README](./post_train/alignment/policy_gradient/README.md) |
| **强化学习基础 (RL Basics)** | MDP | [MDP README](./post_train/rl_basics/mdp/README.md) |
| | TD Learning | [TD Learning README](./post_train/rl_basics/td_learning/README.md) |
| | GAE | [GAE README](./post_train/rl_basics/gae/README.md) |
| | Advantage | [Advantage README](./post_train/rl_basics/advantage/README.md) |
| **离线强化学习 (Offline RL)** | BCQ | [BCQ README](./post_train/offline_rl/bcq/README.md) |
| | CQL | [CQL README](./post_train/offline_rl/cql/README.md) |
| **多模态模型 (Multimodal)** | LLaVA | [LLaVA README](./pre_train/vlm/llava/README.md) |
| | BLIP-2 | [BLIP-2 README](./pre_train/vlm/blip2/README.md) |
| | Flamingo | [Flamingo README](./pre_train/vlm/flamingo/README.md) |
| **生成模型 (Generative)** | Diffusion | [Diffusion README](./pre_train/generation/diffusion/README.md) |
| | DiT | [DiT README](./pre_train/generation/dit/README.md) |
| **系统与工程 (Systems)** | Megatron-LM | [Megatron-LM README](./pre_train/llm/megatron/README.md) |
| | Attention | [Attention README](./pre_train/llm/attention.md) |
| | DeepSpeed | [DeepSpeed README](./post_train/systems/deepspeed/README.md) |
| | CUDA | [CUDA README](./post_train/systems/cuda/README.md) |
| | Mixed Precision | [Mixed Precision README](./post_train/systems/mixed_precision/README.md) |
| | Inference | [Inference README](./post_train/systems/inference/README.md) |

- `assets/`: 示例数据与历史实验产物归档

## 模块内标准目录含义

- `code/`: 主流程代码，直接运行即可看到训练/推理过程。
- `data/`: 样本数据、数据索引与配置。
- `models/`: 训练完成后的最终模型文件（用于推理和部署）。
- `checkpoints/`: 训练过程中的中间状态（用于断点续训和回溯）。
- `output/`: 可视化图、指标表、日志与总结（常见为 `csv/png/json`）。

详细原理、区别、运行与产物说明见各子目录 README。
