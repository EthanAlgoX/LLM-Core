# LLM-Core 学习手册 (Learning Guide)

本项目是一套面向面试准备的 LLM / VLM / RLHF 实战复现教程。本手册汇总了 14 天学习路线与通关验收标准。

---

## 🚀 核心动作 (每天必做)

1. **跑模块**：`python run.py --module <name> --toy`
2. **读输出**：看 `summary.json` + `training_curves.png`
3. **生口稿**：`python scripts/interview_brief.py --module <name>`
4. **做测验**：查阅 [learning/quizzes/](./learning/quizzes/)

---

## 📅 14天学习计划 (14-Day Roadmap)

| Day | 目标 | 命令 | 必须掌握的概念 |
| --- | --- | --- | --- |
| 1 | [项目认知](./post_train/rl_basics/mdp/README.md) | `python run.py --module mdp --toy` | MDP五元组 (S, A, R, P, γ) |
| 2 | [RL 基础](./post_train/rl_basics/td_learning/README.md) | `python run.py --module td_learning --toy` | TD vs Monte Carlo |
| 3 | [优势估计](./post_train/rl_basics/advantage/README.md) | `python run.py --module advantage --toy` | Advantage 的方差缩减作用 |
| 4 | [GAE 算法](./post_train/rl_basics/gae/README.md) | `python run.py --module gae --toy` | GAE λ 的偏差/方差权衡 |
| 5 | [SFT 入门](./post_train/alignment/sft/README.md) | `python run.py --module sft --toy` | MLE 损失与交叉熵 |
| 6 | [DPO 对齐](./post_train/alignment/dpo/README.md) | `python run.py --module dpo --toy` | 隐式奖励 vs 显式奖励 |
| 7 | [PPO 算法](./post_train/alignment/ppo/README.md) | `python run.py --module ppo --toy` | 策略剪切 (Clipped Objective) |
| 8 | [GRPO 创新](./post_train/alignment/grpo/README.md) | `python run.py --module grpo --toy` | 组内相对奖励 (Group Relative) |
| 9 | [RLHF 全流程](./post_train/alignment/rlhf/README.md) | `python run.py --module rlhf --toy` | SFT -> RM -> PPO 三阶段 |
| 10 | [离线 RL (1)](./post_train/offline_rl/cql/README.md) | `python run.py --module cql --toy` | 保守值估计 (Conservative) |
| 11 | [离线 RL (2)](./post_train/offline_rl/bcq/README.md) | `python run.py --module bcq --toy` | OOD 动作过滤机制 |
| 12 | [多模态 (1)](./pre_train/vlm/blip2/README.md) | `python run.py --module blip2 --toy` | Projector vs Q-Former |
| 13 | [多模态 (2)](./pre_train/vlm/flamingo/README.md) | `python run.py --module flamingo --toy` | Gated Cross-Attention |
| 14 | [综合回顾](./pre_train/llm/megatron/README.md) | `python run.py --module megatron --toy` | 模型并行 (TP/PP/DP) |

---

## 🎯 通关验收标准 (Level Checkpoints)

### Level 1：RL 基础 ([mdp](./post_train/rl_basics/mdp/code/), [td_learning](./post_train/rl_basics/td_learning/code/))

- [ ] 能解释状态、动作、奖励。
- [ ] 能口述一张训练曲线（至少 60s）。

### Level 2：优势估计 ([advantage](./post_train/rl_basics/advantage/code/), [gae](./post_train/rl_basics/gae/code/))

- [ ] 能定义 $A(s,a) = Q(s,a) - V(s)$。
- [ ] 能说明为什么优势函数能让训练更稳。

### Level 3：对齐起步 ([sft](./post_train/alignment/sft/code/), [dpo](./post_train/alignment/dpo/code/))

- [ ] 能写出 DPO 的 Chosen/Rejected 输入格式。
- [ ] 能列举 SFT vs DPO 的三个核心区别。

### Level 4：强化学习对齐 ([ppo](./post_train/alignment/ppo/code/), [grpo](./post_train/alignment/grpo/code/), [rlhf](./post_train/alignment/rlhf/code/))

- [ ] 能说明 PPO 的四个模型角色。
- [ ] 能说明为何 GRPO 不需要 Critic 网络。

### Level 5：多模态与工程 ([vlm](./pre_train/vlm/), [megatron](./pre_train/llm/megatron/code/), [deepspeed](./post_train/systems/deepspeed/code/))

- [ ] 能比较 BLIP2 与 LLaVA 的架构差异。
- [ ] 能解释零冗余优化 (ZeRO 1/2/3)。

---

## 📚 辅助资源

- **每日题目**：[learning/quizzes/](./learning/quizzes/)
- **学霸卡模板**：[learning/cards/TEMPLATE.md](./learning/cards/TEMPLATE.md)
- **批量导出**：`python scripts/export_interview_briefs.py`
