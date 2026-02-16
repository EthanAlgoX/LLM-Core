# 14天学习路线（面试版）

统一环境：
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
```

## 每天固定动作
1. 跑模块：`python run.py --module <name> --toy`
2. 读输出：看 `summary.json` + `training_curves.png`
3. 生成口述稿：`python scripts/interview_brief.py --module <name>`
4. 完成当天 5 题并口述录音 3 分钟

## 日程表
| Day | 目标 | 命令 | 当天必须说清楚 |
|---|---|---|---|
| 1 | 认识项目与训练产物 | `python run.py --list` + `python run.py --module mdp --toy` | 什么是状态、动作、奖励、价值函数 |
| 2 | TD/Q-learning 基础 | `python run.py --module td_learning --toy` | TD 和 MC 的区别 |
| 3 | Advantage 直觉 | `python run.py --module advantage --toy` | Advantage 为什么能降方差 |
| 4 | GAE 直觉 | `python run.py --module gae --toy` | `lambda` 如何折中偏差/方差 |
| 5 | SFT 入门 | `python run.py --module sft --toy` | SFT 优化目标是什么 |
| 6 | DPO 入门 | `python run.py --module dpo --toy` | DPO 和 SFT 的本质区别 |
| 7 | PPO 入门 | `python run.py --module ppo --toy` | PPO 为什么比裸策略梯度稳 |
| 8 | GRPO 对比 PPO | `python run.py --module grpo --toy` | 组内相对奖励解决了什么 |
| 9 | RLHF 全流程 | `python run.py --module rlhf --toy` | RLHF = 哪三个阶段 |
| 10 | Offline RL：CQL | `python run.py --module cql --toy` | CQL 为什么“保守” |
| 11 | Offline RL：BCQ | `python run.py --module bcq --toy` | BCQ 和 CQL 处理 OOD 的方式差异 |
| 12 | VLM：BLIP2/LLaVA | `python run.py --module blip2 --toy` + `python run.py --module llava --toy` | Q-Former vs projector |
| 13 | VLM：Flamingo | `python run.py --module flamingo --toy` | 跨注意力注入的意义 |
| 14 | 预训练与工程综合 | `python run.py --module diffusion --toy` + `python run.py --module megatron --toy` | 预训练、对齐、工程优化如何串起来 |

## 第14天输出（必须完成）
1. 一份 10 分钟讲解稿（LLM + VLM + RLHF）
2. 三个对比题口述：
   `SFT vs DPO`、`PPO vs GRPO`、`BLIP2 vs LLaVA vs Flamingo`
3. 一次全量回归：
   `python scripts/smoke_test.py --allow-fail`
