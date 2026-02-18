# LLM-Core 学习项目 (LLM / VLM / RLHF)

本项目致力于通过“最小闭环”复现，帮助开发者在 14 天内掌握 LLM、VLM 与后训练（Alignment）核心流程，面向面试准备场景。

---

## 📌 快速导航 (Quick Navigation)

- **[新手学习指南 (Learning Guide)](./LEARNING_GUIDE.md)**：14天路线、通关标准与每日测验。
- **[面试核心速记 (Cheat Sheet)](./CHEAT_SHEET.md)**：显存计算公式、算法对比矩阵与系统优化总结。

---

## 🛠️ 环境与一键运行

```bash
# 激活环境
conda activate finetune

# 1. 查看支持的模块
python run.py --list

# 2. 运行模块 (推荐配合 --toy 参数快速体验)
python run.py --module sft --toy
python run.py --module grpo --toy
```

---

## 📂 项目结构 (Project Structure)

### 核心训练流程

- `pre_train/`：预训练（Megatron-LM / Diffusion / VLM 入门）
- `post_train/`：对齐与基础（RL Basics / SFT / DPO / PPO / GRPO）
- `scripts/`：面试口述稿生成、自动化冒烟测试工具。

### 模块内标准目录

- `code/`：主逻辑代码。
- `data/Models/Checkpoints/`：资源与产物（自动生成）。
- `output/`：训练指标、可视化图表。

---

## 🧪 自动回归与健康度

运行以下命令确保环境与逻辑无虞：

```bash
python scripts/smoke_test.py
```

报告输出至：`output/smoke_reports/*.json`

---

## 💡 深度学习说明

详细原理、算法推导及工程坑位说明见各子目录下的 `README.md`。例如：

- [PPO 原理](./post_train/alignment/ppo/README.md)
- [GRPO 原理](./post_train/alignment/grpo/README.md)
