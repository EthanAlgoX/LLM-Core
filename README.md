# LLM-Core 学习项目 (LLM / VLM / RLHF)

本项目致力于通过“最小闭环”复现，帮助开发者在 14 天内掌握 LLM、VLM 与后训练（Alignment）核心流程，面向面试准备场景。

---

## 🛠️ 环境与一键运行

```bash
# 激活环境
conda activate finetune

# 1. 运行模块 (推荐配合 --toy 参数快速体验)
python run.py --module sft --toy
python run.py --module ppo --toy
```

---

## 📅 14 天闯关路线图 (Roadmap)

### 第一阶段：强化学习基础 (Foundation)

| Day | 目标 | 必须说清楚逻辑 |
| --- | --- | --- |
| 1 | [项目认知](./post_train/rl_basics/mdp/README.md) | MDP五元组 (S, A, R, P, γ) 的物理含义 |
| 2 | [RL 基础](./post_train/rl_basics/td_learning/README.md) | TD 误差如何引导价值函数收敛 |
| 3 | [优势估计](./post_train/rl_basics/advantage/README.md) | 为什么 $Q(s,a) - V(s)$ 能降低训练方差 |
| 4 | [GAE 算法](./post_train/rl_basics/gae/README.md) | GAE λ 越大，方差越大但偏差越小 |

### 第二阶段：语言模型对齐 (Alignment)

| Day | 目标 | 必须说清楚逻辑 |
| --- | --- | --- |
| 5 | [SFT 入门](./post_train/alignment/sft/README.md) | 交叉熵损失 vs 负对数似然的一致性 |
| 6 | [DPO 对齐](./post_train/alignment/dpo/README.md) | 为何 DPO 不需要奖励模型也可实现对齐 |
| 7 | [PPO 算法](./post_train/alignment/ppo/README.md) | 裁剪目标函数如何防止策略剧烈抖动 |
| 8 | [GRPO 创新](./post_train/alignment/grpo/README.md) | 组内相对标准化如何抹平题目难度差异 |
| 9 | [RLHF 闭环](./post_train/alignment/rlhf/README.md) | 三阶段中数据流与模型角色的演变 |

### 第三阶段：多模态与系统 (Systems)

| Day | 目标 | 必须说清楚逻辑 |
| --- | --- | --- |
| 10 | [离线 RL](./post_train/offline_rl/README.md) | 为何 CQL 需要对未见动作进行处罚 |
| 11 | [VLM 入门](./pre_train/vlm/README.md) | Q-Former 如何实现视觉词表对齐 |
| 12 | [并行策略](./pre_train/llm/megatron/README.md) | TP/PP/DP 对显存与带宽的影响 |
| 13 | [复习自测](./learning/quizzes/) | 处理长文本时 GQA 相比 MHA 的优势 |
| 14 | **综合面试模拟** | [模拟面试工具](./scripts/qa_simulator.py) |

---

## 🧠 面试核心速记 (Cheat Sheet)

### 1. 显存与计算 (Memory & Compute)

- **权重显存**：`fp16` 占 2 Bytes/Param。7B 模型加载约需 14GB。
- **KV Cache**：$2 \times \text{layers} \times \text{heads} \times \text{dim} \times \text{precision}$ (针对每个 Token)。
- **PEFT (LoRA)**：$\Delta W = A \times B$。秩 $r$ 选 8-16 效果最佳。

### 2. 核心算法对比

| 特性 | SFT | PPO | DPO | GRPO |
| :--- | :--- | :--- | :--- | :--- |
| **显存压力** | 低 | **极高** (4模型) | 中 | 中 (省去 Critic) |
| **收敛难度** | 容易 | 难 (RL 抖动) | 较容易 | 较容易 |
| **核心场景** | 习得格式 | 安全边界 | 离线偏好 | **数学推理** |

### 3. 注意力与优化

- **变体**：Llama 3 主流使用 **GQA** (Grouped-Query)，平衡了 MHA 的精度与 MQA 的速度。
- **Flash Attention**：通过 SRAM 分块与重计算，消除 IO 瓶颈，加速长序列训练。
- **ZeRO 优化**：ZeRO-3 划分权重、梯度与优化器状态，实现极大规模参数训练。

---

## 🎯 训练监控与 Pro Tips

- **KL 散度**：过高说明模型在刷分（Reward Hacking），需调大 KL 惩罚。
- **Reward Mean**：应稳步上升。若跳变严重说明学习率过大。
- **Q: 为什么 DPO 更好用？**
  - *Ans*: 避开了复杂的 PPO 超参调优，无需维护奖励模型，显存更省且训练更稳。

---

## 📂 项目结构 (Project Structure)

- `pre_train/`：预训练（Megatron-LM / VLM 入门）
- `post_train/`：对齐与基础（RL Basics / SFT / DPO / PPO / GRPO）
- `scripts/`：面试口述稿生成、自动化冒烟测试工具。
- `learning/`：每日测验、[学霸模板](./learning/cards/TEMPLATE.md)。

---

## 🧪 自动回归与健康度

```bash
python scripts/smoke_test.py  # 报告输出至 output/smoke_reports/
```
