# LLM-Core 学习手册 (Learning Guide)

本项目是一套面向面试准备的 LLM / VLM / RLHF 实战复现教程。

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

### 第三阶段：多模态与离线 RL (Specialized)

| Day | 目标 | 必须说清楚逻辑 |
| --- | --- | --- |
| 10 | [离线 RL (1)](./post_train/offline_rl/cql/README.md) | 为何 CQL 需要对未见动作进行处罚 |
| 11 | [离线 RL (2)](./post_train/offline_rl/bcq/README.md) | 行为克隆约束如何防止动作过冲 |
| 12 | [VLM (1)](./pre_train/vlm/blip2/README.md) | Q-Former 如何实现视觉词表对齐 |
| 13 | [VLM (2)](./pre_train/vlm/flamingo/README.md) | Cross-Attention 注入比简单拼接好在哪里 |

### 第四阶段：工程优化与综合 (Systems)

| Day | 目标 | 必须说清楚逻辑 |
| --- | --- | --- |
| 14 | [综合回顾](./pre_train/llm/megatron/README.md) | 并行策略 (TP/PP/DP) 对显存与带宽的影响 |

---

## 🎯 训练报告解读指南 (Metric Guide)

运行完成后请务必检查 `output/` 目录下的可视化产物：

1. **Loss 曲线**：SFT 应平滑下降；PPO 中 Loss 下降不代表成功，需观察 **Reward Mean** 是否稳步上升。
2. **KL 散度**：若 KL 持续走高且 Reward 停滞，说明模型正在进行 **Reward Hacking** (寻找漏洞刷分)，需要调小学习率或调大 KL 惩罚系数。
3. **Reward 分布**：观察 Reward Std。标准差过小说明模型输出单一，可能存在模式坍塌。

---

## 💡 面试 Pro Tips (典型考点)

- **Q: 为什么 PPO 后面要出个 DPO？**
  - *Ans*: PPO 显存要求极高 (4个模型)，且对超参极其敏感；DPO 极其简洁，只需 2 个模型即可达到类似效果。
- **Q: 什么是 Flash Attention？**
  - *Ans*: IO 感知优化。通过 SRAM 分块计算和重计算，消除了中间 $N \times N$ 矩阵读写，极大提升了长文本训练速度。
- **Q: LoRA 的秩 $r$ 怎么选？**
  - *Ans*: 实践中 8-16 效果极佳，盲目增加 $r$ 往往边际效益递减而增加显存开销。

---

## 📚 辅助资源

- **每日题目**：[learning/quizzes/](./learning/quizzes/) | **学霸模板**：[cards/TEMPLATE.md](./learning/cards/TEMPLATE.md)
- **批量导出**：`python scripts/export_interview_briefs.py`
