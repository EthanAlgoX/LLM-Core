# LLM-Core 学习路径

> 一句话通俗理解：按目标选路线，避免“全学一遍却不会用”。

## 路线 A：面试冲刺（2-3 周）

1. [Transformer Core](../modules/02_architecture/llm/llm.md)
2. [Attention Mechanisms](../modules/02_architecture/llm/attention.md)
3. [SFT](../modules/03_alignment/sft/sft.md)
4. [DPO](../modules/03_alignment/dpo/dpo.md)
5. [PPO](../modules/03_alignment/ppo/ppo.md)
6. [Inference Systems](../modules/05_engineering/inference/inference.md)
7. [ChatGPT / InstructGPT](../modules/07_classic_models/chatgpt/chatgpt.md)

目标产出：能完整回答“架构-对齐-工程-案例”四类高频问题。

## 路线 B：工程落地（4-6 周）

1. [Transformer Core](../modules/02_architecture/llm/llm.md)
2. [Generation & Decoding](../modules/02_architecture/generation/generation.md)
3. [CUDA](../modules/05_engineering/cuda/cuda.md)
4. [Mixed Precision](../modules/05_engineering/mixed_precision/mixed_precision.md)
5. [Megatron](../modules/05_engineering/megatron/megatron.md)
6. [DeepSpeed](../modules/05_engineering/deepspeed/deepspeed.md)
7. [Inference Systems](../modules/05_engineering/inference/inference.md)
8. [PEFT](../modules/03_alignment/peft/peft.md)

目标产出：能解释并实施“训练并行 + 推理优化 + 轻量微调”的最小工程闭环。

## 路线 C：研究进阶（6-8 周）

1. [RL Foundation](../modules/01_foundation_rl/01_foundation_rl.md)
2. [RLHF](../modules/03_alignment/rlhf/rlhf.md)
3. [GRPO](../modules/03_alignment/grpo/grpo.md)
4. [Data Synthesis](../modules/03_alignment/data_synthesis/data_synthesis.md)
5. [Offline RL](../modules/04_advanced_topics/offline_rl/offline_rl.md)
6. [VLM 总览](../modules/02_architecture/vlm/vlm.md)
7. [DeepSeek-R1](../modules/07_classic_models/deepseek_r1/deepseek_r1.md)

目标产出：能独立完成“方法比较-实验设计-风险评估”的研究型笔记。

## 使用建议

- 每条路线都先读总览页，再进子主题细节页。
- 每读完一个主题，补一段“关键公式 + 关键步骤代码”的复述。
- 每周回看一次 [导航索引](./NAVIGATION.md) 调整学习顺序。
