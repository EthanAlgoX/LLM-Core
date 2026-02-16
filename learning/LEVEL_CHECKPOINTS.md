# 关卡验收标准（不过关不升级）

## Level 1：RL 基础
范围：`mdp`、`td_learning`

通过标准：
1. 能解释 MDP 五元组。
2. 能解释 TD 更新目标。
3. 能口述一张曲线图含义（至少 60 秒）。

## Level 2：优势估计
范围：`advantage`、`gae`

通过标准：
1. 能解释 Advantage 定义。
2. 能解释 GAE 的 `lambda` 作用。
3. 能说明“为何降方差有助于策略学习稳定”。

## Level 3：对齐起步
范围：`sft`、`dpo`

通过标准：
1. 能说明 SFT 的目标函数。
2. 能说明 DPO 的输入数据结构（chosen/rejected）。
3. 能给出 SFT vs DPO 至少 3 点区别。

## Level 4：强化学习对齐
范围：`ppo`、`grpo`、`rlhf`

通过标准：
1. 能说明 PPO 的核心约束思想。
2. 能说明 GRPO 的组内相对比较机制。
3. 能画出 RLHF 三阶段流程图（SFT -> RM -> PPO）。

## Level 5：多模态与工程
范围：`blip2`、`llava`、`flamingo`、`diffusion`、`megatron`

通过标准：
1. 能比较三种 VLM 融合方式。
2. 能说明扩散模型基本训练/采样过程。
3. 能解释 Megatron 与 DeepSpeed 的关注点差异。

## 每关统一口述模板
1. 30秒：这个方法解决什么问题。
2. 60秒：核心机制或公式。
3. 60秒：本项目里运行结果意味着什么。
4. 30秒：工程经验和常见坑。
