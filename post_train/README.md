# Post-Train 学习总览

## 定位与分类
`post_train` 用于学习 LLM/VLM 的后训练阶段，重点是对齐与强化学习。

当前模块按学习主题分为四类：

1. 监督与偏好对齐：`sft`、`dpo`、`grpo`、`ppo`、`policy_gradient`、`actor_critic`、`rlhf`
2. RL 基础理论：`mdp`、`td_learning`、`gae`、`advantage`
3. 离线强化学习：`cql`、`bcq`
4. 工程加速专题：`deepspeed`、`cuda`、`mixed_precision`

## 分类区别（学习路径）
1. `SFT -> DPO -> PPO/GRPO/RLHF`：从监督学习逐步过渡到偏好优化与在线强化学习。
2. `MDP/TD/GAE/Advantage`：先掌握 RL 数学基础，再理解 LLM 对齐中的 advantage 与 policy update。
3. `CQL/BCQ`：聚焦离线数据训练，理解“只能用静态数据”时的策略约束。
4. `DeepSpeed/CUDA/混合精度`：关注训练速度、显存和吞吐，不改变算法目标。

## 通用目录规范（每个模块）
- `code/`: 单文件可运行脚本
- `data/`: 训练或示例数据
- `models/`: 最终模型文件
- `checkpoints/`: 训练中间检查点
- `output/`: 指标、曲线图、配置快照

## 通用运行方式
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/<module>
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/<module>.py
```

> 注：部分模块脚本名不同，例如 `grpo` 使用 `python code/grpo_demo.py`。

## 统一入口（推荐）
在项目根目录可使用统一入口，便于面试时快速切换模块：

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune

python run.py --list
python run.py --module sft --toy
python run.py --module grpo --toy
```
