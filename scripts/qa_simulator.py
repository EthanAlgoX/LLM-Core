#!/usr/bin/env python3
"""
模拟面试工具：随机抽取模块并提出 3 个高频面试问题。
"""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run as interview_run

# 核心题库：针对每个模块设计 3 个具有挑战性的面试题
QUESTIONS: dict[str, list[str]] = {
    "sft": [
        "1. 为什么 SFT 阶段通常使用较大的 Batch Size 和较小的 Learning Rate？",
        "2. 如果 SFT 后的模型出现了明显的模式坍塌（Over-optimization），你会首先排查哪个环节？",
        "3. Teacher Forcing 在 SFT 训练和推理时的区别是什么？"
    ],
    "ppo": [
        "1. PPO 的 Clipped Objective 到底解决了什么问题？为什么直接用原始 Ratio 不行？",
        "2. KL Penalty 在 PPO 中的作用是什么？如果去掉会发生什么？",
        "3. 在 LLM 的 PPO 训练中，Critic 模型的初始化通常如何选择？"
    ],
    "grpo": [
        "1. DeepSeek 为什么要在大规模训练中用 GRPO 替代 PPO？核心省在了哪里？",
        "2. 如果组内采样数量 (num_generations) 太小，对 GRPO 的优势估计会有什么影响？",
        "3. GRPO 如何处理组内所有样本得分都极高或极低的情况？"
    ],
    "dpo": [
        "1. DPO 的灵活性体现在哪？为什么它不需要一个显式的奖励模型？",
        "2. DPO 中的 beta 参数控制什么？调大或调小分别代表什么倾向？",
        "3. 什么样的偏好数据会让 DPO 的训练变得极度不稳定？"
    ],
    "rlhf": [
        "1. 描述一下 RLHF 的三阶段流程，并说明为什么每一步都是必须的。",
        "2. 奖励模型 (RM) 的训练数据通常是如何标注的？为什么要用 Pairwise 而不是直接打分？",
        "3. 什么是 Reward Hacking？在项目实践中你如何缓解它？"
    ],
    "deepspeed": [
        "1. 详细描述 ZeRO-1, ZeRO-2, ZeRO-3 的区别，它们分别切分了什么？",
        "2. 在 ZeRO-3 中，参数是如何在各个 GPU 之间“流动”的？",
        "3. DeepSpeed 的 Offload 机制（CPU/NVMe）在什么场景下会拖累训练速度？"
    ],
    "cuda": [
        "1. 描述 GPU 的内存体系结构（Global, Shared, Registers），并说明 Shared Memory 的优化作用。",
        "2. 什么是 Kernel Launch Overhead？如何通过算子融合 (Operator Fusion) 来缓解？",
        "3. 如果你的训练速度受限于 IO (Memory Bound)，你会采取哪些手段？"
    ],
    "mixed_precision": [
        "1. 既然有了 FP16，为什么还需要维护一份 FP32 的 Master Weights？",
        "2. 描述 Loss Scaling 的工作机制，以及它如何防止梯度下溢。",
        "3. BF16 相比 FP16 在深度学习训练中的核心优势是什么？"
    ],
    "llava": [
        "1. LLaVA 的 Projector 层起到了什么作用？为什么不直接微调 LLM？",
        "2. 在 LLaVA 的两阶段训练中，第一阶段（Alignment）的主要目标是什么？",
        "3. 多模态指令微调数据和纯文本指令微调数据在格式上有何异同？"
    ]
}

def main():
    parser = argparse.ArgumentParser(description="AI Interview Simulator")
    parser.add_argument("--module", choices=sorted(interview_run.MODULES.keys()), help="指定模拟面试的模块")
    parser.add_argument("--all", action="store_true", help="列出所有题库")
    args = parser.parse_args()

    if args.all:
        for mod, q_list in QUESTIONS.items():
            print(f"\n### [{mod.upper()}]")
            for q in q_list:
                print(f"  - {q}")
        return

    module = args.module
    if not module:
        # 排除没有题目的模块，随机选一个
        available = [m for m in interview_run.MODULES.keys() if m in QUESTIONS]
        module = random.choice(available)

    print(f"\n" + "="*50)
    print(f"🚀 进入【{module.upper()}】模拟面试环节")
    print("="*50)
    
    q_list = QUESTIONS.get(module, ["(该模块暂无详细题库，请结合 README 自行准备)"])
    for i, q in enumerate(q_list, 1):
        print(f"\nQ{i}: {q}")
        input(f"  [按回车键查看口述建议...]")
        # 这里可以扩展从 interview_briefs 读入建议，目前引导用户看 README
        print(f"  💡 建议：参考 post_train/alignment/{module}/README.md 中的“核心原理”和“工程经验”部分。")

    print("\n" + "="*50)
    print(f"🏁 练习结束！你可以使用 `python scripts/interview_brief.py --module {module}` 查看详细口述稿。")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
