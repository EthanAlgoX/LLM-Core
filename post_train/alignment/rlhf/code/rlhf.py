#!/usr/bin/env python3
"""
RLHF 学习入口（主流程 + 可视化）。

说明：
1) 主脚本只保留学习路径。
2) 训练与可视化实现位于 `rlhf_core.py`。
"""

from __future__ import annotations

from rlhf_core import main as core_main


LEARNING_STEPS = [
    "步骤 1：准备策略模型与奖励模型参数",
    "步骤 2：自动检测设备与精度",
    "步骤 3：执行 RLHF 强化学习训练（PPO 近似实现）",
    "步骤 4：整理模型与 checkpoint 产物",
    "步骤 5：导出可视化（loss/reward/kl/lr）",
]


def print_learning_steps() -> None:
    """打印学习主流程。"""
    print("=== RLHF 主流程（学习版）===", flush=True)
    for i, step in enumerate(LEARNING_STEPS, start=1):
        print(f"{i}. {step}", flush=True)
    print("===========================", flush=True)


def run() -> None:
    """运行 RLHF 主流程，并生成可视化结果。"""
    print_learning_steps()
    core_main()


if __name__ == "__main__":
    run()
