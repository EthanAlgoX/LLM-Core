#!/usr/bin/env python3
"""
GRPO 学习入口（主流程 + 可视化）。

说明：
1) 主脚本只保留学习路径。
2) 训练与可视化实现位于 `grpo_demo_core.py`。
"""

from __future__ import annotations

from grpo_demo_core import main as core_main


LEARNING_STEPS = [
    "步骤 1：准备训练样本与奖励函数",
    "步骤 2：构建 GRPO 训练配置",
    "步骤 3：执行 GRPO 训练",
    "步骤 4：保存模型与 checkpoint",
    "步骤 5：导出可视化（loss/reward/learning rate）",
]


def print_learning_steps() -> None:
    """打印学习主流程。"""
    print("=== GRPO 主流程（学习版）===", flush=True)
    for i, step in enumerate(LEARNING_STEPS, start=1):
        print(f"{i}. {step}", flush=True)
    print("===========================", flush=True)


def run() -> None:
    """运行 GRPO 主流程，并生成可视化结果。"""
    print_learning_steps()
    core_main()


if __name__ == "__main__":
    run()
