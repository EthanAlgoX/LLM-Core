#!/usr/bin/env python3
"""
统一学习入口（核心原理复现）。

用法示例：
  python run.py --list
  python run.py --module sft --toy
  python run.py --module grpo --toy -- --train-size 16
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModuleSpec:
    category: str
    script: str
    summary: str
    toy_args: tuple[str, ...]


MODULES: dict[str, ModuleSpec] = {
    # post_train: alignment / rl / systems
    "sft": ModuleSpec(
        "post_train/alignment",
        "post_train/alignment/sft/code/sft.py",
        "监督微调，后训练起点",
        (
            "--max-samples",
            "8",
            "--num-train-epochs",
            "0.01",
            "--batch-size",
            "1",
            "--grad-accum",
            "1",
            "--logging-steps",
            "1",
            "--save-steps",
            "1000",
            "--output-dir",
            "output/toy_sft",
        ),
    ),
    "dpo": ModuleSpec(
        "post_train/alignment",
        "post_train/alignment/dpo/code/dpo.py",
        "直接偏好优化",
        (
            "--max-samples",
            "8",
            "--num-train-epochs",
            "0.01",
            "--batch-size",
            "1",
            "--grad-accum",
            "1",
            "--logging-steps",
            "1",
            "--save-steps",
            "1000",
            "--output-dir",
            "output/toy_dpo",
        ),
    ),
    "grpo": ModuleSpec(
        "post_train/alignment",
        "post_train/alignment/grpo/code/grpo_demo.py",
        "组相对策略优化",
        (
            "--train-size",
            "4",
            "--num-train-epochs",
            "0.05",
            "--num-generations",
            "2",
            "--generation-batch-size",
            "2",
            "--max-completion-length",
            "32",
            "--logging-steps",
            "1",
            "--save-steps",
            "100",
            "--output-dir",
            "output/toy_grpo",
        ),
    ),
    "ppo": ModuleSpec(
        "post_train/alignment",
        "post_train/alignment/ppo/code/ppo.py",
        "PPO 对齐训练",
        (
            "--reward-model",
            "Qwen/Qwen3-0.6B",
            "--max-samples",
            "8",
            "--num-train-epochs",
            "0.01",
            "--batch-size",
            "1",
            "--grad-accum",
            "1",
            "--logging-steps",
            "1",
            "--save-steps",
            "1000",
            "--output-dir",
            "output/toy_ppo",
        ),
    ),
    "policy_gradient": ModuleSpec(
        "post_train/alignment",
        "post_train/alignment/policy_gradient/code/policy_gradient.py",
        "策略梯度对齐训练",
        (
            "--reward-model",
            "Qwen/Qwen3-0.6B",
            "--max-samples",
            "8",
            "--num-train-epochs",
            "0.01",
            "--batch-size",
            "1",
            "--grad-accum",
            "1",
            "--logging-steps",
            "1",
            "--save-steps",
            "1000",
            "--output-dir",
            "output/toy_policy_gradient",
        ),
    ),
    "actor_critic": ModuleSpec(
        "post_train/alignment",
        "post_train/alignment/actor_critic/code/actor_critic.py",
        "Actor-Critic 对齐训练",
        (
            "--reward-model",
            "Qwen/Qwen3-0.6B",
            "--max-samples",
            "8",
            "--num-train-epochs",
            "0.01",
            "--batch-size",
            "1",
            "--grad-accum",
            "1",
            "--logging-steps",
            "1",
            "--save-steps",
            "1000",
            "--output-dir",
            "output/toy_actor_critic",
        ),
    ),
    "rlhf": ModuleSpec(
        "post_train/alignment",
        "post_train/alignment/rlhf/code/rlhf.py",
        "RLHF 闭环训练",
        (
            "--reward-model",
            "Qwen/Qwen3-0.6B",
            "--max-samples",
            "8",
            "--num-train-epochs",
            "0.01",
            "--batch-size",
            "1",
            "--grad-accum",
            "1",
            "--logging-steps",
            "1",
            "--save-steps",
            "1000",
            "--output-dir",
            "output/toy_rlhf",
        ),
    ),
    "mdp": ModuleSpec(
        "post_train/rl_basics",
        "post_train/rl_basics/mdp/code/mdp.py",
        "MDP 值迭代",
        (
            "--max-iters",
            "20",
            "--save-every-iters",
            "10",
            "--output-dir",
            "output/toy_mdp",
        ),
    ),
    "td_learning": ModuleSpec(
        "post_train/rl_basics",
        "post_train/rl_basics/td_learning/code/td_learning.py",
        "TD/Q-learning 基础",
        (
            "--episodes",
            "20",
            "--max-steps-per-episode",
            "20",
            "--log-every-episodes",
            "5",
            "--save-every-episodes",
            "10",
            "--output-dir",
            "output/toy_td_learning",
        ),
    ),
    "gae": ModuleSpec(
        "post_train/rl_basics",
        "post_train/rl_basics/gae/code/gae.py",
        "GAE 优势估计",
        (
            "--iterations",
            "4",
            "--episodes-per-iter",
            "8",
            "--log-every",
            "1",
            "--save-every",
            "2",
            "--output-dir",
            "output/toy_gae",
        ),
    ),
    "advantage": ModuleSpec(
        "post_train/rl_basics",
        "post_train/rl_basics/advantage/code/advantage.py",
        "优势函数估计对比",
        (
            "--iterations",
            "4",
            "--episodes-per-iter",
            "8",
            "--log-every",
            "1",
            "--save-every",
            "2",
            "--output-dir",
            "output/toy_advantage",
        ),
    ),
    "cql": ModuleSpec(
        "post_train/offline_rl",
        "post_train/offline_rl/cql/code/cql.py",
        "离线 RL：CQL",
        (
            "--dataset-episodes",
            "24",
            "--updates",
            "24",
            "--eval-every",
            "6",
            "--log-every",
            "6",
            "--save-every",
            "12",
            "--output-dir",
            "output/toy_cql",
        ),
    ),
    "bcq": ModuleSpec(
        "post_train/offline_rl",
        "post_train/offline_rl/bcq/code/bcq.py",
        "离线 RL：BCQ",
        (
            "--dataset-episodes",
            "24",
            "--updates",
            "24",
            "--eval-every",
            "6",
            "--log-every",
            "6",
            "--save-every",
            "12",
            "--output-dir",
            "output/toy_bcq",
        ),
    ),
    "deepspeed": ModuleSpec(
        "post_train/systems",
        "post_train/systems/deepspeed/code/deepspeed.py",
        "训练系统优化：DeepSpeed",
        (
            "--steps",
            "12",
            "--log-every",
            "3",
            "--save-every",
            "6",
            "--output-dir",
            "output/toy_deepspeed",
        ),
    ),
    "cuda": ModuleSpec(
        "post_train/systems",
        "post_train/systems/cuda/code/cuda.py",
        "CUDA 与吞吐观察",
        (
            "--benchmark-iters",
            "3",
            "--warmup-iters",
            "1",
            "--train-steps",
            "12",
            "--log-every",
            "3",
            "--output-dir",
            "output/toy_cuda",
        ),
    ),
    "mixed_precision": ModuleSpec(
        "post_train/systems",
        "post_train/systems/mixed_precision/code/mixed_precision.py",
        "混合精度训练",
        (
            "--steps",
            "20",
            "--log-every",
            "5",
            "--save-every",
            "10",
            "--output-dir",
            "output/toy_mixed_precision",
        ),
    ),
    # pre_train
    "diffusion": ModuleSpec(
        "pre_train/generation",
        "pre_train/generation/diffusion/code/diffusion.py",
        "扩散模型基础",
        (
            "--epochs",
            "1",
            "--steps-per-epoch",
            "10",
            "--logging-steps",
            "2",
            "--save-every-epochs",
            "1",
            "--num-vis-samples",
            "4",
            "--output-dir",
            "output/toy_diffusion",
        ),
    ),
    "dit": ModuleSpec(
        "pre_train/generation",
        "pre_train/generation/dit/code/dit.py",
        "DiT 训练示例",
        (
            "--epochs",
            "1",
            "--steps-per-epoch",
            "10",
            "--logging-steps",
            "2",
            "--save-every-epochs",
            "1",
            "--num-vis-samples",
            "4",
            "--output-dir",
            "output/toy_dit",
        ),
    ),
    "blip2": ModuleSpec(
        "pre_train/vlm",
        "pre_train/vlm/blip2/code/blip2.py",
        "BLIP2 推理示例",
        ("--dry-run", "--output-dir", "output/toy_blip2"),
    ),
    "llava": ModuleSpec(
        "pre_train/vlm",
        "pre_train/vlm/llava/code/llava.py",
        "LLaVA 推理示例",
        ("--dry-run", "--output-dir", "output/toy_llava"),
    ),
    "flamingo": ModuleSpec(
        "pre_train/vlm",
        "pre_train/vlm/flamingo/code/flamingo.py",
        "Flamingo 推理示例",
        ("--dry-run", "--output-dir", "output/toy_flamingo"),
    ),
    "megatron": ModuleSpec(
        "pre_train/llm",
        "pre_train/llm/megatron/code/megatron.py",
        "Megatron 并行训练示例",
        (
            "--steps",
            "20",
            "--log-every",
            "5",
            "--save-every",
            "10",
            "--output-dir",
            "output/toy_megatron",
        ),
    ),
}


def build_default_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified learning entrypoint for core principle demos.")
    parser.add_argument("--list", action="store_true", help="列出所有可用模块。")
    parser.add_argument("--module", choices=sorted(MODULES.keys()), help="要运行的模块。")
    parser.add_argument("--toy", action="store_true", help="启用最小量化参数（快速打通流程）。")
    parser.add_argument("--dry-run-cmd", action="store_true", help="仅打印命令，不执行。")
    parser.add_argument(
        "--extra",
        default="",
        help='附加参数字符串，例如 --extra="--learning-rate 1e-5 --seed 7"',
    )
    return parser.parse_known_args()[0]


def print_module_list() -> None:
    print("Available modules:")
    for name in sorted(MODULES.keys()):
        spec = MODULES[name]
        print(f"- {name:16s} | {spec.category:20s} | {spec.summary}")


def build_command(module: str, toy: bool, extra: str) -> list[str]:
    spec = MODULES[module]
    script_path = ROOT / spec.script
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = [sys.executable, str(script_path)]
    if toy:
        cmd.extend(spec.toy_args)
    if extra.strip():
        cmd.extend(shlex.split(extra))
    return cmd


def main() -> None:
    args = build_default_args()

    if args.list:
        print_module_list()
        return

    if not args.module:
        raise ValueError("`--module` is required unless `--list` is used.")

    spec = MODULES[args.module]
    cmd = build_command(module=args.module, toy=args.toy, extra=args.extra)
    print(f"Module   : {args.module}")
    print(f"Category : {spec.category}")
    print(f"Script   : {spec.script}")
    print(f"Mode     : {'toy' if args.toy else 'default'}")
    print(f"Command  : {' '.join(shlex.quote(x) for x in cmd)}")

    if args.dry_run_cmd:
        return

    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
